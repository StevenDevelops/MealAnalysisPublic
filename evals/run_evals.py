"""
    File summary:
    Calls the OpenAI API for each model, parses the outputs, and writes results to a CSV file.
    Save the models outputs from each model. These are outputs that will be scored and evaluated. 
"""
import csv
import time
import os
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Literal, Any, Mapping

# Imports to make the OpenAI Calls
import base64
import random
from openai import OpenAI
from dotenv import load_dotenv
import yaml

# Imports to help parse and sanitize json outputs from the model
import re

ALLOWED_IMAGE_EXT = ".jpeg"
DEFAULT_CONFIG_PATH = os.path.join("evals", "agent_models.yaml")
AGENT_TO_IO: Dict[str, str] = {
    "inputGuardrail": "vision",
    "mealAnalysis": "vision",
    "outputGuardrail": "text",
}
AGENTS_ALLOWING_EMPTY_MODELS = {"inputGuardrail", "outputGuardrail"}

"""CLASS DEFINITIONS"""
# Each Sample is a single test input -> expected output, both local files
# The Id is derived from the filename, used to match local image <-> json files
@dataclass(frozen=True) # @dataclass for boilerplate, frozen=True for immutability
class Sample:
    id: str
    img_path: str
    json_path: str
    gt: Dict  # parsed ground truth JSON

AgentName = Literal["inputGuardrail", "mealAnalysis", "outputGuardrail"]
AgentIO = Literal["vision", "text"]

# Each AgentConfig represents the configuration for one agent
# including its name, IO type, and list of models to evaluate
@dataclass(frozen=True)
class AgentConfig:
    name: AgentName
    io: AgentIO
    models: List[str]

# Each EvalJob represents a single evaluation run of one agent 
# (with specific model) on one sample
@dataclass(frozen=True)
class EvalJob:
    agent: AgentName
    io: AgentIO
    model: str
    sample: Sample

"""METHODS FOR LOADING DATASET FROM LOCAL FILES"""
def _require_dir_nonempty(path: str, name: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} directory not found: {path}")
    entries = [f for f in os.listdir(path) if not f.startswith(".")]
    if len(entries) == 0:
        raise RuntimeError(f"{name} directory is empty: {path}")


def _list_ids_in_images(images_dir: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Returns: (tuple of...)
        - mapping id -> absolute image path (only .jpeg)
        - list of unexpected image files (wrong extension)
    """
    id_to_img: Dict[str, str] = {}
    unexpected: List[str] = []

    for fn in os.listdir(images_dir):
        if fn.startswith("."): # Ignore hidden files like .DS_Store, .gitkeep
            continue
        base, ext = os.path.splitext(fn)
        ext_lower = ext.lower()

        if ext_lower == ALLOWED_IMAGE_EXT:
            id_to_img[base] = os.path.join(images_dir, fn)
        else:
            # any non-.jpeg file in images/ is unexpected for this project
            unexpected.append(fn)

    return id_to_img, unexpected


def _list_ids_in_json(json_dir: str) -> Dict[str, str]:
    """Returns mapping id -> absolute json path for *.json files."""
    id_to_json: Dict[str, str] = {}

    for fn in os.listdir(json_dir):
        if fn.startswith("."): # Ignore hidden files like .DS_Store, .gitkeep
            continue
        base, ext = os.path.splitext(fn)
        if ext.lower() == ".json":
            id_to_json[base] = os.path.join(json_dir, fn)

    return id_to_json


def load_dataset(root: str) -> List[Sample]:
    data_dir = os.path.join(root, "data")
    images_dir = os.path.join(data_dir, "images")
    json_dir = os.path.join(data_dir, "json-files")

    _require_dir_nonempty(images_dir, "images")
    _require_dir_nonempty(json_dir, "json-files")

    # Get a mapping of id -> image path,
    # Along with any unexpected files in images/
    id_to_img, unexpected_imgs = _list_ids_in_images(images_dir)
    id_to_json = _list_ids_in_json(json_dir)

    # Strict mode: error if any unexpected image extension exists
    if unexpected_imgs:
        raise RuntimeError(
            f"Unexpected non-{ALLOWED_IMAGE_EXT} files found in images/: "
            f"{unexpected_imgs[:20]}{'...' if len(unexpected_imgs) > 20 else ''}. "
            f"Fix/remove these files or relax extension rules."
        )

    # Combine IDs from both sources to find missing items into sorted list
    all_ids = sorted(set(id_to_img.keys()) | set(id_to_json.keys()))

    warnings_missing_image: List[str] = []
    warnings_missing_json: List[str] = []

    samples: List[Sample] = []

    for sid in all_ids:
        img_path = id_to_img.get(sid)
        json_path = id_to_json.get(sid)

        if img_path is None:
            warnings_missing_image.append(sid)
            continue
        if json_path is None:
            warnings_missing_json.append(sid)
            continue

        # both image and json exist -> load ground truth json
        with open(json_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        samples.append(Sample(id=sid, img_path=img_path, json_path=json_path, gt=gt))

    # Print warnings (do not stop execution)
    if warnings_missing_image or warnings_missing_json:
        print("WARNING: Some samples are missing image or JSON and will be excluded from evals.")
        if warnings_missing_image:
            print(f"- Missing image (.jpeg) for {len(warnings_missing_image)} id(s):")
            for sid in warnings_missing_image[:50]:
                print(f"  {sid}")
            if len(warnings_missing_image) > 50:
                print("  ...")
        if warnings_missing_json:
            print(f"- Missing JSON (.json) for {len(warnings_missing_json)} id(s):")
            for sid in warnings_missing_json[:50]:
                print(f"  {sid}")
            if len(warnings_missing_json) > 50:
                print("  ...")

    if not samples:
        raise RuntimeError("No valid (image,json) pairs found after excluding missing items.")

    return samples


"""METHODS FOR LOADING AGENT CONFIGURATIONS, AND BUILDING EVALUTION JOBS"""
def build_jobs(samples: List[Sample], agents: List[AgentConfig]) -> List[EvalJob]:
    """
    Builds the Cartesian product of:
    (agent config) × (models for that agent) × (samples)
    No API calls here — just prepares a list of work items.
    """
    jobs: List[EvalJob] = []
    for agent_cfg in agents:
        for model in agent_cfg.models:
            for s in samples:
                jobs.append(EvalJob(
                    agent=agent_cfg.name,
                    io=agent_cfg.io,
                    model=model,
                    sample=s,
                ))
    return jobs

def dry_run_print_jobs(jobs: List[EvalJob], n: int = 10) -> None:
    print(f"Planned {len(jobs)} eval jobs total.")
    print("We'll print a few jobs so we know the harness is correct before we run all the API calls.")
    for j in jobs[:n]:
        print(f"- agent={j.agent} io={j.io} model={j.model} sample_id={j.sample.id}")


def load_agent_configs(config_path: str) -> List[AgentConfig]:
    """Load agent/model config from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Agent config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if isinstance(raw, dict):
        items = raw.get("agents")
    elif isinstance(raw, list):
        items = raw
    else:
        items = None

    if not isinstance(items, list) or len(items) == 0:
        raise RuntimeError(
            "Invalid agent config YAML. Expected either a top-level list or "
            "a dict with key 'agents' as a non-empty list."
        )

    agents: List[AgentConfig] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid agent config entry at index {idx}: expected object")

        name_raw = item.get("name")
        io_raw = item.get("io")
        models_raw = item.get("models")

        if not isinstance(name_raw, str) or name_raw not in AGENT_TO_IO:
            raise RuntimeError(
                f"Invalid agent name at index {idx}: {name_raw!r}. "
                f"Allowed: {sorted(AGENT_TO_IO.keys())}"
            )
        name: AgentName = name_raw  # type: ignore[assignment]

        expected_io = AGENT_TO_IO[name]
        if io_raw is None:
            io: AgentIO = expected_io
        elif isinstance(io_raw, str) and io_raw in {"vision", "text"}:
            if io_raw != expected_io:
                raise RuntimeError(
                    f"Invalid io for agent {name!r} at index {idx}: {io_raw!r}. "
                    f"Expected: {expected_io!r}"
                )
            io = io_raw  # type: ignore[assignment]
        else:
            raise RuntimeError(
                f"Invalid io at index {idx}: {io_raw!r}. Allowed: 'vision' or 'text'"
            )

        # Allow optional skipping of guardrail agents by setting models: []
        # or omitting models entirely in YAML.
        if models_raw is None:
            models_raw = []
        if not isinstance(models_raw, list):
            raise RuntimeError(f"Invalid models list for agent {name!r} at index {idx}")
        if len(models_raw) == 0 and name not in AGENTS_ALLOWING_EMPTY_MODELS:
            raise RuntimeError(
                f"Invalid models list for agent {name!r} at index {idx}: "
                "provide at least one model for this agent."
            )

        models: List[str] = []
        for m in models_raw:
            if not isinstance(m, str) or not m.strip():
                raise RuntimeError(
                    f"Invalid model entry for agent {name!r} at index {idx}: {m!r}"
                )
            models.append(m.strip())

        # Preserve order while removing duplicates.
        deduped_models = list(dict.fromkeys(models))
        agents.append(AgentConfig(name=name, io=io, models=deduped_models))

    return agents

# ---- CSV schema ----
CSV_COLUMNS = [
    # identity
    "id", 
    "title",
    "agent", "io", "model",
    # paths (useful for debugging)
    "img_path", "json_path",
    # timing / usage (filled later when we call OpenAI)
    "latency_ms", "input_tokens", "output_tokens",
    # common eval fields (filled later by scoring)
    "parse_ok", "error", "raw_output_snippet",

    # inputGuardrail / safety booleans
    "is_food", "no_pii", "no_humans", "no_captcha",

    # mealAnalysis fields
    "recommendation", "meal_title", "meal_description", "guidance_message",
    "calories", "carbohydrates", "fats", "proteins", "ingredients_json",

    # outputGuardrail / safetyChecks booleans (dataset keys)
    "no_insuline_guidance",
    "no_carb_content",
    "no_emotional_or_judgmental_language",
    "no_risky_ingredient_substitutions",
    "no_treatment_recommendation",
    "no_medical_diagnosis",
]

"""METHODS FOR RUNNING AGENTS AND OUTPUT RESULTS TO CSV"""
def _response_params_for_model(model: str) -> Dict[str, Any]:
    model_lc = (model or "").strip().lower()

    # GPT-5+ models on Responses API often do not support `temperature`.
    # To reduce variability and keep evaluations as deterministic as possible,
    # we use fixed reasoning effort for GPT-5 family models instead.
    if model_lc.startswith("gpt-5.2-pro"):
        return {"reasoning": {"effort": "medium"}}
    if model_lc.startswith("gpt-5.2"):
        return {"reasoning": {"effort": "none"}}
    if model_lc.startswith("gpt-5"):
        return {"reasoning": {"effort": "minimal"}}
    return {"temperature": 0}


def _create_response(client: OpenAI, model: str, input_payload: Any):
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": input_payload,
    }
    kwargs.update(_response_params_for_model(model))
    return client.responses.create(**kwargs)


def run_agent(
    job: EvalJob,
    client: OpenAI,
    image_b64_cache: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], float, int, int, bool, str, str]:
    """
    Executes one agent call using OpenAI Responses API with model-aware
    deterministic request settings.

    Returns:
    (agent_output, latency_ms, input_tokens, output_tokens, parse_ok, error_message, raw_output)
    """
    t0 = time.time()

    try:
        # ---- Build input message based on agent type ----
        if job.io == "vision":
            # Read image from cache when available to avoid repeated base64 work.
            img_b64 = ""
            if image_b64_cache is not None:
                img_b64 = image_b64_cache.get(job.sample.id, "")
            if not img_b64:
                img_b64 = image_to_base64(job.sample.img_path)
            image_part = {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"}

            if job.agent == "inputGuardrail":
                prompt = (
                    "You are an input guardrails classifier.\n"
                    "Return ONLY valid JSON with keys: is_food, no_pii, no_humans, no_captcha.\n"
                    "Each value must be a boolean. No extra text.\n"
                    "Do not wrap the JSON in ``` fences."
                )
                text_part = {"type": "input_text", "text": prompt}

            elif job.agent == "mealAnalysis":
                prompt = (
                    "You are a meal analysis system. Return ONLY valid JSON with:\n"
                    "- is_food (boolean)\n"
                    "- recommendation (green|yellow|orange|red)\n"
                    "- meal_title, meal_description, guidance_message (strings; neutral; no medical claims)\n"
                    "- macros: {calories, carbohydrates, fats, proteins} (numbers)\n"
                    "- ingredients: [{name, impact}] where impact in (green|yellow|orange|red)\n"
                    "No extra text outside JSON.\n"
                    "Do not wrap the JSON in ``` fences."
                )
                text_part = {"type": "input_text", "text": prompt}
            else:
                raise ValueError(f"Unexpected vision agent: {job.agent}")

            resp = _create_response(
                client=client,
                model=job.model,
                input_payload=[{"role": "user", "content": [text_part, image_part]}],
            )
        
        else:
            # text-only agent: outputGuardrail
            gt = job.sample.gt or {}

            # Use GROUND TRUTH of mealAnalysis text as the input to safetyChecks
            ma = gt.get("mealAnalysis", {}) or {}
            meal_title = ma.get("meal_title", "")
            meal_desc = ma.get("meal_description", "")
            guidance = ma.get("guidance_message", "")

            text_to_check = "\n".join([x for x in [meal_title, meal_desc, guidance] if x]).strip()
            if not text_to_check:
                text_to_check = "No mealAnalysis text available in ground truth."

            prompt = (
                "You are an output safety guardrails classifier.\n"
                "Analyze the TEXT and return ONLY valid JSON with these EXACT boolean keys:\n"
                "- no_insuline_guidance\n"
                "- no_carb_content\n"
                "- no_emotional_or_judgmental_language\n"
                "- no_risky_ingredient_substitutions\n"
                "- no_treatment_recommendation\n"
                "- no_medical_diagnosis\n"
                "No extra keys. No extra text. Do not wrap JSON in ``` fences.\n\n"
                f"TEXT:\n{text_to_check}"
            )

            resp = _create_response(
                client=client,
                model=job.model,
                input_payload=prompt,
            )

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0

        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        raw = (resp.output_text or "").strip()

        # ---- Parse JSON ----
        try:
            agent_output = parse_json_loose(raw)
            return agent_output, latency_ms, input_tokens, output_tokens, True, "", raw
        
        except Exception as je:
            # JSON parse failure — return raw text in error
            return {}, latency_ms, input_tokens, output_tokens, False, f"json_parse_error: {je}", raw

    except Exception as e:
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        return {}, latency_ms, 0, 0, False, f"openai_error: {repr(e)}", ""

# json sanitizer to strip code fences and other common wrappers
# that might cause json parsing to fail (clean json output from model)
def parse_json_loose(raw: str) -> dict:
    s = (raw or "").strip()

    # Strip ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)   # remove opening fence + optional "json"
        s = re.sub(r"\s*```$", "", s)                 # remove closing fence
        s = s.strip()

    # If model wrote extra text, extract first JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    return json.loads(s)

def _project_relpath(path: str, root: str) -> str:
    """
    Return project-relative path when possible.
    Falls back to original path if relative conversion is not valid.
    """
    try:
        rel = os.path.relpath(path, start=root)
        if rel.startswith(".."):
            return path
        return rel
    except Exception:
        return path

"""
Convert (job + agent_output) into a single flat CSV row.
Includes scoring, tokens, real latency.
"""
def make_csv_row(
    job: EvalJob,
    agent_output: Dict[str, Any],
    root: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    parse_ok: bool,
    error: str,
    raw_output: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: "" for k in CSV_COLUMNS}

    row["id"] = job.sample.id
    row["title"] = job.sample.gt.get("title", "")
    row["agent"] = job.agent
    row["io"] = job.io
    row["model"] = job.model
    row["img_path"] = _project_relpath(job.sample.img_path, root)
    row["json_path"] = _project_relpath(job.sample.json_path, root)

    row["latency_ms"] = round(latency_ms, 2)
    row["input_tokens"] = input_tokens
    row["output_tokens"] = output_tokens

    row["parse_ok"] = 1 if parse_ok else 0
    row["error"] = error
    row["raw_output_snippet"] = raw_output[:800] # store a truncated raw output for debugging

    # inputGuardrail fields
    for k in ["is_food", "no_pii", "no_humans", "no_captcha"]:
        if k in agent_output:
            row[k] = agent_output[k]

    # mealAnalysis fields
    if job.agent == "mealAnalysis":
        row["recommendation"] = agent_output.get("recommendation", "")
        row["meal_title"] = agent_output.get("meal_title", "")
        row["meal_description"] = agent_output.get("meal_description", "")
        row["guidance_message"] = agent_output.get("guidance_message", "")

        macros = agent_output.get("macros", {}) or {}
        row["calories"] = macros.get("calories", "")
        row["carbohydrates"] = macros.get("carbohydrates", "")
        row["fats"] = macros.get("fats", "")
        row["proteins"] = macros.get("proteins", "")
        ingredients = agent_output.get("ingredients", [])
        row["ingredients_json"] = json.dumps(ingredients, ensure_ascii=False)

    # outputGuardrail / safetyChecks fields
    if job.agent == "outputGuardrail":
        for k in [
            "no_insuline_guidance",
            "no_carb_content",
            "no_emotional_or_judgmental_language",
            "no_risky_ingredient_substitutions",
            "no_treatment_recommendation",
            "no_medical_diagnosis",
        ]:
            if k in agent_output:
                row[k] = agent_output[k]

    return row


# CSV writing utilities: write header once, then append rows as we go
# Will ovewrrite existing output CSV file if it exists
def write_csv_header(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

def append_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        for r in rows:
            writer.writerow(r)

"""METHODS FOR CONNECTING TO OPENAI AND MAKING API CALLS"""
def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing environment variable {name}. "
            f"Set it, e.g. export {name}='...'"
        )
    return val

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_b64_cache(samples: List[Sample]) -> Dict[str, str]:
    unique_images: Dict[str, str] = {}
    for sample in samples:
        # Keep one encode per sample id.
        if sample.id not in unique_images:
            unique_images[sample.id] = sample.img_path

    cache: Dict[str, str] = {}
    total = len(unique_images)
    print(f"Pre-encoding {total} image(s) into base64 cache...", flush=True)
    for idx, (sample_id, img_path) in enumerate(unique_images.items(), start=1):
        cache[sample_id] = image_to_base64(img_path)
        if idx % 25 == 0 or idx == total:
            print(f"[cache] encoded {idx}/{total} image(s)", flush=True)
    return cache

def build_openai_client(root: str) -> OpenAI:
    # Load .env from project root (if present)
    load_dotenv(dotenv_path=os.path.join(root, ".env"), override=False)

    api_key = require_env("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID") or None

    return OpenAI(api_key=api_key, organization=org_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents-config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML agent config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Randomly select N samples from dataset before building jobs. "
        "Default: run all samples.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=12,
        help="Random seed used only when --num-samples is set (default: 12).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel OpenAI request workers (default: 10).",
    )
    args = parser.parse_args()

    # file lives in evals/eval_agents.py -> project root is one level up from evals/
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.agents_config
    if not os.path.isabs(config_path):
        config_path = os.path.join(root, config_path)

    samples = load_dataset(root)
    total_samples = len(samples)

    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise RuntimeError("--num-samples must be > 0")
        if args.num_samples > total_samples:
            raise RuntimeError(
                f"--num-samples={args.num_samples} exceeds available samples ({total_samples})"
            )
        random.seed(args.sample_seed)
        samples = random.sample(samples, k=args.num_samples)
        print(
            f"Selected {len(samples)} random sample(s) out of {total_samples} "
            f"(seed={args.sample_seed})."
        )
    else:
        print(f"Using all {total_samples} samples (no random subsampling).")

    print(f"Loaded {len(samples)} valid samples.")
    for s in samples[:5]:
        print(f"- {s.id}")
        print(f"  gt_keys: {list(s.gt.keys())}")
        print('________________________________________________')
    
    agents = load_agent_configs(config_path)
    print(f"Loaded {len(agents)} agent config(s) from {config_path}")
    for a in agents:
        print(f"- {a.name} ({a.io}): {a.models}")

    has_vision_work = any(a.io == "vision" and len(a.models) > 0 for a in agents)
    image_b64_cache: Dict[str, str] = {}
    if has_vision_work:
        image_b64_cache = build_image_b64_cache(samples)

    jobs = build_jobs(samples, agents)

    # Dry run: just print a few jobs so we know the harness is correct
    dry_run_print_jobs(jobs, n=10)

    # Create CSV file with headers (will overwrite if already exists)
    out_csv = os.path.join(root, "outputs", "results.csv")
    write_csv_header(out_csv)

    max_workers = max(1, int(args.max_workers))
    print(f"Using max_workers={max_workers}")
    print('============ Starting agent evals... may take a while ============')
    rows: List[Dict[str, Any]] = []
    total_jobs = len(jobs)

    if total_jobs == 0:
        print("No eval jobs to run (all configured model lists are empty).")
    elif max_workers == 1:
        # Single-thread mode remains useful for debugging.
        client = build_openai_client(root)
        for idx, job in enumerate(jobs, start=1):
            agent_output, latency_ms, in_tok, out_tok, parse_ok, err, raw = run_agent(
                job, client, image_b64_cache=image_b64_cache
            )
            row = make_csv_row(
                job, agent_output, root,
                latency_ms, in_tok, out_tok,
                parse_ok, err, raw
            )
            rows.append(row)

            print(
                f"[{idx}/{total_jobs}] Ran job: "
                f"agent={job.agent} model={job.model} id={job.sample.id} parse_ok={parse_ok}"
            )
    else:
        # One OpenAI client per worker thread avoids shared-state issues.
        thread_state = threading.local()
        ordered_rows: List[Optional[Dict[str, Any]]] = [None] * total_jobs

        def _run_one_job(job: EvalJob) -> Tuple[Dict[str, Any], bool]:
            client = getattr(thread_state, "client", None)
            if client is None:
                client = build_openai_client(root)
                thread_state.client = client

            agent_output, latency_ms, in_tok, out_tok, parse_ok, err, raw = run_agent(
                job, client, image_b64_cache=image_b64_cache
            )
            row = make_csv_row(
                job, agent_output, root,
                latency_ms, in_tok, out_tok,
                parse_ok, err, raw
            )
            return row, parse_ok

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_meta = {
                pool.submit(_run_one_job, job): (idx, job)
                for idx, job in enumerate(jobs, start=1)
            }

            for fut in as_completed(future_to_meta):
                idx, job = future_to_meta[fut]
                row, parse_ok = fut.result()
                ordered_rows[idx - 1] = row
                completed += 1

                print(
                    f"[{completed}/{total_jobs}] Ran job: "
                    f"agent={job.agent} model={job.model} id={job.sample.id} parse_ok={parse_ok}"
                )

        rows = [r for r in ordered_rows if r is not None]

    append_csv_rows(out_csv, rows)
    print(f"Wrote {len(rows)} REAL rows to {out_csv}")

if __name__ == "__main__":
    main()
