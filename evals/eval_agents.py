import csv
import time
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Literal, Any

ALLOWED_IMAGE_EXT = ".jpeg"

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
    for j in jobs[:n]:
        print(f"- agent={j.agent} io={j.io} model={j.model} sample_id={j.sample.id}")

# ---- CSV schema ----
CSV_COLUMNS = [
    # identity
    "id", "agent", "io", "model",
    # paths (useful for debugging)
    "img_path", "json_path",
    # timing / usage (filled later when we call OpenAI)
    "latency_ms", "input_tokens", "output_tokens",
    # common eval fields (filled later by scoring)
    "parse_ok", "error",
    # guardrail/safety booleans (optional per agent)
    "is_food", "no_pii", "no_humans", "no_captcha",
    # mealAnalysis fields (optional per agent)
    "recommendation", "meal_title", "meal_description", "guidance_message",
    "calories", "carbohydrates", "fats", "proteins",
    # scores (filled later)
    "guardrail_pass", "safety_pass",
    "macros_score_0_100", "ingredients_accuracy_0_100", "text_quality_0_100",
    "meal_composite_0_100",
]

"""METHODS FOR RUNNING AGENTS AND OUTPUT RESULTS TO CSV"""
def run_agent(job: EvalJob) -> Dict[str, Any]:
    """
    Stub for now: no OpenAI calls.
    Returns a dict shaped like an "agent output" so we can test plumbing.

    Later, replace this body with:
    - prompt+schema building
    - OpenAI Responses API call
    - JSON parsing
    """
    # Minimal dummy outputs by agent type
    if job.agent == "guardrailCheck":
        return {
            "is_food": True,
            "no_pii": True,
            "no_humans": True,
            "no_captcha": True,
        }
    if job.agent == "safetyChecks":
        return {
            "no_judgmental_language": True,
            "no_medical_advice": True,
            "no_diagnosis": True,
            "no_treatment_recommendations": True,
        }
    # mealAnalysis dummy
    return {
        "is_food": True,
        "recommendation": "green",
        "meal_title": "Dummy Meal Title",
        "meal_description": "Dummy meal description.",
        "guidance_message": "Dummy guidance message.",
        "macros": {
            "calories": 500,
            "carbohydrates": 50,
            "fats": 20,
            "proteins": 25,
        },
        "ingredients": [
            {"name": "dummy ingredient", "impact": "green"}
        ],
    }

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


def make_csv_row(job: EvalJob, agent_output: Dict[str, Any], root: str) -> Dict[str, Any]:
    """
    Convert (job + agent_output) into a single flat CSV row.
    For now: no scoring, no tokens, no real latency. Just plumbing.
    """
    row: Dict[str, Any] = {k: "" for k in CSV_COLUMNS}

    row["id"] = job.sample.id
    row["agent"] = job.agent
    row["io"] = job.io
    row["model"] = job.model
    row["img_path"] = _project_relpath(job.sample.img_path, root)
    row["json_path"] = _project_relpath(job.sample.json_path, root)

    # pretend we did work
    row["parse_ok"] = 1
    row["error"] = ""

    # Fill guardrail outputs if present
    for k in ["is_food", "no_pii", "no_humans", "no_captcha"]:
        if k in agent_output:
            row[k] = agent_output[k]

    # Fill mealAnalysis outputs if present
    if "recommendation" in agent_output:
        row["recommendation"] = agent_output.get("recommendation", "")
        row["meal_title"] = agent_output.get("meal_title", "")
        row["meal_description"] = agent_output.get("meal_description", "")
        row["guidance_message"] = agent_output.get("guidance_message", "")

        macros = agent_output.get("macros", {}) or {}
        row["calories"] = macros.get("calories", "")
        row["carbohydrates"] = macros.get("carbohydrates", "")
        row["fats"] = macros.get("fats", "")
        row["proteins"] = macros.get("proteins", "")

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

def main():
    # file lives in evals/eval_agents.py -> project root is one level up from evals/
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    samples = load_dataset(root)

    print(f"Loaded {len(samples)} valid samples.")
    for s in samples[:5]:
        print(f"- {s.id}")
        print(f"  gt_keys: {list(s.gt.keys())}")
        print('________________________________________________')
    
    # Set up agent configurations
    agents: List[AgentConfig] = [
        AgentConfig(name="inputGuardrail", io="vision", models=["gpt-4.1-mini", "gpt-4.1"]),
        AgentConfig(name="mealAnalysis",   io="vision", models=["gpt-4.1-mini", "gpt-4.1"]),
        AgentConfig(name="outputGuardrail",   io="text",   models=["gpt-4.1-nano", "gpt-4.1-mini"]),
    ]

    jobs = build_jobs(samples, agents)

    # Dry run: just print a few jobs so we know the harness is correct
    dry_run_print_jobs(jobs, n=10)

    # Create CSV file with headers (will overwrite if already exists)
    out_csv = os.path.join(root, "outputs", "results.csv")
    write_csv_header(out_csv)

    # next safe step: run just a handful of jobs with dummy outputs
    rows: List[Dict[str, Any]] = []
    for job in jobs[:10]:
        agent_output = run_agent(job)         # stubbed (no OpenAI)
        row = make_csv_row(job, agent_output, root) # flatten for CSV
        rows.append(row)

    append_csv_rows(out_csv, rows)
    print(f"Wrote {len(rows)} dummy rows to {out_csv}")
        

if __name__ == "__main__":
    main()
