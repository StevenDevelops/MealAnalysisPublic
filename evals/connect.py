import os
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv
import base64

# Simple connection test for OpenAI API using the Responses API.
# - Reads OPENAI_API_KEY from env.
# - Optionally reads OPENAI_ORG_ID / OPENAI_PROJECT_ID if you use them.
# - Sends a text-only request (fast sanity check).
# - Optionally sends an image+text request if --image is provided.

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

def main():
    parser = argparse.ArgumentParser()
    # You can later pin this to a snapshot like: gpt-4.1-mini-2024-09-01
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name to test")
    parser.add_argument("--image", default=None, help="Optional path to a local image (e.g. .jpeg)")
    args = parser.parse_args()

    # File lives in evals/connect.py, so root is one level up.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(root, ".env"), override=False)

    # API key is required.
    api_key = require_env("OPENAI_API_KEY")

    # Optional organization header (only needed if you belong to multiple orgs).
    org_id = os.getenv("OPENAI_ORG_ID") or None

    # v1 SDK constructor signature for 1.12.0 – no `project` argument here
    client = OpenAI(
        api_key=api_key,
        organization=org_id,
    )

    # 1) Text-only smoke test
    t0 = time.time()
    resp = client.responses.create(
        model=args.model,
        input="Say 'connection_ok, haha' and nothing else.",
    )
    t1 = time.time()

    print("=== TEXT TEST ===")
    print("model:", args.model)
    print("latency_ms:", round((t1 - t0) * 1000, 2))
    # `output_text` is the recommended shortcut for the Responses API
    print("output_text:", resp.output_text)

    usage = getattr(resp, "usage", None)
    if usage:
        print("usage:", usage)

    # 2) Optional vision test
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")

        img_b64 = image_to_base64(args.image) # this is how we pass in local image

        t0 = time.time()
        resp2 = client.responses.create(
            model=args.model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image? Reply in one short sentence."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                ],
            }],
        )

        t1 = time.time()

        print("\n=== VISION TEST ===")
        print("image:", args.image)
        print("latency_ms:", round((t1 - t0) * 1000, 2))
        print("output_text:", resp2.output_text)
        usage2 = getattr(resp2, "usage", None)
        if usage2:
            print("usage:", usage2)


if __name__ == "__main__":
    main()