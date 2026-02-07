import os
import time
import argparse
from openai import OpenAI

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name to test")
    parser.add_argument("--image", default=None, help="Optional path to a local image (e.g. .jpeg)")
    args = parser.parse_args()

    # API key is required.
    require_env("OPENAI_API_KEY")

    # Optional IDs (only needed if your setup requires them).
    # If you don't use these, ignore.
    org_id = os.getenv("OPENAI_ORG_ID")
    project_id = os.getenv("OPENAI_PROJECT_ID")

    client = OpenAI(
        organization=org_id,
        project=project_id,
    )

    # 1) Text-only smoke test (connection + auth + basic model access)
    t0 = time.time()
    resp = client.responses.create(
        model=args.model,
        input="Say 'connection_ok' and nothing else.",
    )
    t1 = time.time()

    print("=== TEXT TEST ===")
    print("model:", args.model)
    print("latency_ms:", round((t1 - t0) * 1000, 2))
    print("output_text:", resp.output_text)

    # usage fields exist when available
    usage = getattr(resp, "usage", None)
    if usage:
        print("usage:", usage)

    # 2) Optional vision test (proves image input wiring works)
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")

        t0 = time.time()
        resp2 = client.responses.create(
            model=args.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is in this image? Reply in one short sentence."},
                        {"type": "input_image", "image_url": f"file://{os.path.abspath(args.image)}"},
                    ],
                }
            ],
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
