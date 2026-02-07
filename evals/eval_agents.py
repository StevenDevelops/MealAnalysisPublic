import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

ALLOWED_IMAGE_EXT = ".jpeg"

@dataclass(frozen=True) # @dataclass for boilerplate, frozen=True for immutability
class Sample:
    id: str
    img_path: str
    json_path: str
    gt: Dict  # parsed ground truth JSON


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


def main():
    # file lives in evals/eval_agents.py -> project root is one level up from evals/
    root = os.path.dirname(os.path.dirname(__file__))

    samples = load_dataset(root)

    print(f"Loaded {len(samples)} valid samples.")
    for s in samples[:5]:
        print(f"- {s.id}")
        print(f"  gt_keys: {list(s.gt.keys())}")
        print('________________________________________________')
        


if __name__ == "__main__":
    main()
