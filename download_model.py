#!/usr/bin/env python3
"""
Download a YOLO model from URL, convert to ONNX, extract classes, and register it.

Usage:
    python download_model.py <url> <model_name>

Example:
    python download_model.py https://example.com/best.pt my-model
"""

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


MODELS_DIR = Path("models")
REGISTRY_PATH = MODELS_DIR / "registry.json"
INPUT_SIZE = 640


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 8192
            while True:
                data = response.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()
    except Exception as e:
        print(f"\nError downloading file: {e}", file=sys.stderr)
        sys.exit(1)


def convert_to_onnx(pt_path: Path, output_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not found. Install it with: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {pt_path} ...")
    model = YOLO(str(pt_path))

    print(f"Exporting to ONNX (imgsz={INPUT_SIZE}) ...")
    exported = model.export(format="onnx", imgsz=INPUT_SIZE, simplify=True)

    exported_path = Path(exported)
    shutil.move(str(exported_path), str(output_path))
    print(f"ONNX model saved to {output_path}")

    return model


def extract_classes(model, classes_path: Path) -> list[str]:
    names: dict = model.names  # {0: 'cat', 1: 'dog', ...}
    classes = [names[i] for i in sorted(names.keys())]
    classes_path.write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(f"classes.txt saved ({len(classes)} classes): {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
    return classes


def update_registry(model_id: str, model_name: str) -> None:
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = {"models": []}

    # Remove existing entry with same id
    registry["models"] = [m for m in registry["models"] if m["id"] != model_id]

    registry["models"].append({
        "id": model_id,
        "name": model_name,
        "modelPath": f"models/{model_id}/model.onnx",
        "classesPath": f"models/{model_id}/classes.txt",
    })

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"registry.json updated — model id: '{model_id}'")


def main():
    parser = argparse.ArgumentParser(description="Download YOLO model, convert to ONNX, register it.")
    parser.add_argument("url", help="URL to download the YOLO .pt model file")
    parser.add_argument("model_name", help="Display name for the model (e.g. 'My Custom Model')")
    args = parser.parse_args()

    # Derive a filesystem-safe id from the display name
    model_id = args.model_name.lower().replace(" ", "-")
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = model_dir / "model.onnx"
    classes_path = model_dir / "classes.txt"

    with tempfile.TemporaryDirectory() as tmp:
        pt_path = Path(tmp) / "model.pt"
        download_file(args.url, pt_path)
        model = convert_to_onnx(pt_path, onnx_path)

    extract_classes(model, classes_path)
    update_registry(model_id, args.model_name)

    print(f"\nDone. Model '{args.model_name}' is ready at models/{model_id}/")


if __name__ == "__main__":
    main()
