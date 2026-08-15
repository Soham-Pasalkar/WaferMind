import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import NAFNetSR
from inference import get_device, load_checkpoint, restore


def save_png(array, path):
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(
        (array * 255.0).round().astype(np.uint8),
        mode="L"
    )
    image.save(path)


def load_npy(path):
    array = np.load(path)
    array = np.asarray(array)

    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]

    if array.ndim != 2:
        raise ValueError(f"Unsupported shape: {array.shape}")

    return array.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="backend/checkpoints/best.pt")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(test_dir.glob("*.npy"))
    if not files:
        raise RuntimeError("No .npy files found.")

    device = get_device()
    model = NAFNetSR(width=32)
    load_checkpoint(model, args.checkpoint, device)

    total_time = 0.0

    print(f"Device: {device}")
    print(f"Images: {len(files)}")

    for index, path in enumerate(files, 1):
        array = load_npy(path)
        output, elapsed = restore(model, array, device)
        total_time += elapsed

        save_png(output, output_dir / f"{path.stem}.png")

        if index % 25 == 0 or index == len(files):
            print(f"{index}/{len(files)}")

    print()
    print(f"Average inference: {total_time / len(files):.3f} ms")
    print(f"Throughput: {1000.0 / (total_time / len(files)):.2f} images/s")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
