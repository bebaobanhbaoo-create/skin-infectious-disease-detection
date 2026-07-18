"""Resize and cache images for faster training."""
import os
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import hashlib

def resize_and_cache(input_dir, output_dir, target_size=256):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for img_file in tqdm(list(input_path.rglob('*'))):
        if img_file.suffix.lower() in image_extensions:
            relative_path = img_file.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                img = Image.open(img_file).convert('RGB')
                img = img.resize((target_size, target_size), Image.LANCZOS)
                img.save(output_file, quality=95)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()
    resize_and_cache(args.input_dir, args.output_dir, args.size)
