#!/usr/bin/env python3

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
import nd2

def save_image(array, outname, fmt):
    # Normalize to 8-bit for PNG/JPG (in general nd2 files are uint16)
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        array = 255 * ((array - array.min()) / (np.ptp(array) or 1))
        array = array.astype(np.uint8)

    img = Image.fromarray(array)
    img.save(outname, format=fmt.upper())

def main():
    parser = argparse.ArgumentParser(
        description="Convert ND2 files to PNG or JPG images.")
    parser.add_argument("files", nargs="+",
        help="Input ND2 files (filenames or glob patterns, e.g. '*.nd2')")
    parser.add_argument("-f", "--format", choices=["png", "jpg", "jpeg"], default="png",
        help="Output format (default: png)")
    parser.add_argument("-o", "--output-dir", default=None,
        help="Optional: output directory (default: alongside input files)")

    args = parser.parse_args()

    # Resolve all files from globs
    resolved_files = set()
    for pat in args.files:
        resolved_files.update(glob.glob(pat))
    if not resolved_files:
        print("No files found matching your patterns.", file=sys.stderr)
        sys.exit(1)

    fmt = "jpeg" if args.format == "jpg" else args.format.lower()

    for path in sorted(resolved_files):
        print(f"Processing {path}...", file=sys.stderr)
        try:
            with nd2.ND2File(path) as f:
                sizes = f.sizes
                is_rgb = f.is_rgb
                arr = f.asarray()  # Load to numpy array
                base = os.path.splitext(os.path.basename(path))[0]
                outdir = args.output_dir or os.path.dirname(os.path.abspath(path)) or "."

                # Axis order checking
                axes = list(sizes.keys())
                n_axes = {ax: sizes[ax] for ax in axes}
                # Priority to split by: T, C, Z, (Y, X) are 2D dims, S is color (in RGB)
                axis_indices = dict(T=None, C=None, Z=None, S=None)
                for ax in axis_indices:
                    axis_indices[ax] = axes.index(ax) if ax in axes else None

                # Collapse all except last 2 (Y,X) and S (for RGB)
                n_frames = n_axes.get('T', 1)
                n_channels = n_axes.get('C', 1)
                n_slices = n_axes.get('Z', 1)
                n_samples = n_axes.get('S', 1)  # RGB

                # If there are >1 frames (T), channels (C), or slices (Z), output separate files
                out_imgs = 0
                for t in range(n_frames):
                    for c in range(n_channels):
                        for z in range(n_slices):

                            # index array:
                            idx = []
                            for ax in axes:
                                if ax == 'T':
                                    idx.append(t)
                                elif ax == 'C':
                                    idx.append(c)
                                elif ax == 'Z':
                                    idx.append(z)
                                else:
                                    idx.append(slice(None))

                            img = arr[tuple(idx)]

                            # If there is an S (samples) axis, move that to last
                            if axis_indices['S'] is not None:
                                # Assumption: S is appended as last axis; no change required for Pillow
                                # But need S==3 for RGB, handle otherwise (skip/gray)
                                if n_samples != 3:
                                    print(f"Skipping frame (T={t},C={c},Z={z}): samples axis not length 3 (S={n_samples})", file=sys.stderr)
                                    continue
                                pass

                            # Save
                            frame_suffix = ""
                            if n_frames > 1:
                                frame_suffix += f"_t{t}"
                            if n_channels > 1:
                                frame_suffix += f"_c{c}"
                            if n_slices > 1:
                                frame_suffix += f"_z{z}"

                            outname = os.path.join(
                                outdir,
                                f"{base}{frame_suffix}.{fmt if fmt != 'jpeg' else 'jpg'}"
                            )
                            save_image(img, outname, fmt)
                            out_imgs += 1

                if out_imgs == 0:
                    print(f"WARNING: No usable 2D images extracted from {path}!", file=sys.stderr)
                else:
                    print(f"Saved {out_imgs} images for {base}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR processing {path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
