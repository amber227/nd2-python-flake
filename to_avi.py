import nd2
import numpy as np
import cv2
import argparse
import os

def rescale_to_uint8(image):
    """Linearly scale any dtype to uint8."""
    imin, imax = np.min(image), np.max(image)
    if imin == imax:
        # Flat image
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - imin) * (255.0 / (imax - imin))
    return scaled.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Convert an ND2 time sequence to an AVI video.")
    parser.add_argument("nd2_file", type=str, help="Input ND2 file")
    parser.add_argument("output_avi", type=str, help="Output AVI file")
    parser.add_argument("--framerate", type=int, default=10, help="AVI framerate (default: 10)")
    parser.add_argument("--channel", type=int, default=0, help="Channel to use if multi-channel (default: 0)")
    parser.add_argument("--colormap", type=str, default=None, help="Apply OpenCV colormap (e.g. 'JET', optional)")
    args = parser.parse_args()

    # Read ND2 file
    with nd2.ND2File(args.nd2_file) as f:
        shape = f.shape
        sizes = f.sizes
        axes = ''.join(sizes.keys())  # e.g., 'TCYX'
        arr = f.asarray()

    # Handle data axes (rearrange to T, Y, X)
    if 'T' not in axes:
        raise ValueError("ND2 file does not contain a time sequence ('T' axis).")
    t_axis = axes.index('T')
    y_axis = axes.index('Y')
    x_axis = axes.index('X')

    # Channel handling
    if 'C' in axes:
        c_axis = axes.index('C')
        # Move axes so we can slice channel
        arr = np.moveaxis(arr, [t_axis, c_axis, y_axis, x_axis], [0, 1, 2, 3])
        arr = arr[:, args.channel]  # pick specified channel
    else:
        arr = np.moveaxis(arr, [t_axis, y_axis, x_axis], [0, 1, 2])

    nframes, height, width = arr.shape

    # Set up OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output_avi, fourcc, args.framerate, (width, height), isColor=False if args.colormap is None else True)

    # Colormap option
    colormap_idx = None
    if args.colormap is not None:
        if hasattr(cv2, f'COLORMAP_{args.colormap.upper()}'):
            colormap_idx = getattr(cv2, f'COLORMAP_{args.colormap.upper()}')
        else:
            raise ValueError(f"Unknown colormap '{args.colormap}' for OpenCV.")

    print(f"Writing {nframes} frames to '{args.output_avi}' at {args.framerate} fps")
    for i in range(nframes):
        frame = rescale_to_uint8(arr[i])
        if colormap_idx is not None:
            frame = cv2.applyColorMap(frame, colormap_idx)
            out.write(frame)
        else:
            # OpenCV expects 3D array for color. For grayscale, provide single channel 2D.
            out.write(frame)
    out.release()
    print("Done!")

if __name__ == "__main__":
    main()
