import nd2
import numpy as np
import cv2
import argparse
import os

def rescale_to_uint8(image):
    """Linearly scale any dtype to uint8."""
    imin, imax = np.min(image), np.max(image)
    if imin == imax:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - imin) * (255.0 / (imax - imin))
    return scaled.astype(np.uint8)

def default_avi_filename(nd2_filename):
    root, ext = os.path.splitext(nd2_filename)
    return root + ".avi"

def default_mp4_filename(nd2_filename):
    root, ext = os.path.splitext(nd2_filename)
    return root + ".mp4"

def main():
    parser = argparse.ArgumentParser(description="Convert an ND2 time sequence to a video (AVI or MP4) with optional crop.")
    parser.add_argument("nd2_file", type=str, help="Input ND2 file")
    parser.add_argument("--output", type=str, default=None, help="Output video file (default: same as ND2 but with extension based on codec)")
    parser.add_argument("--framerate", type=int, default=10, help="Video framerate (default: 10)")
    parser.add_argument("--channel", type=int, default=0, help="Channel to use if multi-channel (default: 0)")
    parser.add_argument("--colormap", type=str, default=None, help="Apply OpenCV colormap (e.g. 'JET', optional)")
    parser.add_argument("--codec", type=str, choices=["avi", "mp4"], default="avi", help="Video format/codec: 'avi' (MJPG, default) or 'mp4' (mp4v)")

    # Crop: x, y, width, height
    parser.add_argument("--crop", type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                        help="Rectangular crop region as X Y WIDTH HEIGHT (pixels)")

    args = parser.parse_args()

    # Choose output file and codec based on selection
    if args.codec == "avi":
        output_file = args.output if args.output else default_avi_filename(args.nd2_file)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    elif args.codec == "mp4":
        output_file = args.output if args.output else default_mp4_filename(args.nd2_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError("Unsupported codec: {}".format(args.codec))

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
        arr = np.moveaxis(arr, [t_axis, c_axis, y_axis, x_axis], [0, 1, 2, 3])
        arr = arr[:, args.channel]  # pick specified channel
    else:
        arr = np.moveaxis(arr, [t_axis, y_axis, x_axis], [0, 1, 2])

    nframes, height, width = arr.shape

    # Crop parameters
    if args.crop is not None:
        cx, cy, cw, ch = args.crop
        # Validate crop rectangle
        if not (0 <= cx < width and 0 <= cy < height and 0 < cw <= width - cx and 0 < ch <= height - cy):
            raise ValueError("Crop rectangle out of bounds (check x, y, width, height versus frame size).")
        out_width, out_height = cw, ch
    else:
        cx, cy = 0, 0
        out_width, out_height = width, height

    # Set up OpenCV video writer
    is_color = args.colormap is not None
    out = cv2.VideoWriter(output_file, fourcc, args.framerate, (out_width, out_height), isColor=is_color)

    # Colormap option
    colormap_idx = None
    if args.colormap is not None:
        if hasattr(cv2, f'COLORMAP_{args.colormap.upper()}'):
            colormap_idx = getattr(cv2, f'COLORMAP_{args.colormap.upper()}')
        else:
            raise ValueError(f"Unknown colormap '{args.colormap}' for OpenCV.")

    print(f"Writing {nframes} frames to '{output_file}' at {args.framerate} fps")
    for i in range(nframes):
        frame = rescale_to_uint8(arr[i])
        # Apply crop BEFORE colormap
        frame = frame[cy:cy+out_height, cx:cx+out_width]
        if colormap_idx is not None:
            frame = cv2.applyColorMap(frame, colormap_idx)
        out.write(frame)
    out.release()
    print("Done!")

if __name__ == "__main__":
    main()
