#!/usr/bin/env python3
"""
Convert MKV or MP4 videos to looped GIFs with cropping options.
"""

import argparse
import sys
from pathlib import Path
import imageio
import numpy as np
from PIL import Image
import cv2


def load_video_frames(video_path, start_time=None, end_time=None):
    """Load video frames with optional time cropping."""
    # Use OpenCV for better MKV support
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices for time cropping
    start_frame = int(start_time * fps) if start_time is not None else 0
    end_frame = int(end_time * fps) if end_time is not None else total_frames
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= start_frame and frame_idx < end_frame:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
        if frame_idx >= end_frame:
            break
    
    cap.release()
    return frames, fps


def crop_frame(frame, x, y, width, height):
    """Crop a frame to specified dimensions."""
    return frame[y:y+height, x:x+width]


def resize_frame(frame, scale_factor):
    """Resize frame by scale factor."""
    if scale_factor == 1.0:
        return frame
    
    pil_image = Image.fromarray(frame)
    new_size = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))
    resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    return np.array(resized)


def video_to_gif(input_path, output_path, start_time=None, end_time=None, 
                 crop=None, scale=1.0, fps=10, loop=0):
    """
    Convert video to GIF with optional cropping and scaling.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output GIF file
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        crop: Tuple of (x, y, width, height) for cropping (optional)
        scale: Scale factor for resizing (default: 1.0)
        fps: Output GIF frame rate (default: 10)
        loop: Number of loops (0 = infinite, default: 0)
    """
    print(f"Loading video: {input_path}")
    frames, original_fps = load_video_frames(input_path, start_time, end_time)
    
    if not frames:
        raise ValueError("No frames loaded from video")
    
    print(f"Loaded {len(frames)} frames at {original_fps} fps")
    
    # Process frames
    processed_frames = []
    for frame in frames:
        # Apply spatial cropping if specified
        if crop is not None:
            crop_x, crop_y, crop_width, crop_height = crop
            frame = crop_frame(frame, crop_x, crop_y, crop_width, crop_height)
        
        # Apply scaling if specified
        if scale != 1.0:
            frame = resize_frame(frame, scale)
        
        processed_frames.append(frame)
    
    # Calculate duration per frame for target fps
    duration = 1.0 / fps
    
    print(f"Saving GIF: {output_path}")
    print(f"Output: {len(processed_frames)} frames at {fps} fps")
    if processed_frames:
        print(f"Frame size: {processed_frames[0].shape[1]}x{processed_frames[0].shape[0]}")
    
    # Save as GIF
    imageio.mimsave(
        output_path,
        processed_frames,
        duration=duration,
        loop=loop
    )
    
    print(f"GIF saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Convert MKV/MP4 videos to looped GIFs")
    parser.add_argument("input", help="Input video file (MKV or MP4)")
    parser.add_argument("output", nargs="?", help="Output GIF file (optional)")
    
    # Time cropping options
    parser.add_argument("--start", "-s", type=float, help="Start time in seconds")
    parser.add_argument("--end", "-e", type=float, help="End time in seconds")
    
    # Spatial cropping options
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                       help="Crop region as: x y width height")
    
    # Scaling and output options
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (default: 1.0)")
    parser.add_argument("--fps", type=int, default=10, help="Output GIF frame rate (default: 10)")
    parser.add_argument("--loops", type=int, default=0, help="Number of loops (0 = infinite, default: 0)")
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)
    
    if input_path.suffix.lower() not in ['.mkv', '.mp4', '.avi', '.mov', '.wmv']:
        print(f"Error: Input file must be a video file (MKV, MP4, AVI, MOV, WMV), got '{input_path.suffix}'")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.gif')
    
    # Validate cropping parameters
    if args.crop is not None:
        if any(param < 0 for param in args.crop):
            print("Error: Cropping parameters must be non-negative")
            sys.exit(1)
    
    # Validate time parameters
    if args.start is not None and args.start < 0:
        print("Error: Start time must be non-negative")
        sys.exit(1)
    
    if args.end is not None and args.end < 0:
        print("Error: End time must be non-negative")
        sys.exit(1)
    
    if args.start is not None and args.end is not None and args.start >= args.end:
        print("Error: Start time must be less than end time")
        sys.exit(1)
    
    if args.scale <= 0:
        print("Error: Scale factor must be positive")
        sys.exit(1)
    
    if args.fps <= 0:
        print("Error: FPS must be positive")
        sys.exit(1)
    
    try:
        video_to_gif(
            input_path=input_path,
            output_path=output_path,
            start_time=args.start,
            end_time=args.end,
            crop=args.crop,
            scale=args.scale,
            fps=args.fps,
            loop=args.loops
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
