import nd2
import sys
import os

def summarize_nd2(file_path):
    with nd2.ND2File(file_path) as f:
        print(f"File: {os.path.basename(file_path)}")
        print("="*60)
        print(f"Path:     {f.path}")
        print(f"Shape:    {f.shape}")
        print(f"ndim:     {f.ndim}")
        print(f"dtype:    {f.dtype}")
        print(f"Sizes:    {f.sizes}")  # OrderedDict of axes
        print(f"Axes:     {''.join(f.sizes.keys())}")
        print(f"Is RGB:   {f.is_rgb}")
        print(f"Size:     {f.size:,} voxels")

        axes = f.sizes
        nframes = axes.get('T', 1)
        nchannels = axes.get('C', 1)
        height = axes.get('Y', None)
        width  = axes.get('X', None)
        zstack = axes.get('Z', None)
        print()
        print(f"Frames (T):   {nframes}")
        if 'Z' in axes:
            print(f"Z planes:     {zstack}")
        print(f"Channels (C): {nchannels}")
        print(f"Height (Y):   {height}")
        print(f"Width  (X):   {width}")

        # voxel size
        try:
            vox = f.voxel_size()
            if vox is not None:
                print(f"Voxel size:   {vox}")
        except Exception:
            pass

        # experiment timeline
        try:
            if hasattr(f, "experiment"):
                print()
                print("Experiment structure: ")
                for loop in f.experiment:
                    print(f"  {loop}")
        except Exception:
            pass

        # text info (capture time, comments, optics)
        if hasattr(f, "text_info") and f.text_info:
            print()
            if "date" in f.text_info:
                print(f"File date:    {f.text_info['date']}")
            if "description" in f.text_info:
                desc = f.text_info["description"]
                print("Description:  " + desc.splitlines()[0] if '\n' in desc else desc)
            if "optics" in f.text_info:
                print(f"Optics:       {f.text_info['optics']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python nd2_summary.py <file.nd2>")
        sys.exit(1)
    summarize_nd2(sys.argv[1])
