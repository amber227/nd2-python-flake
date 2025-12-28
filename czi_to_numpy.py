"""
czi_to_numpy.py

Example usage:
    python czi_to_numpy.py /path/to/image.czi
"""

import sys
from pathlib import Path

import numpy as np
from aicsimageio import AICSImage


def load_czi(path: str) -> AICSImage:
    """Load a CZI file with AICSImage."""
    img = AICSImage(path)
    return img


def inspect_image(img: AICSImage):
    """
    Print dimension metadata and channel info.

    AICSImage standard order is usually "TCZYX":
      T: time
      C: channel
      Z: z-slice
      Y: height
      X: width
    """
    print("=== Basic Info ===")
    print(f"Data type      : {img.dtype}")
    print(f"Dimensions     : {img.dims}")       # e.g. Dimensions(T=1, C=3, Z=5, Y=1024, X=1024)
    print(f"Dimension order: {img.dims.order}") # e.g. 'TCZYX'
    print(f"Shape (TCZYX)  : {img.shape}")     # e.g. (1, 3, 5, 1024, 1024) in the given order

    # Channel names (if present in metadata)
    try:
        channel_names = img.get_channel_names()
    except Exception:
        channel_names = None

    print("\n=== Channels ===")
    if channel_names:
        for i, name in enumerate(channel_names):
            print(f"  C={i}: {name}")
    else:
        print("  No channel names found; refer to channels by index (0..C-1).")


def get_all_channels_numpy(img: AICSImage) -> np.ndarray:
    """
    Return the full image as a NumPy array in TCZYX order.

    You can reorder or squeeze dimensions as needed after this.
    """
    data = img.get_image_data("TCZYX")  # ensures a known dimension order
    return data  # shape: (T, C, Z, Y, X)


def get_single_channel_numpy(
    img: AICSImage,
    channel_index: int,
    time_index: int = 0,
    z_index: int | None = None,
) -> np.ndarray:
    """
    Extract a particular channel (and optionally a single time and/or z) as a NumPy array.

    Parameters
    ----------
    channel_index : int
        Channel index (0-based).
    time_index : int
        Timepoint index (0-based). Default: 0.
    z_index : int | None
        If None, return all z-slices for that channel.
        If int, return a single z-slice.

    Returns
    -------
    np.ndarray
        If z_index is None: shape (Z, Y, X)
        If z_index is int:  shape (Y, X)
    """
    # Get full data as TCZYX
    data = img.get_image_data("TCZYX")  # (T, C, Z, Y, X)
    t_dim, c_dim, z_dim, y_dim, x_dim = data.shape

    if not (0 <= channel_index < c_dim):
        raise ValueError(f"channel_index {channel_index} out of range [0, {c_dim - 1}]")
    if not (0 <= time_index < t_dim):
        raise ValueError(f"time_index {time_index} out of range [0, {t_dim - 1}]")

    if z_index is None:
        # All Z for one T, one C  -> (Z, Y, X)
        channel_data = data[time_index, channel_index, :, :, :]
    else:
        if not (0 <= z_index < z_dim):
            raise ValueError(f"z_index {z_index} out of range [0, {z_dim - 1}]")
        # Single Z -> (Y, X)
        channel_data = data[time_index, channel_index, z_index, :, :]

    return channel_data


def main(path_str: str):
    path = Path(path_str)
    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading CZI file: {path}")
    img = load_czi(str(path))

    # Inspect
    inspect_image(img)

    # Example: get all data as NumPy
    all_data = get_all_channels_numpy(img)
    print("\n=== All data ===")
    print(f"all_data.shape (TCZYX): {all_data.shape}")
    print(f"all_data.dtype        : {all_data.dtype}")

    # Example: select a specific channel
    # Here we just pick channel 0; in practice you would choose based on inspect_image()
    channel_idx = 0
    ch0 = get_single_channel_numpy(img, channel_index=channel_idx, time_index=0, z_index=None)
    print("\n=== Single channel example ===")
    print(f"Selected channel index: {channel_idx}")
    print(f"ch0.shape             : {ch0.shape}")  # (Z, Y, X)
    print(f"ch0.dtype             : {ch0.dtype}")

    # If you want to save these arrays (e.g., as .npy), you can do:
    # np.save(path.with_suffix(f'.channel{channel_idx}.npy'), ch0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python czi_to_numpy.py /path/to/image.czi")
        sys.exit(1)
    main(sys.argv[1])
