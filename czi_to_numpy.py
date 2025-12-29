#!/usr/bin/env python3
"""
czi_viewer_cv2.py

Usage:
    uv run python czi_viewer_cv2.py /path/to/some_file.czi

Controls:
    Up / Down arrows or k / j    : cycle channels
    Left / Right arrows or h / l : cycle CZI files in same directory
    y                            : copy current file path to clipboard
    q or Esc                     : quit
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
from aicsimageio import AICSImage

try:
    import pyperclip
except ImportError:
    pyperclip = None


class CziBrowser:
    def __init__(self, start_path: Path):
        self.files: List[Path] = self._find_czi_files(start_path)
        if not self.files:
            raise RuntimeError(f"No .czi files found in directory: {start_path.parent}")

        self.file_index: int = self.files.index(start_path.resolve())
        self.img: Optional[AICSImage] = None

        # Channel index is remembered across files
        self.channel_index: int = 0
        self.channel_names: Optional[List[str]] = None

        self.window_name = "CZI Channel Viewer (cv2)"

        self._load_current_file()

    # ---------- file handling ----------

    def _find_czi_files(self, path: Path) -> List[Path]:
        directory = path.parent
        return sorted(p.resolve() for p in directory.iterdir() if p.suffix.lower() == ".czi")

    def _load_current_file(self):
        """Load current CZI into AICSImage, keep channel_index if possible."""
        current_path = self.files[self.file_index]
        print(f"Loading file: {current_path}")
        self.img = AICSImage(str(current_path))

        try:
            self.channel_names = self.img.get_channel_names()
        except Exception:
            self.channel_names = None

        n_channels = self._get_num_channels()
        if n_channels == 0:
            self.channel_index = 0
        else:
            self.channel_index = max(0, min(self.channel_index, n_channels - 1))

    def _get_num_channels(self) -> int:
        data = self.img.get_image_data("TCZYX")
        return data.shape[1]

    # ---------- data extraction ----------

    def _get_channel_2d(self) -> np.ndarray:
        """
        Return a 2D float array (Y, X) for the current channel:
        - T=0
        - Z-max projection
        """
        data = self.img.get_image_data("TCZYX")  # (T, C, Z, Y, X)
        t_dim, c_dim, z_dim, y_dim, x_dim = data.shape

        t = 0
        c = max(0, min(self.channel_index, c_dim - 1))

        channel_3d = data[t, c]  # (Z, Y, X)
        if z_dim > 1:
            channel_2d = channel_3d.max(axis=0)  # (Y, X)
        else:
            channel_2d = channel_3d[0]

        return channel_2d.astype(np.float64, copy=False)

    def _to_display_uint8(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0–255 uint8 for display with OpenCV."""
        if img.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        min_val = np.nanmin(img)
        max_val = np.nanmax(img)

        if max_val <= min_val:
            return np.zeros_like(img, dtype=np.uint8)

        norm = (img - min_val) / (max_val - min_val)
        return (norm * 255.0).clip(0, 255).astype(np.uint8)

    # ---------- overlay ----------

    def _overlay_info(self, disp: np.ndarray) -> np.ndarray:
        """
        Draw filename and channel info onto the display image.
        """
        # Convert grayscale to BGR for colored text
        if len(disp.shape) == 2:
            img_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = disp.copy()

        current_path = self.files[self.file_index]
        n_channels = self._get_num_channels()

        if self.channel_names and self.channel_index < len(self.channel_names):
            ch_label = f"{self.channel_index} ({self.channel_names[self.channel_index]})"
        else:
            ch_label = f"{self.channel_index}"

        text1 = current_path.name
        text2 = f"Channel {ch_label} / {n_channels - 1}"

        # Drawing parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        color_fg = (255, 255, 255)  # white
        color_bg = (0, 0, 0)        # black

        # Positions
        y0 = 20
        y1 = 40
        x = 10

        # Helper to draw text with a black outline for readability
        def draw_outlined(text, y):
            cv2.putText(img_bgr, text, (x, y), font, scale, color_bg, thickness + 2, cv2.LINE_AA)
            cv2.putText(img_bgr, text, (x, y), font, scale, color_fg, thickness, cv2.LINE_AA)

        draw_outlined(text1, y0)
        draw_outlined(text2, y1)

        return img_bgr

    # ---------- clipboard ----------

    def _copy_current_path_to_clipboard(self):
        current_path = str(self.files[self.file_index])
        if pyperclip is None:
            print(f"[WARN] pyperclip not installed; cannot copy to clipboard. Path: {current_path}")
            return
        try:
            pyperclip.copy(current_path)
            print(f"[INFO] Copied to clipboard: {current_path}")
        except Exception as e:
            print(f"[WARN] Failed to copy to clipboard: {e}. Path: {current_path}")

    # ---------- UI loop ----------

    def show(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while True:
            channel_2d = self._get_channel_2d()
            disp = self._to_display_uint8(channel_2d)
            disp = self._overlay_info(disp)

            current_path = self.files[self.file_index]
            cv2.setWindowTitle(self.window_name, current_path.name)

            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(0) & 0xFFFF

            # q or Esc -> quit
            if key in (ord('q'), 27):
                break

            # 'y' -> copy filename to clipboard
            elif key == ord('y'):
                self._copy_current_path_to_clipboard()

            # File navigation: left/right arrows or h/l
            elif key in (81, ord('h')):  # Left
                self._prev_file()
            elif key in (83, ord('l')):  # Right
                self._next_file()

            # Channel navigation: up/down arrows or k/j
            elif key in (82, ord('k')):  # Up
                self._next_channel()
            elif key in (84, ord('j')):  # Down
                self._prev_channel()

        cv2.destroyAllWindows()

    def _next_channel(self):
        n_channels = self._get_num_channels()
        if n_channels == 0:
            return
        self.channel_index = (self.channel_index + 1) % n_channels
        print(f"Channel -> {self.channel_index}")

    def _prev_channel(self):
        n_channels = self._get_num_channels()
        if n_channels == 0:
            return
        self.channel_index = (self.channel_index - 1) % n_channels
        print(f"Channel -> {self.channel_index}")

    def _next_file(self):
        self.file_index = (self.file_index + 1) % len(self.files)
        self._load_current_file()

    def _prev_file(self):
        self.file_index = (self.file_index - 1) % len(self.files)
        self._load_current_file()


def main():
    if len(sys.argv) != 2:
        print("Usage: python czi_viewer_cv2.py /path/to/some_file.czi")
        sys.exit(1)

    start_path = Path(sys.argv[1]).resolve()
    if not start_path.is_file():
        print(f"File not found: {start_path}")
        sys.exit(1)

    browser = CziBrowser(start_path)
    browser.show()


if __name__ == "__main__":
    main()
