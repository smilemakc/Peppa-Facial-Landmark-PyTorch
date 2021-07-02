import sys
from pathlib import Path, PosixPath
from typing import Union

import cv2
import numpy as np


def read_pts(filename: Union[Path, PosixPath, str]) -> np.ndarray:
    return np.loadtxt(
        str(filename), comments=("version:", "n_points:", "{", "}")
    )


def draw_pts(pts: np.array, frame: np.array) -> None:
    for point in pts.astype(int):
        frame = cv2.circle(frame, tuple(point), 1, (255, 255, 0), 4)


def process_video(path: Union[Path, PosixPath, str]):
    cap = cv2.VideoCapture(str(path / "vid.avi"))
    if path.is_file():
        return 0
    frames_path = path / "annot"
    if not (frames_path).exists():
        frames_path.mkdir()
    frame_number = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fname = f"{frame_number}".rjust(6, "0")
        frame_path = frames_path / f"{fname}.jpg"
        frame_number += 1
        if frame_path.exists():
            continue
        cv2.imwrite(frame_path.as_posix(), frame)
        print(f"frame: {frame_number}", end="\r")
    print("")
    return frame_number


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("run script with path to the parent folder of the 300VW_Dataset_2015_12_14")
        sys.exit(1)
    base_path = Path(sys.argv[1])
    total_frames = 0
    for path in base_path.iterdir():
        print(f"starts process path {path}")
        total_frames += process_video(path)
    print(f"splitting videos on the frames was complete with total frames {total_frames}")
