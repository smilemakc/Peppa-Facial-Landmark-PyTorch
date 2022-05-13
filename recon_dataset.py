import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

new_ds_path = Path("recon_ds")


def rewrite_path(path):
    path = Path(path)
    parts = path.parent.parts[1:]
    new_parent = new_ds_path / f"{parts[0]}_Out"
    new_path = new_parent / "/".join(parts[1:]) / path.name
    if not new_path.parent.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
    return new_path


def do(filename):
    gogo = json.load(open(filename))
    result = []
    for data in tqdm(gogo):
        image_path = Path(data["image_path"])
        landmarks = np.array(data["keypoints"])
        bbox = np.array(data["bbox"])
        img = cv2.imread(image_path.as_posix())
        img_h, img_w = img.shape[0], img.shape[1]
        center = np.array([bbox[2] + bbox[0], bbox[1] + bbox[3]]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ex_w, ex_h = np.array([w, h]) * [0.4, 0.6] / 2
        x1 = center[0] - w / 2 - ex_w * 2
        y1 = center[1] - h / 2 - ex_h * 2
        x2 = center[0] + w / 2 + ex_w * 2
        y2 = center[1] + h / 2 + ex_h * 2
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        new_bbox = np.array([x1, y1, x2, y2]).astype(int)
        landmarks = landmarks - new_bbox[0:2]
        bbox[0:2] = bbox[0:2] - new_bbox[0:2]
        bbox[2:4] = bbox[2:4] - new_bbox[0:2]
        data["bbox"] = bbox.tolist()
        data["keypoints"] = landmarks.tolist()
        img = img[new_bbox[1] : new_bbox[3], new_bbox[0] : new_bbox[2]]
        output_path = rewrite_path(image_path)
        data["image_path"] = output_path.as_posix()
        cv2.imwrite(output_path.as_posix(), img)
        result.append(data)
    json.dump(result, open(filename, "w"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        new_ds_path = Path(sys.argv[1])
    if not new_ds_path.exists():
        new_ds_path.mkdir()
    do("train.json")
    do("val.json")
