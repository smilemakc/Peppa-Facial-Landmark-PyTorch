import os
import random
from pathlib import Path

import numpy as np
import json
import traceback

from faceds.readers.pts import PTSReader
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


"""
i decide to merge more data from CelebA, the data anns will be complex, so json maybe a better way. 
"""

data_dir = os.environ.get("PEPPA_DATASETS_PATH", "ds")  # points to your director,300w

all_files_json = "all_files.json"
train_json = "train.json"
val_json = "val.json"
img_size = 160
eye_close_thres = 0.02
mouth_close_thres = 0.02
big_mouth_open_thres = 0.08

pic_list = [
    x for x in Path(data_dir).rglob("*.*") if x.suffix in [".jpg", ".jpeg", ".png"]
]

random.shuffle(pic_list)
ratio = 0.95

json_list = []
pts_reader = PTSReader()
for pic_path in tqdm(pic_list):
    one_image_ann = {}
    one_image_ann["image_path"] = pic_path.as_posix()
    pts_path = pic_path.parent / pic_path.name.replace(pic_path.suffix, ".pts")
    label = pts_reader.read(pts_path).reshape((-1, 2))
    one_image_ann["keypoints"] = label
    bbox = [
        float(np.min(label[:, 0])),
        float(np.min(label[:, 1])),
        float(np.max(label[:, 0])),
        float(np.max(label[:, 1])),
    ]

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    left_eye_close = (
        np.sqrt(
            np.square(label[37, 0] - label[41, 0])
            + np.square(label[37, 1] - label[41, 1])
        )
        / bbox_height
        < eye_close_thres
        or np.sqrt(
            np.square(label[38, 0] - label[40, 0])
            + np.square(label[38, 1] - label[40, 1])
        )
        / bbox_height
        < eye_close_thres
    )
    right_eye_close = (
        np.sqrt(
            np.square(label[43, 0] - label[47, 0])
            + np.square(label[43, 1] - label[47, 1])
        )
        / bbox_height
        < eye_close_thres
        or np.sqrt(
            np.square(label[44, 0] - label[46, 0])
            + np.square(label[44, 1] - label[46, 1])
        )
        / bbox_height
        < eye_close_thres
    )
    small_eye_distance = (
        np.sqrt(
            np.square(label[36, 0] - label[45, 0])
            + np.square(label[36, 1] - label[45, 1])
        )
        / bbox_width
        < 0.5
    )
    small_mouth_open = (
        np.sqrt(
            np.square(label[62, 0] - label[66, 0])
            + np.square(label[62, 1] - label[66, 1])
        )
        / bbox_height
        > 0.15
    )
    big_mouth_open = (
        np.sqrt(
            np.square(label[62, 0] - label[66, 0])
            + np.square(label[62, 1] - label[66, 1])
        )
        / img_size
        > big_mouth_open_thres
    )
    mouth_close = (
        np.sqrt(
            np.square(label[61, 0] - label[67, 0])
            + np.square(label[61, 1] - label[67, 1])
        )
        / img_size
        < mouth_close_thres
        or np.sqrt(
            np.square(label[62, 0] - label[66, 0])
            + np.square(label[62, 1] - label[66, 1])
        )
        / img_size
        < mouth_close_thres
        or np.sqrt(
            np.square(label[63, 0] - label[65, 0])
            + np.square(label[63, 1] - label[65, 1])
        )
        / img_size
        < mouth_close_thres
    )
    one_image_ann["left_eye_close"] = bool(left_eye_close)
    one_image_ann["right_eye_close"] = bool(right_eye_close)
    one_image_ann["small_eye_distance"] = bool(small_eye_distance)
    one_image_ann["small_mouth_open"] = bool(small_mouth_open)
    one_image_ann["big_mouth_open"] = bool(big_mouth_open)
    one_image_ann["mouth_close"] = bool(mouth_close)

    one_image_ann["bbox"] = bbox
    one_image_ann["attr"] = None
    json_list.append(one_image_ann)

train_list = json_list[: int(ratio * len(json_list))]
val_list = json_list[int(ratio * len(json_list)) :]

with open(train_json, "w") as f:
    json.dump(train_list, f, indent=2, cls=NumpyEncoder)

with open(val_json, "w") as f:
    json.dump(val_list, f, indent=2, cls=NumpyEncoder)
