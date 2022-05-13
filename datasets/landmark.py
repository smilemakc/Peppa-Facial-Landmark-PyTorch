import copy
import logging
import random
from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
import orjson as json
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentation import Rotate_aug, Affine_aug, Mirror, Padding_aug, Img_dropout
from utils.headpose import get_head_pose
from utils.turbo.TurboJPEG import TurboJPEG
from utils.visual_augmentation import ColorDistort, pixel_jitter

logger = logging.getLogger(__name__)
jpeg = TurboJPEG()

symmetry = [
    (0, 16),
    (1, 15),
    (2, 14),
    (3, 13),
    (4, 12),
    (5, 11),
    (6, 10),
    (7, 9),
    (8, 8),
    (17, 26),
    (18, 25),
    (19, 24),
    (20, 23),
    (21, 22),
    (31, 35),
    (32, 34),
    (36, 45),
    (37, 44),
    (38, 43),
    (39, 42),
    (40, 47),
    (41, 46),
    (48, 54),
    (49, 53),
    (50, 52),
    (55, 59),
    (56, 58),
    (60, 64),
    (61, 63),
    (65, 67),
]
base_extend_range = [0.2, 0.3]


class DataInfo:
    def __init__(self, file_path: Union[Path, str]):
        self.file_path = Path(file_path)
        self.metas = []
        self.load_data()

    def load_data(self):
        logger.info(f"starts reading annotation file at {self.file_path.as_posix()}")
        with self.file_path.open("r") as f:
            train_json_list = json.loads(f.read())
        self.metas = train_json_list
        logger.info(f"reading annotation file was ends ok")

    def get_all_sample(self):
        logger.info("starts shuffling")
        random.shuffle(self.metas)
        logger.info("shuffling was ends ok")
        return self.metas


class Landmark(Dataset):
    def __init__(
        self,
        annotation_file_path: Union[Path, str],
        input_size=(160, 160),
        training_flag=True,
    ):
        super(Landmark, self).__init__()
        self.counter = 0
        self.time_counter = 0
        self.training_flag = training_flag
        self.raw_data_set_size = None
        self.color_augmentor = ColorDistort()
        self.items = self.parse_file(annotation_file_path)
        self.input_size = input_size

    def __getitem__(self, item):
        """Data augmentation function."""
        dp = self.items[item]
        image_path = dp["image_path"]
        keypoints = dp["keypoints"]
        bbox = dp["bbox"]
        if Path(image_path).suffix.lower() in (".jpg", ".jpeg"):
            image = jpeg.imread(image_path)
        else:
            image = cv2.imread(image_path)
        if image is None or not image.size:
            logger.warning(f"empty image at {image_path}")
            is_last = item + 1 == len(self.items)
            item = 0 if is_last else item + 1
            return self[item]
        label = np.array(keypoints, dtype=np.float).reshape((-1, 2))
        bbox = np.array(bbox)
        crop_image, label = self.augmentation_crop(
            image, bbox, label, self.training_flag
        )
        rng = np.random.default_rng(seed=42)
        if self.training_flag:
            if rng.random() > 0.5:
                crop_image, label = Mirror(crop_image, label=label, symmetry=symmetry)
            if rng.random() > 0.0:
                angle = rng.uniform(-45, 45)
                crop_image, label = Rotate_aug(crop_image, label=label, angle=angle)
            if rng.random() > 0.5:
                strength = rng.uniform(0, 50)
                crop_image, label = Affine_aug(
                    crop_image, strength=strength, label=label
                )
            if rng.random() > 0.5:
                crop_image = self.color_augmentor(crop_image)
            if rng.random() > 0.5:
                crop_image = pixel_jitter(crop_image, 15)
            if rng.random() > 0.5:
                crop_image = Img_dropout(crop_image, 0.2)
            if rng.random() > 0.5:
                crop_image = Padding_aug(crop_image, 0.3)
        reprojectdst, euler_angle = get_head_pose(label, crop_image)
        PRY = euler_angle.reshape([-1]).astype(np.float32) / 90.0
        cla_label = np.zeros([4])
        if dp["left_eye_close"]:
            cla_label[0] = 1
        if dp["right_eye_close"]:
            cla_label[1] = 1
        if dp["mouth_close"]:
            cla_label[2] = 1
        if dp["big_mouth_open"]:
            cla_label[3] = 1
        crop_image_height, crop_image_width, _ = crop_image.shape
        label = label.astype(np.float32)
        label[:, 0] = label[:, 0] / crop_image_width
        label[:, 1] = label[:, 1] / crop_image_height

        crop_image = crop_image.astype(np.float32)
        label = label.reshape([-1]).astype(np.float32)
        cla_label = cla_label.astype(np.float32)
        label = np.concatenate([label, PRY, cla_label], axis=0)

        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.transpose(crop_image, (2, 0, 1)).astype(np.float32)
        return crop_image, label

    def __len__(self) -> int:
        return len(self.items)

    def parse_file(self, annotation_file_path: Union[Path, str]) -> List[dict]:
        data_info = DataInfo(annotation_file_path)
        all_samples = data_info.get_all_sample()
        self.raw_data_set_size = len(all_samples)
        print("Raw Samples: " + str(self.raw_data_set_size))
        balanced_samples = all_samples
        if self.training_flag:
            balanced_samples = self.balance(all_samples)
            print("Balanced Samples: " + str(len(balanced_samples)))
        return balanced_samples

    def balance(self, samples: List[dict]) -> List[dict]:
        balanced_samples = copy.deepcopy(samples)
        lar_count = 0
        for ann in tqdm(samples, desc="balancing samples"):
            bbox = ann["bbox"]
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            if bbox_width < 50 or bbox_height < 50:
                balanced_samples.remove(ann)
            left_eye_close = ann["left_eye_close"]
            right_eye_close = ann["right_eye_close"]
            if left_eye_close or right_eye_close:
                for i in range(10):
                    balanced_samples.append(ann)
            if ann["small_eye_distance"]:
                for i in range(20):
                    balanced_samples.append(ann)
            if ann["small_mouth_open"]:
                for i in range(20):
                    balanced_samples.append(ann)
            if ann["big_mouth_open"]:
                for i in range(50):
                    balanced_samples.append(ann)
            if left_eye_close and not right_eye_close:
                for i in range(40):
                    balanced_samples.append(ann)
                lar_count += 1
            if not left_eye_close and right_eye_close:
                for i in range(40):
                    balanced_samples.append(ann)
                lar_count += 1
        return balanced_samples

    def augmentation_crop(
        self, img: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray, is_training=True
    ):
        bbox = (
            np.array(bbox)
            .reshape(
                4,
            )
            .astype(np.float32)
        )
        border = max(img.shape[0], img.shape[1])
        center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        bbox += border
        center += border
        landmarks[:, :2] += border
        gt_width = bbox[2] - bbox[0]
        gt_height = bbox[3] - bbox[1]
        crop_width_half = gt_width * (1 + base_extend_range[0] * 2) // 2
        crop_height_half = gt_height * (1 + base_extend_range[1] * 2) // 2
        if is_training:
            min_x = int(
                center[0]
                - crop_width_half
                + random.uniform(-base_extend_range[0], base_extend_range[0]) * gt_width
            )
            max_x = int(
                center[0]
                + crop_width_half
                + random.uniform(-base_extend_range[0], base_extend_range[0]) * gt_width
            )
            min_y = int(
                center[1]
                - crop_height_half
                + random.uniform(-base_extend_range[1], base_extend_range[1])
                * gt_height
            )
            max_y = int(
                center[1]
                + crop_height_half
                + random.uniform(-base_extend_range[1], base_extend_range[1])
                * gt_height
            )
        else:
            min_x = int(center[0] - crop_width_half)
            max_x = int(center[0] + crop_width_half)
            min_y = int(center[1] - crop_height_half)
            max_y = int(center[1] + crop_height_half)
        landmarks[:, 0] = landmarks[:, 0] - min_x
        landmarks[:, 1] = landmarks[:, 1] - min_y
        bordered_img = cv2.copyMakeBorder(
            img,
            border,
            border,
            border,
            border,
            borderType=cv2.BORDER_CONSTANT,
            value=[127.0, 127.0, 127.0],
        )
        img = bordered_img[min_y:max_y, min_x:max_x, :]
        crop_image_height, crop_image_width, _ = img.shape
        landmarks[:, 0] = landmarks[:, 0] / crop_image_width
        landmarks[:, 1] = landmarks[:, 1] / crop_image_height
        interp_methods = [
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4,
        ]
        interp_method = random.choice(interp_methods)
        img = cv2.resize(
            img, (self.input_size[0], self.input_size[1]), interpolation=interp_method
        )
        landmarks[:, 0] = landmarks[:, 0] * self.input_size[0]
        landmarks[:, 1] = landmarks[:, 1] * self.input_size[1]
        return img, landmarks
