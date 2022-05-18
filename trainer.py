import math
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from callbacks import Callback
from metrics import Metrics
from utils.wing_loss import WingLoss


class Trainer:
    def __init__(
        self,
        name: str,
        model: torch.nn.Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List[Callback]] = None,
        lr: float = 0.00001,
        initial_epoch: int = 1,
        num_epochs: int = 100,
        scheduler_mode: str = "custom_own",
        device: str = "cuda",
    ):
        self.name = name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        self.scheduler_mode = scheduler_mode
        self.num_epochs = num_epochs
        self.current_lr = lr
        self.initial_epoch = initial_epoch
        self.current_epoch = initial_epoch
        self.device = device
        self.wing_loss_fn = WingLoss()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    def run(self):
        print("start training")
        for self.current_epoch in range(self.initial_epoch, self.num_epochs):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.adjust_lr(self.current_epoch, self.current_lr)
                self.current_lr = param_group["lr"]
            train_time = self.train(self.current_epoch) / 60
            eval_time = self.evaluate(self.current_epoch) / 60
            for callback in self.callbacks:
                callback.on_epoch_end(
                    epoch=self.current_epoch,
                    data=dict(train_time=train_time, eval_time=eval_time),
                )
        print("training was end")

    def train(self, epoch):
        self.model.train()
        metrics = Metrics()
        total_samples = 0
        start = time.time()
        print(
            "============================Training Phase===========================",
            flush=True,
        )
        print(f"Current LR:{list(self.optimizer.param_groups)[0]['lr']}", flush=True)
        scores = []
        base_message = f"Train {epoch}:{self.num_epochs}"
        train_progress = tqdm(self.train_loader, desc=base_message)
        for images, labels in train_progress:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            acc, total_acc = calculate_accuracy(
                predictions, labels, images.shape[-1], normolization=False
            )
            (
                loss,
                landmark_loss,
                loss_pose,
                leye_loss,
                reye_loss,
                mouth_loss,
                score_loss,
            ) = self.calculate_loss(predictions, labels, acc)
            scores.append(float(torch.mean(predictions[:, 143])))
            metrics.update(
                landmark_loss,
                loss_pose,
                leye_loss,
                reye_loss,
                mouth_loss,
                score_loss,
                total_acc,
            )
            loss.backward()
            self.optimizer.step()

            total_samples += len(images)

            base_message = (
                f"Epoch (train): {epoch} "
                f"Acc: {total_acc.item():.4f} "
                f"Loss (total): {loss.item():.4f} "
                f"Landmark: {landmark_loss.item():.4f} "
                f"Pose: {loss_pose.item():.4f} "
                f"LEye: {leye_loss.item():.4f} "
                f"REye: {reye_loss.item():.4f} "
                f"Score: {score_loss.item():.4f}"
            )
            train_progress.set_description(base_message)

        (
            avg_loss,
            avg_landmark_loss,
            avg_loss_pose,
            avg_leye_loss,
            avg_reye_loss,
            avg_mouth_loss,
            avg_score_loss,
            avg_ac,
        ) = metrics.summary()
        print(
            f"Train Avg Acc -- {avg_ac:.4f} "
            f"Avg Loss -- Total: {avg_loss:.4f} "
            f"Landmark: {avg_landmark_loss:.4f} "
            f"Poss: {avg_loss_pose:.4f} "
            f"LEye: {avg_leye_loss:.4f} "
            f"REye: {avg_reye_loss:.4f} "
            f"Mouth: {avg_mouth_loss:.4f} "
            f"Score: {avg_score_loss:.4f}",
            flush=True,
        )
        metrics_data = dict(
            avg_loss=avg_loss,
            avg_landmark_loss=avg_landmark_loss,
            avg_loss_pose=avg_loss_pose,
            avg_leye_loss=avg_leye_loss,
            avg_reye_loss=avg_reye_loss,
            avg_mouth_loss=avg_mouth_loss,
            avg_score_loss=avg_score_loss,
            avg_ac=avg_ac,
        )
        for callback in self.callbacks:
            callback.on_epoch_end_train(epoch=epoch, data=metrics_data)
        return time.time() - start

    def evaluate(self, epoch):
        self.model.eval()
        metrics = Metrics()
        start = time.time()
        print("=============================Eval Phase===============================")
        scores = []
        base_message = f""
        val_progress = tqdm(self.val_loader, desc=base_message)
        for images, labels in val_progress:
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                predictions = self.model(images)
                acc, total_acc = calculate_accuracy(
                    predictions, labels, images.shape[-1], normolization=False
                )
                (
                    loss,
                    landmark_loss,
                    loss_pose,
                    leye_loss,
                    reye_loss,
                    mouth_loss,
                    score_loss,
                ) = self.calculate_loss(predictions, labels, acc)
            scores.append(float(torch.mean(predictions[:, 143])))
            metrics.update(
                landmark_loss,
                loss_pose,
                leye_loss,
                reye_loss,
                mouth_loss,
                score_loss,
                total_acc,
            )
            base_message = (
                f"Epoch (val): {epoch} "
                f"Acc: {total_acc.item():.4f} "
                f"Loss (total): {loss.item():.4f} "
                f"Landmark: {landmark_loss.item():.4f} "
                f"Pose: {loss_pose.item():.4f} "
                f"LEye: {leye_loss.item():.4f} "
                f"REye: {reye_loss.item():.4f} "
                f"Score: {score_loss.item():.4f}"
            )
            val_progress.set_description(base_message)
        (
            avg_loss,
            avg_landmark_loss,
            avg_loss_pose,
            avg_leye_loss,
            avg_reye_loss,
            avg_mouth_loss,
            avg_score_loss,
            avg_ac,
        ) = metrics.summary()
        print(
            f"Eval Avg Acc -- {avg_ac:.4f} "
            f"Avg Loss -- Total: {avg_loss:.4f} "
            f"Landmark: {avg_landmark_loss:.4f} "
            f"Poss: {avg_loss_pose:.4f} "
            f"LEye: {avg_leye_loss:.4f} "
            f"REye: {avg_reye_loss:.4f} "
            f"Mouth: {avg_mouth_loss:.4f} "
            f"Score: {avg_score_loss:.4f}",
            flush=True,
        )
        metrics_data = dict(
            avg_loss=avg_loss,
            avg_landmark_loss=avg_landmark_loss,
            avg_loss_pose=avg_loss_pose,
            avg_leye_loss=avg_leye_loss,
            avg_reye_loss=avg_reye_loss,
            avg_mouth_loss=avg_mouth_loss,
            avg_score_loss=avg_score_loss,
            avg_ac=avg_ac,
        )
        for callback in self.callbacks:
            callback.on_epoch_end_eval(epoch=epoch, data=metrics_data)
        self.save(metrics, epoch)
        return time.time() - start

    def save(self, metrics: Metrics, epoch: int):
        torch.save(
            self.model.state_dict(),
            open(
                f"weights/{self.name}_epoch_{epoch}_{metrics.landmark_loss / metrics.counter:.4f}.pth",
                "wb",
            ),
        )

    def adjust_lr(self, epoch, lr):
        if self.scheduler_mode == "custom_torch":
            lr_decay_every_epoch = [1, 25, 35, 75, 150]
            lr_value_every_epoch = [0.00001, 0.0001, 0.00005, 0.00001, 0.000001]
        elif self.scheduler_mode == "custom_own":
            lr_decay_every_epoch = [1, 5, 30, 50, 75, 100, 150]
            # lr_decay_every_epoch = [1, 5, 15, 25, 40, 55, 60]
            lr_value_every_epoch = [
                0.00001,
                0.0001,
                0.001,
                0.0003,
                0.0001,
                0.00001,
                0.0000001,
            ]
        elif self.scheduler_mode == "custom_tf":
            lr_decay_every_epoch = [1, 2, 100, 150, 200, 250, 300]
            lr_value_every_epoch = [
                0.00001,
                0.0001,
                0.001,
                0.0001,
                0.00001,
                0.000001,
                0.0000001,
            ]
        else:
            raise ValueError("Unsuitable mode type, change it to 'torch' or 'tf'.")
        for decay, value in zip(lr_decay_every_epoch, lr_value_every_epoch):
            if epoch <= decay:
                return value
        return lr

    def calculate_loss(
        self, predictions: torch.Tensor, gt_label: torch.Tensor, accuracy: np.ndarray
    ):
        landmark_label = gt_label[:, 0:136]
        pose_label = gt_label[:, 136:139]
        leye_cls_label = gt_label[:, 139]
        reye_cls_label = gt_label[:, 140]
        mouth_cls_label = gt_label[:, 141]
        big_mouth_cls_label = gt_label[:, 142]
        score_label = torch.FloatTensor(accuracy).to(self.device)
        landmark_predict = predictions[:, 0:136]
        pose_predict = predictions[:, 136:139]
        leye_cls_predict = predictions[:, 139]
        reye_cls_predict = predictions[:, 140]
        mouth_cls_predict = predictions[:, 141]
        big_mouth_cls_predict = predictions[:, 142]
        score_predict = predictions[:, 143]
        landmark_loss = 2 * self.wing_loss_fn(landmark_predict, landmark_label)
        loss_pose = self.mse_loss_fn(pose_predict, pose_label)
        leye_loss = 0.8 * self.bce_loss_fn(leye_cls_predict, leye_cls_label)
        reye_loss = 0.8 * self.bce_loss_fn(reye_cls_predict, reye_cls_label)
        mouth_loss = self.bce_loss_fn(mouth_cls_predict, mouth_cls_label)
        mouth_loss_big = self.bce_loss_fn(big_mouth_cls_predict, big_mouth_cls_label)
        mouth_loss = 0.5 * (mouth_loss + mouth_loss_big)
        score_loss = self.mse_loss_fn(score_predict, score_label)
        loss_sum = (
            landmark_loss + loss_pose + leye_loss + reye_loss + mouth_loss + score_loss
        )
        return (
            loss_sum,
            landmark_loss,
            loss_pose,
            leye_loss,
            reye_loss,
            mouth_loss,
            score_loss,
        )


def calculate_nme(
    gt_lands: torch.Tensor,
    checked_lands: torch.Tensor,
    dist: torch.Tensor,
    n_points: torch.Tensor,
):
    points_errors = []
    for gt, ch in zip(gt_lands, checked_lands):
        x0, y0 = gt
        x1, y1 = ch
        point_error = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / dist
        points_errors.append(point_error)
    return np.sum(points_errors) / n_points


def calculate_accuracy(
    predictions: torch.Tensor, gt_label: torch.Tensor, sz, normolization=False
):
    if not normolization:
        sz = 1
    landmark_label = gt_label[:, 0:136]
    landmark_predict = predictions[:, 0:136]
    n = landmark_label.shape[1] / 2
    nme_all = []
    for label, predict in zip(landmark_label, landmark_predict):
        label = label.reshape((-1, 2))
        predict = predict.reshape((-1, 2))
        nme = calculate_nme(label, predict, sz, n)
        nme_all.append(1 - nme)
    return np.array(nme_all), np.mean(nme_all)
