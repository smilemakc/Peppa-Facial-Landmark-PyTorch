from typing import Tuple

import torch


class Metrics:
    def __init__(self):
        self.landmark_loss = 0
        self.loss_pose = 0
        self.leye_loss = 0
        self.reye_loss = 0
        self.mouth_loss = 0
        self.score_loss = 0
        self.accuracy = 0
        self.counter = 0

    def update(
        self,
        landmark_loss: torch.Tensor,
        loss_pose: torch.Tensor,
        leye_loss: torch.Tensor,
        reye_loss: torch.Tensor,
        mouth_loss: torch.Tensor,
        score_loss: torch.Tensor,
        accuracy: torch.Tensor,
    ):
        self.landmark_loss += landmark_loss.item()
        self.loss_pose += loss_pose.item()
        self.leye_loss += leye_loss.item()
        self.reye_loss += reye_loss.item()
        self.mouth_loss += mouth_loss.item()
        self.score_loss += score_loss.item()
        self.accuracy += accuracy.item()
        self.counter += 1

    def summary(self) -> Tuple:
        total = (
            self.landmark_loss
            + self.loss_pose
            + self.leye_loss
            + self.reye_loss
            + self.mouth_loss
            + self.score_loss
        ) / self.counter
        lands = self.landmark_loss / self.counter
        pose = self.loss_pose / self.counter
        leye = self.leye_loss / self.counter
        reye = self.reye_loss / self.counter
        mouth = self.mouth_loss / self.counter
        score = self.score_loss / self.counter
        acc = self.accuracy / self.counter
        return total, lands, pose, leye, reye, mouth, score, acc
