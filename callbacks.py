from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import visdom
from clearml import Logger


class Callback(ABC):
    def on_train_start(self, data: Any, **kwargs):
        pass

    def on_train_end(self, data: Any, **kwargs):
        pass

    def on_batch_start(self, data: Any, **kwargs):
        pass

    def on_batch_end(self, data: Any, **kwargs):
        pass

    def on_epoch_start(self, epoch: int, data: Any, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, data: Any, **kwargs):
        pass

    def on_epoch_start_train(self, epoch: int, data: Any, **kwargs):
        pass

    def on_epoch_end_train(self, epoch: int, data: Any, **kwargs):
        pass

    def on_epoch_start_eval(self, epoch: int, data: Any, **kwargs):
        pass

    def on_epoch_end_eval(self, epoch: int, data: Any, **kwargs):
        pass


class ClearMLCallback(Callback):
    def on_epoch_start_train(self, epoch: int, data: Any, **kwargs):
        Logger.current_logger().report_scalar(
            title="lr", series="series", iteration=epoch, value=data
        )

    def on_epoch_end(self, epoch: int, data: Any, **kwargs):
        Logger.current_logger().report_scalar(
            title="train time",
            series="series",
            iteration=epoch,
            value=data["train_time"],
        )

    def on_epoch_end_train(self, epoch: int, data: Any, **kwargs):
        for key, value in data.items():
            Logger.current_logger().report_scalar(
                title=key, series="series", iteration=epoch, value=value
            )

    def on_epoch_end_eval(self, epoch: int, data: Any, **kwargs):
        for key, value in data.items():
            Logger.current_logger().report_scalar(
                title=key, series="series", iteration=epoch, value=value
            )


class VisdomCallback(Callback):
    def __init__(self):
        viz = visdom.Visdom()
        viz.line(
            [0.0], [0], win="train_loss_total", opts=dict(title="train_loss_total")
        )
        viz.line(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [0],
            win="train_loss",
            opts=dict(
                title="train_loss",
                legend=["landmark", "pose", "leye", "reye", "mouth", "score"],
            ),
        )
        viz.line([0.0], [0], win="eval_loss_total", opts=dict(title="eval_loss_total"))
        viz.line(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [0],
            win="eval_loss",
            opts=dict(
                title="eval_loss",
                legend=["landmark", "pose", "leye", "reye", "mouth", "score"],
            ),
        )
        viz.line([0.0], [0], win="train_acc", opts=dict(title="train_acc"))
        viz.line([0.0], [0], win="eval_acc", opts=dict(title="eval_acc"))
        viz.line([0.0], [0], win="lr", opts=dict(title="lr"))
        viz.line(
            [[0.0, 0.0]],
            [0],
            win="epoch_time",
            opts=dict(title="epoch_time, m", legend=["train_phase", "eval_phase"]),
        )
        viz.line(
            [[0.0, 0.0]],
            [0],
            win="Acc/Score train",
            opts=dict(title="Acc/Score train", legend=["acc", "score"]),
        )
        viz.line(
            [[0.0, 0.0]],
            [0],
            win="Acc/Score eval",
            opts=dict(title="Acc/Score eval", legend=["acc", "score"]),
        )
        self.viz = viz

    def on_epoch_end(self, epoch: int, data: Any, **kwargs):
        self.viz.line(
            [
                [
                    data["train_time"],
                    data["eval_time"],
                ]
            ],
            [epoch],
            win="epoch_time",
            update="append",
        )

    def on_epoch_start_train(self, epoch: int, data: Any, **kwargs):
        self.viz.line([data], [epoch], win="lr", update="append")

    def on_epoch_end_train(self, epoch: int, data: Any, **kwargs):
        self.viz.line(
            [data["avg_loss"]], [epoch], win="train_loss_total", update="append"
        )
        self.viz.line(
            [
                [
                    data["avg_landmark_loss"],
                    data["avg_loss_pose"],
                    data["avg_leye_loss"],
                    data["avg_reye_loss"],
                    data["avg_mouth_loss"],
                    data["avg_score_loss"],
                ]
            ],
            [epoch],
            win="train_loss",
            update="append",
        )
        self.viz.line([data["avg_ac"]], [epoch], win="train_acc", update="append")
        self.viz.line(
            [
                [
                    data["avg_ac"],
                    np.mean(data["scores"]),
                ]
            ],
            [epoch],
            win="Acc/Score train",
            update="append",
        )

    def on_epoch_end_eval(self, epoch: int, data: Any, **kwargs):
        self.viz.line(
            [data["avg_loss"]], [epoch], win="eval_loss_total", update="append"
        )
        self.viz.line(
            [
                [
                    data["avg_landmark_loss"],
                    data["avg_loss_pose"],
                    data["avg_leye_loss"],
                    data["avg_reye_loss"],
                    data["avg_mouth_loss"],
                    data["avg_score_loss"],
                ]
            ],
            [epoch],
            win="eval_loss",
            update="append",
        )
        self.viz.line([data["avg_ac"]], [epoch], win="eval_acc", update="append")
        self.viz.line(
            [
                [
                    data["avg_ac"],
                    np.mean(data["scores"]),
                ]
            ],
            [epoch],
            win="Acc/Score eval",
            update="append",
        )
