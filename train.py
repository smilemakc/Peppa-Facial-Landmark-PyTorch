import os
import torch
from torch.utils.data import DataLoader
from datasets.landmark import Landmark
from utils.wing_loss import WingLoss
from models.slim import Slim, SlimScore
import time
from utils.consoler import rewrite, next_line
import visdom
import numpy as np
from benchmarks_nme import calculate_nme
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_size", default=160, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--name", default="slim", type=str)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--num_epochs", default=300, type=int)
parser.add_argument("--lr", default=0.00001, type=float,
                    help="base learning rate")
parser.add_argument("--scheduler", default="custom_torch", type=str,
                    help="custom_torch || custom_tf")
parser.add_argument("--device", default="cuda",
                    help="device (cpu, cuda or cuda_ids)")
parser.add_argument("--checkpoint", default=None, type=str,
                    help="pretrained checkpoint weights path")

args, unknown = parser.parse_known_args()

# !python3 -m visdom.server at the terminal for starting Visdom server

input_size = (args.input_size, args.input_size)
batch_size = args.batch_size


def init_visdom():
    viz.line([0.0], [0], win="train_loss_total",
             opts=dict(title="train_loss_total"))
    viz.line(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [0],
        win="train_loss",
        opts=dict(
            title="train_loss",
            legend=["landmark", "pose", "leye", "reye", "mouth", "score"]
        ),
    )
    viz.line([0.0], [0], win="eval_loss_total",
             opts=dict(title="eval_loss_total"))
    viz.line(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [0],
        win="eval_loss",
        opts=dict(
            title="eval_loss",
            legend=["landmark", "pose", "leye", "reye", "mouth", "score"]
        ),
    )
    viz.line([0.0], [0], win="train_acc", opts=dict(title="train_acc"))
    viz.line([0.0], [0], win="eval_acc", opts=dict(title="eval_acc"))
    viz.line([0.0], [0], win="lr", opts=dict(title="lr"))
    viz.line(
        [[0.0, 0.0]],
        [0],
        win="epoch_time",
        opts=dict(
            title="epoch_time, m", legend=["train_phase", "eval_phase"]
        ),
    )
    viz.line(
        [[0.0, 0.0]],
        [0],
        win="Acc/Score train",
        opts=dict(
            title="Acc/Score train",
            legend=["acc", "score"]
        ),
    )
    viz.line(
        [[0.0, 0.0]],
        [0],
        win="Acc/Score eval",
        opts=dict(
            title="Acc/Score eval",
            legend=["acc", "score"]
        ),
    )


def adjust_lr(epoch, lr, mode="custom_torch"):
    if mode == "custom_torch":
        lr_decay_every_epoch = [1, 25, 35, 75, 150]
        lr_value_every_epoch = [0.00001, 0.0001, 0.00005, 0.00001, 0.000001]
    elif mode == "custom_own":
        lr_decay_every_epoch = [1, 5, 30, 50, 75, 100, 150]
        # lr_decay_every_epoch = [1, 5, 15, 25, 40, 55, 60]
        lr_value_every_epoch = [0.00001, 0.0001, 0.001, 0.0003, 0.0001, 0.00001,
                                0.0000001]
    elif mode == "custom_tf":
        lr_decay_every_epoch = [1, 2, 100, 150, 200, 250, 300]
        lr_value_every_epoch = [0.00001, 0.0001, 0.001, 0.0001, 0.00001,
                                0.000001, 0.0000001]
    else:
        raise ValueError("Unsuitable mode type, change it to 'torch' or 'tf'.")
    for decay, value in zip(lr_decay_every_epoch, lr_value_every_epoch):
        if epoch <= decay:
            return value
    return lr


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

    def update(self, landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss,
               score_loss, accuracy):
        self.landmark_loss += landmark_loss.item()
        self.loss_pose += loss_pose.item()
        self.leye_loss += leye_loss.item()
        self.reye_loss += reye_loss.item()
        self.mouth_loss += mouth_loss.item()
        self.score_loss += score_loss.item()
        self.accuracy += accuracy.item()
        self.counter += 1

    def summary(self):
        total = (
                 self.landmark_loss + self.loss_pose + self.leye_loss +
                 self.reye_loss + self.mouth_loss + self.score_loss
                ) / self.counter
        lands = self.landmark_loss / self.counter
        pose = self.loss_pose / self.counter
        leye = self.leye_loss / self.counter
        reye = self.reye_loss / self.counter
        mouth = self.mouth_loss / self.counter
        score = self.score_loss / self.counter
        acc = self.accuracy / self.counter
        return total, lands, pose, leye, reye, mouth, score, acc


def calculate_loss(predict_keypoints, label_keypoints, accuracy):
    landmark_label = label_keypoints[:, 0:136]
    pose_label = label_keypoints[:, 136:139]
    leye_cls_label = label_keypoints[:, 139]
    reye_cls_label = label_keypoints[:, 140]
    mouth_cls_label = label_keypoints[:, 141]
    big_mouth_cls_label = label_keypoints[:, 142]
    score_label = torch.FloatTensor(accuracy).to(device)
    landmark_predict = predict_keypoints[:, 0:136]
    pose_predict = predict_keypoints[:, 136:139]
    leye_cls_predict = predict_keypoints[:, 139]
    reye_cls_predict = predict_keypoints[:, 140]
    mouth_cls_predict = predict_keypoints[:, 141]
    big_mouth_cls_predict = predict_keypoints[:, 142]
    score_predict = predict_keypoints[:, 143]
    landmark_loss = 2 * wing_loss_fn(landmark_predict, landmark_label)
    loss_pose = mse_loss_fn(pose_predict, pose_label)
    leye_loss = 0.8 * bce_loss_fn(leye_cls_predict, leye_cls_label)
    reye_loss = 0.8 * bce_loss_fn(reye_cls_predict, reye_cls_label)
    mouth_loss = bce_loss_fn(mouth_cls_predict, mouth_cls_label)
    mouth_loss_big = bce_loss_fn(big_mouth_cls_predict, big_mouth_cls_label)
    mouth_loss = 0.5 * (mouth_loss + mouth_loss_big)
    score_loss = mse_loss_fn(score_predict, score_label)
    loss_sum = landmark_loss + loss_pose + leye_loss + reye_loss + mouth_loss + score_loss
    return loss_sum, landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss, score_loss


def calculate_accuracy(predict_keypoints, label_keypoints, sz,
                       normolization=False):
    if not normolization:
        sz = 1
    landmark_label = label_keypoints[:, 0:136]
    landmark_predict = predict_keypoints[:, 0:136]
    n = landmark_label.shape[1] / 2
    nme_all = []
    for label, predict in zip(landmark_label, landmark_predict):
        label = label.reshape((-1, 2))
        predict = predict.reshape((-1, 2))
        nme = calculate_nme(label, predict, sz, n)
        nme_all.append(1 - nme)
    return nme_all, np.mean(nme_all)


def train(epoch):
    model.train()
    metrics = Metrics()
    total_samples = 0
    start = time.time()
    print(
        "============================Training Phase==========================="
    )
    print(f"Current LR:{list(optimizer.param_groups)[0]['lr']}")
    viz.line([list(optimizer.param_groups)[0]['lr']], [epoch], win="lr",
             update="append")
    scores = []
    progress = tqdm(train_loader)
    for imgs, labels in progress:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        acc, total_acc = calculate_accuracy(preds, labels, imgs.shape[-1],
                                            normolization=False)
        (
            loss,
            landmark_loss,
            loss_pose,
            leye_loss,
            reye_loss,
            mouth_loss,
            score_loss,
        ) = calculate_loss(preds, labels, acc)
        scores.append(float(torch.mean(preds[:, 143])))
        metrics.update(landmark_loss, loss_pose, leye_loss, reye_loss,
                       mouth_loss, score_loss, total_acc)
        loss.backward()
        optimizer.step()

        total_samples += len(imgs)
        end = time.time()
        speed = (i + 1) / (end - start)
        progress = total_samples / len(train_loader)
        rewrite(
            "Epoch: {} Acc -- {:.4f}; Loss -- Total: {:.4f} Landmark: {:.4f} "
            "Pose: {:.4f} LEye: {:.4f} REye: {:.4f} Mouth: {:.4f} "
            "Score: {:.4f} Progress: {:.4f} Speed: {:.4f}it/s {}".format(
                epoch,
                total_acc.item(),
                loss.item(),
                landmark_loss.item(),
                loss_pose.item(),
                leye_loss.item(),
                reye_loss.item(),
                mouth_loss.item(),
                score_loss.item(),
                progress,
                speed,
                f"{len(train_loader)}/{i}"
            )
        )

    next_line()
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
        "Train Avg Acc -- {:.4f}; Avg Loss -- Total: {:.4f} "
        "Landmark: {:.4f} Poss: {:.4f} LEye: {:.4f} REye: {:.4f} "
        "Mouth: {:.4f} Score: {:.4f}".format(
            avg_ac,
            avg_loss,
            avg_landmark_loss,
            avg_loss_pose,
            avg_leye_loss,
            avg_reye_loss,
            avg_mouth_loss,
            avg_score_loss,
        )
    )

    viz.line([avg_loss], [epoch], win="train_loss_total", update="append")
    viz.line(
        [
            [
                avg_landmark_loss,
                avg_loss_pose,
                avg_leye_loss,
                avg_reye_loss,
                avg_mouth_loss,
                avg_score_loss,
            ]
        ],
        [epoch],
        win="train_loss",
        update="append",
    )
    viz.line([avg_ac], [epoch], win="train_acc", update="append")
    viz.line(
        [
            [
                avg_ac,
                np.mean(scores),
            ]
        ],
        [epoch],
        win="Acc/Score train",
        update="append",
    )
    return time.time() - start


def evaluate(epoch):
    model.eval()
    metrics = Metrics()
    start = time.time()
    total_samples = 0
    print(
        "=============================Eval Phase==============================="
    )
    scores = []
    for i, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(imgs)
            acc, total_acc = calculate_accuracy(preds, labels, imgs.shape[-1],
                                                normolization=False)
            (
                loss,
                landmark_loss,
                loss_pose,
                leye_loss,
                reye_loss,
                mouth_loss,
                score_loss,
            ) = calculate_loss(preds, labels, acc)
        scores.append(float(torch.mean(preds[:, 143])))
        metrics.update(landmark_loss, loss_pose, leye_loss, reye_loss,
                       mouth_loss, score_loss, total_acc)
        total_samples += len(imgs)
        end = time.time()
        speed = (i + 1) / (end - start)
        progress = total_samples / len(val_dataset)
        rewrite(
            "Epoch: {} Acc -- {:.4f}; Loss -- Total: {:.4f} Landmark: {:.4f} "
            "Pose: {:.4f} LEye: {:.4f} REye: {:.4f} Mouth: {:.4f} "
            "Score: {:.4f} Progress: {:.4f} Speed: {:.4f}it/s".format(
                epoch,
                total_acc.item(),
                loss.item(),
                landmark_loss.item(),
                loss_pose.item(),
                leye_loss.item(),
                reye_loss.item(),
                mouth_loss.item(),
                score_loss.item(),
                progress,
                speed,
            )
        )

    next_line()
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
        "Eval Avg Acc -- {:.4f}; Avg Loss  -- Total: {:.4f} "
        "Landmark: {:.4f} Poss: {:.4f} LEye: {:.4f} "
        "REye: {:.4f} Mouth: {:.4f} Score: {:.4f}".format(
            avg_ac,
            avg_loss,
            avg_landmark_loss,
            avg_loss_pose,
            avg_leye_loss,
            avg_reye_loss,
            avg_mouth_loss,
            avg_score_loss,
        )
    )
    torch.save(
        model.state_dict(),
        open(
            f"weights/{args.name}_{input_size[0]}x{input_size[0]}_epoch_{epoch}_{avg_landmark_loss:.4f}.pth",
            "wb",
        ),
    )

    viz.line([avg_loss], [epoch], win="eval_loss_total", update="append")
    viz.line(
        [
            [
                avg_landmark_loss,
                avg_loss_pose,
                avg_leye_loss,
                avg_reye_loss,
                avg_mouth_loss,
                avg_score_loss,
            ]
        ],
        [epoch],
        win="eval_loss",
        update="append",
    )
    viz.line([avg_ac], [epoch], win="eval_acc", update="append")
    viz.line(
        [
            [
                avg_ac,
                np.mean(scores),
            ]
        ],
        [epoch],
        win="Acc/Score eval",
        update="append",
    )
    return time.time() - start


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_dataset = Landmark("train.json", input_size, True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Landmark("val.json", input_size, False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    model = SlimScore()
    model.train()

    if args.device != "cpu" and torch.cuda.is_available():
        if len(args.device) > 1:
            device = "cuda"
            model = torch.nn.DataParallel(model, device_ids=[args.device])
        else:
            device = f"cuda:{args.device}"
            model.to(device)
    else:
        device = "cpu"
        model.to(device)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
        start_epoch = int(args.checkpoint.split("_")[-2]) + 1
    else:
        start_epoch = 0

    wing_loss_fn = WingLoss()
    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )

    viz = visdom.Visdom()
    init_visdom()

    current_lr = args.lr

    for ep in range(start_epoch, args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = adjust_lr(ep, current_lr, mode=args.scheduler)
            current_lr = param_group["lr"]
        train_time = train(ep) / 60
        eval_time = evaluate(ep) / 60
        viz.line(
            [
                [
                    train_time,
                    eval_time,
                ]
            ],
            [ep],
            win="epoch_time",
            update="append",
        )
