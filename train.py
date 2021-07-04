import os
import torch
from torch.utils.data import DataLoader
from datasets.landmark import Landmark
from utils.wing_loss import WingLoss
from models.slim import Slim
import time
from utils.consoler import rewrite, next_line
import visdom
import numpy as np
from benchmarks_nme import calculate_nme
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_size", default=160, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--name", default="slim", type=str)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--visdom", default=False, type=bool)
args = parser.parse_args()

if args.visdom:
    # TODO: that seems is no good (runs server in the train script)
    os.system("nohup python3 -m visdom.server --hostname 0.0.0.0 > visdom.log &")
    time.sleep(2)

viz = visdom.Visdom()

lr_decay_every_epoch = [1, 25, 35, 75, 150]
lr_value_every_epoch = [0.00001, 0.0001, 0.00005, 0.00001, 0.000001]
weight_decay_factor = 5.0e-4
l2_regularization = weight_decay_factor

input_size = (args.input_size, args.input_size)
batch_size = args.batch_size


def init_visdom():
    viz.line([0.0], [0], win="train_loss_total", opts=dict(title="train_loss_total"))
    viz.line(
        [[0.0, 0.0, 0.0, 0.0, 0.0]],
        [0],
        win="train_loss",
        opts=dict(
            title="train_loss", legend=["landmark", "pose", "leye", "reye", "mouth"]
        ),
    )
    viz.line([0.0], [0], win="eval_loss_total", opts=dict(title="eval_loss_total"))
    viz.line(
        [[0.0, 0.0, 0.0, 0.0, 0.0]],
        [0],
        win="eval_loss",
        opts=dict(
            title="eval_loss", legend=["landmark", "pose", "leye", "reye", "mouth"]
        ),
    )
    viz.line([0.], [0], win="train_acc", opts=dict(title="train_acc"))
    viz.line([0.], [0], win="eval_acc", opts=dict(title="eval_acc"))
    viz.line([0.], [0], win="lr", opts=dict(title="lr"))


class Metrics:
    def __init__(self):
        self.landmark_loss = 0
        self.loss_pose = 0
        self.leye_loss = 0
        self.reye_loss = 0
        self.mouth_loss = 0
        self.accuracy = 0
        self.counter = 0

    def update(self, landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss, accuracy):
        self.landmark_loss += landmark_loss.item()
        self.loss_pose += loss_pose.item()
        self.leye_loss += leye_loss.item()
        self.reye_loss += reye_loss.item()
        self.mouth_loss += mouth_loss.item()
        self.accuracy += accuracy.item()
        self.counter += 1

    def summary(self):
        total = (self.landmark_loss + self.loss_pose + self.leye_loss + self.reye_loss + self.mouth_loss) / self.counter
        lands = self.landmark_loss / self.counter
        pose = self.loss_pose / self.counter
        leye = self.leye_loss / self.counter
        reye = self.reye_loss / self.counter
        mouth = self.mouth_loss / self.counter
        acc = self.accuracy / self.counter
        return total, lands, pose, leye, reye, mouth, acc


def decay(epoch):
    if epoch < lr_decay_every_epoch[0]:
        return lr_value_every_epoch[0]
    if lr_decay_every_epoch[0] <= epoch < lr_decay_every_epoch[1]:
        return lr_value_every_epoch[1]
    if lr_decay_every_epoch[1] <= epoch < lr_decay_every_epoch[2]:
        return lr_value_every_epoch[2]
    if lr_decay_every_epoch[2] <= epoch < lr_decay_every_epoch[3]:
        return lr_value_every_epoch[3]
    if lr_decay_every_epoch[3] <= epoch < lr_decay_every_epoch[4]:
        return lr_value_every_epoch[4]


def calculate_loss(predict_keypoints, label_keypoints):
    landmark_label = label_keypoints[:, 0:136]
    pose_label = label_keypoints[:, 136:139]
    leye_cls_label = label_keypoints[:, 139]
    reye_cls_label = label_keypoints[:, 140]
    mouth_cls_label = label_keypoints[:, 141]
    big_mouth_cls_label = label_keypoints[:, 142]
    landmark_predict = predict_keypoints[:, 0:136]
    pose_predict = predict_keypoints[:, 136:139]
    leye_cls_predict = predict_keypoints[:, 139]
    reye_cls_predict = predict_keypoints[:, 140]
    mouth_cls_predict = predict_keypoints[:, 141]
    big_mouth_cls_predict = predict_keypoints[:, 142]
    landmark_loss = 2 * wing_loss_fn(landmark_predict, landmark_label)
    loss_pose = mse_loss_fn(pose_predict, pose_label)
    leye_loss = 0.8 * bce_loss_fn(leye_cls_predict, leye_cls_label)
    reye_loss = 0.8 * bce_loss_fn(reye_cls_predict, reye_cls_label)
    mouth_loss = bce_loss_fn(mouth_cls_predict, mouth_cls_label)
    mouth_loss_big = bce_loss_fn(big_mouth_cls_predict, big_mouth_cls_label)
    mouth_loss = 0.5 * (mouth_loss + mouth_loss_big)
    loss_sum = landmark_loss + loss_pose + leye_loss + reye_loss + mouth_loss
    return loss_sum, landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss


def calculate_accuracy(predict_keypoints, label_keypoints, sz, normolization=False):
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
        nme_all.append(nme)
    return 1 - np.mean(nme_all)


def train(epoch):
    model.train()
    metrics = Metrics()
    total_samples = 0
    start = time.time()
    print(
        "==================================Training Phase================================="
    )
    print("Current LR:{}".format(list(optim.param_groups)[0]["lr"]))
    viz.line([list(optim.param_groups)[0]["lr"]], [epoch], win="lr", update="append")
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.cuda()  # no need for CPU
        labels = labels.cuda()  # no need for CPU
        optim.zero_grad()
        preds = model(imgs)
        (
            loss,
            landmark_loss,
            loss_pose,
            leye_loss,
            reye_loss,
            mouth_loss,
        ) = calculate_loss(preds, labels)
        acc = calculate_accuracy(preds, labels, imgs.shape[-1], normolization=False)
        metrics.update(landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss, acc)
        loss.backward()
        optim.step()

        total_samples += len(imgs)
        end = time.time()
        speed = (i + 1) / (end - start)
        progress = total_samples / len(train_dataset)
        rewrite(
            "Epoch: {} Acc -- {:.4f}; Loss -- Total: {:.4f} Landmark: {:.4f} Pose: {:.4f} LEye: {:.4f} REye: {:.4f} "
            "Mouth: {:.4f} Progress: {:.4f} Speed: {:.4f}it/s".format(
                epoch,
                acc.item(),
                loss.item(),
                landmark_loss.item(),
                loss_pose.item(),
                leye_loss.item(),
                reye_loss.item(),
                mouth_loss.item(),
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
        avg_ac
    ) = metrics.summary()
    print(
        "Train Avg Acc -- {:.4f}; Avg Loss -- Total: {:.4f} Landmark: {:.4f} Poss: {:.4f} LEye: {:.4f} REye: {:.4f} "
        "Mouth: {:.4f}".format(
            avg_ac,
            avg_loss,
            avg_landmark_loss,
            avg_loss_pose,
            avg_leye_loss,
            avg_reye_loss,
            avg_mouth_loss,
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
            ]
        ],
        [epoch],
        win="train_loss",
        update="append",
    )
    viz.line([avg_ac], [epoch],  win="train_acc", update="append")


def evaluate(epoch):
    model.eval()
    metrics = Metrics()
    start = time.time()
    total_samples = 0
    print(
        "==================================Eval Phase================================="
    )
    for i, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.cuda()  # no need for CPU
        labels = labels.cuda()  # no need for CPU
        with torch.no_grad():
            preds = model(imgs)
            (
                loss,
                landmark_loss,
                loss_pose,
                leye_loss,
                reye_loss,
                mouth_loss,
            ) = calculate_loss(preds, labels)
            acc = calculate_accuracy(preds, labels, imgs.shape[-1], normolization=False)
        metrics.update(landmark_loss, loss_pose, leye_loss, reye_loss, mouth_loss, acc)
        total_samples += len(imgs)
        end = time.time()
        speed = (i + 1) / (end - start)
        progress = total_samples / len(val_dataset)
        rewrite(
            "Epoch: {} Acc -- {:.4f}; Loss -- Total: {:.4f} Landmark: {:.4f} Pose: {:.4f} LEye: {:.4f} REye: {:.4f} "
            "Mouth: {:.4f} Progress: {:.4f} Speed: {:.4f}it/s".format(
                epoch,
                acc.item(),
                loss.item(),
                landmark_loss.item(),
                loss_pose.item(),
                leye_loss.item(),
                reye_loss.item(),
                mouth_loss.item(),
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
        avg_ac,
    ) = metrics.summary()
    print(
        "Eval Avg Acc -- {:.4f}; Avg Loss  -- Total: {:.4f} Landmark: {:.4f} Poss: {:.4f} LEye: {:.4f} REye: {:.4f} "
        "Mouth: {:.4f}".format(
            avg_ac,
            avg_loss,
            avg_landmark_loss,
            avg_loss_pose,
            avg_leye_loss,
            avg_reye_loss,
            avg_mouth_loss,
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
            ]
        ],
        [epoch],
        win="eval_loss",
        update="append",
    )
    viz.line([avg_ac], [epoch], win="eval_acc", update="append")


if __name__ == "__main__":
    checkpoint = os.environ.get("PEPPA_START_CHECKPOINT", None)
    torch.backends.cudnn.benchmark = True
    train_dataset = Landmark("train.json", input_size, True)
    # num_workers=0 for CPU
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_dataset = Landmark("val.json", input_size, False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = Slim()
    model.train()
    model.cuda()  # no need for CPU
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        start_epoch = int(checkpoint.split("_")[-2]) + 1
    else:
        start_epoch = 0

    wing_loss_fn = WingLoss()
    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    init_visdom()

    optim = torch.optim.Adam(
        model.parameters(), lr=lr_value_every_epoch[0], weight_decay=5e-4
    )

    for ep in range(start_epoch, 150):
        for param_group in optim.param_groups:
            param_group["lr"] = decay(ep)
        train(ep)
        evaluate(ep)
