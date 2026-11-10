import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml

from .dataset import make_loader
from .transforms import (
    GroupCenterCrop,
    GroupMultiScaleCrop,
    GroupNormalize,
    GroupRandomHorizontalFlip,
    GroupScale,
    Stack,
    ToTorchFormatTensor,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_arguments(training):
    parser = argparse.ArgumentParser(description="StPR training" if training else "StPR evaluation")
    parser.add_argument("-conf", "--config", required=training)
    parser.add_argument("--gpu", type=int, help="GPU index; -1 selects CPU")
    if training:
        parser.add_argument("--output", help="Directory for checkpoints and metrics")
        parser.add_argument("--seed", type=int)
        parser.add_argument("--num-tasks", type=int)
    else:
        parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def read_config(path):
    with open(path, encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    return resolve_paths(conf)


def resolve_paths(conf):
    dataset = conf["dataset"]
    for key in ("frames", "frames_train", "frames_test", "pivot_split", "train_list", "test_list"):
        if dataset.get(key):
            path = Path(dataset[key]).expanduser()
            dataset[key] = str(path if path.is_absolute() else ROOT / path)
    if conf["clip_model"] != "ViT-B/16":
        path = Path(conf["clip_model"]).expanduser()
        conf["clip_model"] = str(path if path.is_absolute() else ROOT / path)
    return conf


def select_device(conf, gpu=None):
    gpu = conf["gpu"] if gpu is None else gpu
    if gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")


def seed_training(seed):
    random.seed(10 if seed is None else seed)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def build_transforms(model):
    normalize = GroupNormalize(model.input_mean, model.input_std)
    train = torchvision.transforms.Compose([
        GroupMultiScaleCrop(model.crop_size, [1, .875, .75, .66]),
        GroupRandomHorizontalFlip(),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize,
    ])
    test = torchvision.transforms.Compose([
        GroupScale(model.scale_size),
        GroupCenterCrop(model.crop_size),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize,
    ])
    return train, test


def build_loader(conf, task, transform, training, class_names=None):
    dataset = conf["dataset"]
    split = "train" if training else "test"
    root = dataset.get(f"frames_{split}", dataset.get("frames"))
    if root is None:
        raise ValueError(f"Missing frame root for {split}")
    return make_loader(
        task, root, conf["batch_size"], training, conf["num_workers"],
        conf["num_frames"], transform, dataset["name"], training, training,
        class_names=class_names,
    )


def build_test_loaders(conf, tasks, num_tasks, transform):
    seen = tasks["test"][:num_tasks]
    classes = [name for task in tasks["train"][:num_tasks] for name in task]
    loaders = [build_loader(conf, task, transform, False, classes)[0] for task in seen]
    return loaders, [len(task) for task in seen], classes
