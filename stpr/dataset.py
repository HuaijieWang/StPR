import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader


def class_to_text(name, dataset_name):
    key = dataset_name.lower()
    if "ucf" in key:
        if (" " not in name) and name != "YoYo" and name != "BMX":
            return " ".join(re.findall("[A-Z][^A-Z]*", name))
        return name
    if "hmdb" in key:
        return name.replace("_", " ")
    return name


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def frames(self):
        return self._data[1]

    @property
    def num_frames(self):
        return self._data[2]

    @property
    def label(self):
        return self._data[3]

    @property
    def text(self):
        return self._data[4]


def collect_frames(task, frame_root, num_frames):
    collected = {}
    for class_name, videos in task.items():
        kept = []
        for video_name in videos:
            path_video = os.path.join(frame_root, class_name, video_name)
            if not os.path.exists(path_video):
                path_video = os.path.join(frame_root, video_name)
            if not os.path.isdir(path_video):
                candidates = [
                    Path(frame_root) / class_name / Path(video_name).stem,
                    Path(frame_root) / Path(video_name).stem,
                ]
                path_video = next((str(path) for path in candidates if path.is_dir()), path_video)
            if not os.path.isdir(path_video):
                raise FileNotFoundError(f"Missing frame directory: {class_name}/{video_name} under {frame_root}")
            frames = [name for name in os.listdir(path_video) if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png")]
            frames.sort(key=lambda x: int(x.split(".")[0].replace("frame", "").replace("img_", "")))
            if len(frames) >= num_frames:
                kept.append({"path": path_video, "frames": frames, "class_name": class_name})
        collected[class_name] = kept
    return collected


def make_loader(task, frame_root, batch_size, shuffle, num_workers, num_frames, transform,
                dataset_name, drop_last, random_shift, class_names=None):
    samples = collect_frames(task, frame_root, num_frames)
    names = list(task.keys()) if class_names is None else list(class_names)
    dataset = FrameDataset(samples, num_frames, transform, dataset_name, names, random_shift=random_shift)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=drop_last,
    )
    if len(loader) == 0:
        raise ValueError(f"Empty loader under {frame_root}; check frame paths, frame counts, and batch_size")
    return loader, dataset.texts


class FrameDataset(data.Dataset):
    def __init__(self, samples, num_frames, transform, dataset_name, class_names, random_shift=True):
        self.num_frames = num_frames
        self.new_length = 1
        self.transform = transform
        self.dataset_name = dataset_name
        self.random_shift = random_shift
        self.class_names = list(class_names)
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        self.texts = [class_to_text(name, dataset_name) for name in self.class_names]
        self.records = []
        for class_name in self.class_names:
            text = class_to_text(class_name, dataset_name)
            label = self.class_to_index[class_name]
            for video in samples.get(class_name, []):
                self.records.append(VideoRecord([
                    video["path"], video["frames"], len(video["frames"]), label, text,
                ]))

    def __len__(self):
        return len(self.records)

    def _sample_indices(self, record):
        average_duration = (record.num_frames - self.new_length + 1) // self.num_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_frames)), average_duration) + np.random.randint(average_duration, size=self.num_frames)
        elif record.num_frames > self.num_frames:
            offsets = np.sort(np.random.randint(record.num_frames - self.new_length + 1, size=self.num_frames))
        else:
            offsets = np.zeros((self.num_frames,))
        return offsets

    def _val_indices(self, record):
        if record.num_frames >= self.num_frames + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_frames)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_frames)])
        else:
            offsets = np.zeros((self.num_frames,))
        return offsets

    def __getitem__(self, index):
        record = self.records[index]
        if self.random_shift:
            indices = self._sample_indices(record)
        else:
            indices = self._val_indices(record)
        images = []
        for seg_ind in indices:
            frame_id = int(seg_ind)
            with Image.open(os.path.join(record.path, record.frames[frame_id])) as image:
                images.append(image.convert("RGB"))
        video = self.transform(images)
        return video, record.label, record.text
