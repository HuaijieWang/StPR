import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_pivot_tasks(path, num_tasks, base_classes):
    # UCF101 and Kinetics-400 retain PIVOT/vCLIMB class order and video assignments.
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    classes = [name for task in data.get('train', []) for name in task]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError('PIVOT training tasks must contain each class exactly once')
    groups = partition_classes(classes, num_tasks, base_classes)
    result = {}
    for split in ('train', 'test'):
        if split not in data:
            raise ValueError(f"PIVOT split is missing '{split}'")
        videos = {name: samples for task in data[split] for name, samples in task.items()}
        if set(videos) != set(classes):
            raise ValueError(f'PIVOT {split} and train classes differ')
        result[split] = [{name: videos[name] for name in group} for group in groups]
    return result


def partition_classes(classes, num_tasks, base_classes):
    if num_tasks < 2 or not 0 < base_classes < len(classes):
        raise ValueError('Expected at least two tasks and a nonempty initial and incremental split')
    remaining, extra = divmod(len(classes) - base_classes, num_tasks - 1)
    if not remaining:
        raise ValueError('The number of incremental tasks exceeds the remaining classes')
    groups = [classes[:base_classes]]
    start = base_classes
    for index in range(num_tasks - 1):
        end = start + remaining + (index < extra)
        groups.append(classes[start:end])
        start = end
    return groups


def build_hmdb_tasks(train_list, test_list, num_tasks, base_classes, seed=99):
    random.seed(seed)
    np.random.seed(seed)
    train = read_video_list(train_list)
    test = read_video_list(test_list)
    if set(train) != set(test):
        raise ValueError('HMDB51 train and test lists must contain the same classes')
    classes = sorted(train)
    random.shuffle(classes)
    groups = partition_classes(classes, num_tasks, base_classes)
    return {
        split: [{name: videos[name] for name in group} for group in groups]
        for split, videos in (('train', train), ('test', test))
    }


def read_video_list(path):
    videos = defaultdict(list)
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = line.split()[0]
            if '/' not in sample:
                raise ValueError(f'Expected class/video on line {number} of {path}')
            name, video = sample.split('/', 1)
            videos[name].append(video)
    if not videos:
        raise ValueError(f'No videos found in {path}')
    return dict(videos)


def prepare_tasks(conf):
    dataset = conf['dataset']
    name = dataset['name']
    if name in ('UCF101', 'Kinetics400'):
        tasks = load_pivot_tasks(dataset['pivot_split'], dataset['num_tasks'], dataset['base_classes'])
    elif name == 'HMDB51':
        tasks = build_hmdb_tasks(
            dataset['train_list'], dataset['test_list'], dataset['num_tasks'],
            dataset['base_classes'], dataset['seed'],
        )
    else:
        raise ValueError('Supported datasets: UCF101, HMDB51, Kinetics400')
    classes = [name for task in tasks['train'] for name in task]
    if len(classes) != dataset['num_classes']:
        raise ValueError(f"Expected {dataset['num_classes']} classes, found {len(classes)}")
    return tasks


def make_hmdb_lists(split_directory, output_directory, split=2):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(split_directory).glob(f'*_test_split{split}.txt'))
    if len(files) != 51:
        raise ValueError(f'Expected 51 official HMDB51 split files, found {len(files)}')
    rows = {1: [], 2: []}
    for path in files:
        name = path.name.removesuffix(f'_test_split{split}.txt')
        for line in path.read_text().splitlines():
            video, membership = line.split()
            if int(membership) in rows:
                rows[int(membership)].append(f'{name}/{video}\n')
    for membership, prefix in ((1, 'train'), (2, 'test')):
        (output_directory / f'{prefix}list{split:02d}.txt').write_text(''.join(rows[membership]))
