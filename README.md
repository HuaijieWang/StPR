# StPR

Spatiotemporal Preservation and Routing for Exemplar-Free Video Class-Incremental Learning (ICLR 2026).

## Environment

Python 3.12, PyTorch 2.5.1, torchvision 0.20.1.

```bash
conda env create -f environment.yml
conda activate StPR
```

To use an existing environment:

```bash
conda activate StPR
python -m pip install -r requirements.txt
```

## Datasets

Prepare RGB frames and edit the paths in `configs/`. Relative dataset paths are resolved from the repository root.

| Dataset | Configuration | Task sequence | Required metadata |
| --- | --- | --- | --- |
| UCF101 | `configs/ucf101.yaml` | 51 base classes + 5 tasks of 10 classes | `UCF101_data.pkl` |
| HMDB51 | `configs/hmdb51.yaml` | 26 base classes + 5 tasks of 5 classes | Official split-2 train/test lists; class-order seed 99 |
| Kinetics-400 | `configs/kinetics400.yaml` | 20 tasks of 20 classes | `Kinetics400_data_tasks_20.pkl` |

UCF101 and Kinetics-400 use the class order and video assignments adopted by [PIVOT](https://github.com/ojedaf/PIVOT), distributed with [vCLIMB](https://github.com/ojedaf/vCLIMB_Benchmark/tree/main/data). Place these metadata files in `data/splits/`. UCF101 merges the first five source tasks into its 51-class base task and retains the remaining task order.

Store each video's frames in `<frame_root>/<class>/<video>/frame000001.jpg`, `frame000002.jpg`, etc. `<frame_root>/<video>/` is also supported. Video folder names may retain or omit the video extension used in the metadata. Videos with fewer than eight frames are excluded. For Kinetics-400, configure the train and test frame roots separately to match the metadata.

Extract frames with FFmpeg, retaining the original frame rate:

```bash
mkdir -p /path/to/frame_root/class/video
ffmpeg -i /path/to/video.avi -q:v 2 -start_number 1 /path/to/frame_root/class/video/frame%06d.jpg
```

For HMDB51, obtain the official `*_test_split2.txt` files from the [dataset website](https://serre.lab.brown.edu/hmdb51.html) and create the two lists:

```bash
python prepare_hmdb.py --splits /path/to/testTrainMulti_7030_splits --output data/hmdb51
```

CLIP ViT-B/16 and its tokenizer vocabulary are downloaded to `~/.cache/clip` on first use. For offline execution, set `clip_model` to your local OpenAI CLIP ViT-B/16 checkpoint and set `CLIP_BPE_PATH` to the local `bpe_simple_vocab_16e6.txt.gz`. Dataset files, pretrained weights, and trained checkpoints are not included.

## Train and test

Run from the repository root:

```bash
python train.py --config configs/ucf101.yaml --gpu 0
python train.py --config configs/hmdb51.yaml --gpu 0
python train.py --config configs/kinetics400.yaml --gpu 0
```

Each command trains the sequence and evaluates all seen classes after every task. The last checkpoint and metrics are written to `outputs/<dataset>/last.pt` and `metrics.json`. Use `--output /path/to/run` to select another output directory and `--seed 10` to fix the run seed.

```bash
python test.py --checkpoint outputs/ucf101/last.pt --gpu 0
python test.py --checkpoint outputs/hmdb51/last.pt --gpu 0
python test.py --checkpoint outputs/kinetics400/last.pt --gpu 0
```

To relocate data or pretrained weights, provide `--config configs/<dataset>.yaml` to `test.py` with the updated paths and the same model and task settings.

For the other task schedules, use separate output directories:

```bash
python train.py --config configs/ucf101.yaml --num-tasks 11 --output outputs/ucf101_11
python train.py --config configs/ucf101.yaml --num-tasks 26 --output outputs/ucf101_26
python train.py --config configs/hmdb51.yaml --num-tasks 26 --output outputs/hmdb51_26
```

For 10-task Kinetics-400, set `num_tasks: 10`, `base_classes: 40`, and `pivot_split: data/splits/Kinetics400_data_tasks_10.pkl` in its configuration before training.

## Citation

```bibtex
@inproceedings{wang2026stpr,
  title={{StPR}: Spatiotemporal Preservation and Routing for Exemplar-Free Video Class-Incremental Learning},
  author={Wang, Huaijie and Cheng, De and Li, Guozhang and Xu, Zhipeng and He, Lingfeng and Li, Jie and Wang, Nannan and Gao, Xinbo},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
