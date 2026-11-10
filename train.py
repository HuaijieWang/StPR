import json
from pathlib import Path

import torch

from stpr.model import StPR
from stpr.runtime import (
    ROOT, build_loader, build_test_loaders, build_transforms, parse_arguments,
    read_config, seed_training, select_device,
)
from stpr.splits import prepare_tasks


def main():
    args = parse_arguments(training=True)
    conf = read_config(args.config)
    if args.num_tasks is not None:
        conf["dataset"]["num_tasks"] = args.num_tasks
        if conf["dataset"]["name"] == "Kinetics400":
            conf["dataset"]["base_classes"] = conf["dataset"]["num_classes"] // args.num_tasks
    seed_training(args.seed)
    model = StPR(conf, select_device(conf, args.gpu))
    train_transform, test_transform = build_transforms(model)
    tasks = prepare_tasks(conf)
    dataset = conf["dataset"]
    directory = Path(args.output) if args.output else ROOT / "outputs" / dataset["name"].lower()
    directory.mkdir(parents=True, exist_ok=True)
    history = []
    for task_id, task in enumerate(tasks["train"]):
        loader, class_texts = build_loader(conf, task, train_transform, training=True)
        model.train_task(task_id, loader, class_texts, conf["statistics_eval_mode"])
        test_loaders, counts, classes = build_test_loaders(conf, tasks, task_id + 1, test_transform)
        metrics = model.evaluate(test_loaders, counts, model.class_texts)
        metrics["task"] = task_id + 1
        history.append(metrics)
        result = {
            "config": conf,
            "seed": args.seed,
            "class_order": classes,
            "history": history,
            "final_accuracy": metrics["accuracy"],
            "average_accuracy": sum(stage["accuracy"] for stage in history) / len(history),
        }
        (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.save({
            "config": conf,
            "class_order": classes,
            "history": history,
            "state": model.checkpoint_state(),
        }, directory / "last.pt")


if __name__ == "__main__":
    main()
