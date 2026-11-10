import torch

from stpr.model import StPR
from stpr.runtime import (
    build_test_loaders, build_transforms, parse_arguments, read_config,
    resolve_paths, select_device,
)
from stpr.splits import prepare_tasks


def main():
    args = parse_arguments(training=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    conf = read_config(args.config) if args.config else resolve_paths(checkpoint["config"])
    for key in (
        "num_frames", "adapter_bottleneck", "adapter_scale", "adapter_dropout",
        "spatiotemporal", "text_prompt", "temporal_fusion_weight", "spatial_fusion_weight",
    ):
        if conf[key] != checkpoint["config"][key]:
            raise ValueError(f"Evaluation configuration differs from checkpoint: {key}")
    model = StPR(conf, select_device(conf, args.gpu))
    model.load_checkpoint_state(checkpoint["state"])
    _, transform = build_transforms(model)
    tasks = prepare_tasks(conf)
    loaders, counts, classes = build_test_loaders(conf, tasks, model.num_learned_experts, transform)
    if classes != checkpoint["class_order"] or counts != model.task_class_counts:
        raise ValueError("Evaluation task order differs from the checkpoint")
    model.evaluate(loaders, counts, model.class_texts)


if __name__ == "__main__":
    main()
