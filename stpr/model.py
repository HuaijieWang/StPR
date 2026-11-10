from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import clip
from .spatiotemporal import TemporalExpert


def normalize_features(x):
    return x / x.norm(dim=-1, keepdim=True)


def top1_accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    correct = pred.eq(labels.view(-1)).float().sum(0)
    return (correct * 100.0 / labels.size(0)).item()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


class StPR(nn.Module):
    def __init__(self, conf, device):
        super().__init__()
        self.conf = conf
        self.device = device
        self.num_frames = conf["num_frames"]
        self.lr = conf["lr"]
        self.epochs = conf["epochs"]
        self.double_base_epochs = conf["double_base_epochs"]
        self.logit_scale = conf["logit_scale"]
        self.spatial_logit_scale = conf["spatial_logit_scale"]
        self.fssd_weight = conf["fssd_weight"]
        self.temporal_fusion = conf["temporal_fusion_weight"]
        self.spatial_fusion = conf["spatial_fusion_weight"]
        self.text_prompt = conf["text_prompt"]
        self.num_classes = conf["dataset"]["num_classes"]
        self.num_tasks = conf["dataset"]["num_tasks"]
        self.eps = 1e-8

        adapter_config = {
            "bottleneck": conf["adapter_bottleneck"],
            "scale": conf["adapter_scale"],
            "dropout": conf["adapter_dropout"],
        }
        self.clip_model = clip.load(conf["clip_model"], device=device, adapter_config=adapter_config)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for block in self.clip_model.visual.transformer.resblocks:
            for param in block.adapter.parameters():
                param.requires_grad = True

        width = self.clip_model.visual.input_resolution
        self.crop_size = width
        self.scale_size = width * 256 // 224
        self.input_mean = [0.48145466, 0.4578275, 0.40821073]
        self.input_std = [0.26862954, 0.26130258, 0.27577711]

        temporal_conf = conf["spatiotemporal"]
        self.temporal_conf = temporal_conf
        self.spatiotemporal = TemporalExpert(
            device, temporal_conf["dim"], temporal_conf["heads"], temporal_conf["layers"], self.num_frames,
        ).to(device)
        self.temporal_experts = nn.ModuleList()
        self.task_class_counts = []
        self.temporal_anchors = []
        self.semantic_importance = None
        self.previous_spatial = None
        self.num_learned_experts = 0
        self.class_texts = None
        self.initial_task_accuracy = []

    def train_mode(self):
        self.clip_model.eval()
        self.spatiotemporal.train()
        for block in self.clip_model.visual.transformer.resblocks:
            block.adapter.train()

    def eval_mode(self):
        self.clip_model.eval()
        self.spatiotemporal.eval()

    def trainable_parameters(self):
        params = [p for p in self.spatiotemporal.parameters() if p.requires_grad]
        for block in self.clip_model.visual.transformer.resblocks:
            params.extend([p for p in block.adapter.parameters() if p.requires_grad])
        return params

    def encode_frames(self, videos, spatial_encoder=None):
        videos = videos.to(self.device, non_blocking=True)
        images = videos.view((-1, 3) + videos.size()[-2:])
        encoder = self.clip_model if spatial_encoder is None else spatial_encoder
        features = encoder.encode_image(images)
        return features.view(features.size(0) // self.num_frames, self.num_frames, -1)

    def encode_texts(self, texts):
        if self.text_prompt:
            texts = ["A good photo of a " + text.lower() for text in texts]
        tokens = clip.tokenize(list(texts)).to(self.device)
        return self.clip_model.encode_text(tokens)

    def contrastive_loss(self, labels, logits_per_video, logits_per_text):
        label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
        loss_v2t = self._directional_loss(logits_per_video, label_mask)
        loss_t2v = self._directional_loss(logits_per_text, label_mask.t())
        return (loss_v2t + loss_t2v) / 2

    def _directional_loss(self, logits, mask):
        shifted = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(shifted)
        positive = torch.sum(exp_logits * mask, dim=1)
        total = torch.sum(exp_logits, dim=1)
        return -torch.log((positive + self.eps) / (total + self.eps)).mean()

    def frame_shared_distillation(self, videos, spatial_feature):
        if self.num_learned_experts == 0 or self.semantic_importance is None:
            return spatial_feature.new_zeros(())
        with torch.no_grad():
            old_frames = self.encode_frames(videos, self.previous_spatial)
            old_spatial = normalize_features(old_frames.mean(dim=1))
            importance = self.semantic_importance
            importance = (importance - torch.min(importance)) / (torch.max(importance) - torch.min(importance) + self.eps)
            importance = importance.abs().to(device=spatial_feature.device, dtype=spatial_feature.dtype)
        drift = ((spatial_feature - old_spatial) ** 2).mean(dim=0)
        return (importance * drift).mean()

    def training_loss(self, videos, labels, texts):
        frame_features = self.encode_frames(videos)
        spatial_feature = normalize_features(frame_features.mean(dim=1))
        spatiotemporal = self.spatiotemporal(frame_features)
        text_feature = normalize_features(self.encode_texts(texts))
        video_feature = normalize_features(spatiotemporal)
        logits_st = self.logit_scale * video_feature @ text_feature.t()
        logits_s = self.spatial_logit_scale * spatial_feature @ text_feature.t()
        loss_st = self.contrastive_loss(labels, logits_st, logits_st.t())
        loss_s = self.contrastive_loss(labels, logits_s, logits_s.t())
        loss_fssd = self.frame_shared_distillation(videos, spatial_feature)
        loss = loss_st + loss_s + self.fssd_weight * loss_fssd
        return loss, loss_s, loss_fssd, video_feature

    def route(self, frame_features):
        spatial_feature = normalize_features(frame_features.mean(dim=1))
        anchors = torch.stack(self.temporal_anchors, dim=0).to(frame_features.device)
        anchors = anchors.unsqueeze(0).expand(frame_features.size(0), -1, -1)
        start = 0
        scores = []
        expert_features = []
        for index, expert in enumerate(self.temporal_experts):
            expert.eval()
            feature = normalize_features(expert(frame_features))
            expert_features.append(feature)
            temporal = normalize_features(feature - spatial_feature)
            temporal = temporal.unsqueeze(1).expand(-1, anchors.size(1), -1)
            similarity = F.cosine_similarity(temporal, anchors, dim=2)
            end = start + self.task_class_counts[index]
            score = similarity[:, start:end].max(dim=1)[0]
            start = end
            scores.append(score)
        weights = torch.stack(scores, dim=1)
        weights = (weights - torch.min(weights)) / (torch.max(weights) - torch.min(weights) + self.eps)
        mixed = None
        denom = 0
        for index, feature in enumerate(expert_features):
            weight = weights[:, index:index + 1]
            mixed = feature * weight if mixed is None else mixed + feature * weight
            denom = denom + weight
        temporal = mixed / torch.where(denom > 0, denom, torch.ones_like(denom))
        temporal = torch.where(denom > 0, temporal, torch.stack(expert_features).mean(dim=0))
        return (self.temporal_fusion * temporal + self.spatial_fusion * spatial_feature) / (self.temporal_fusion + self.spatial_fusion)

    def predict(self, videos, class_texts, text_features=None):
        frame_features = self.encode_frames(videos)
        if self.num_learned_experts > 0:
            video_feature = self.route(frame_features)
        else:
            video_feature = self.spatiotemporal(frame_features)
        video_feature = normalize_features(video_feature)
        if text_features is None:
            text_features = normalize_features(self.encode_texts(class_texts))
        return video_feature @ text_features.t()

    def train_task(self, task_id, loader, class_texts, statistics_eval_mode=False):
        if self.class_texts is None:
            self.class_texts = list(class_texts)
        else:
            self.class_texts.extend(list(class_texts))
        epochs = self.epochs * 2 if (task_id == 0 and self.double_base_epochs) else self.epochs
        self.optimizer = torch.optim.SGD(self.trainable_parameters(), lr=self.lr)
        print(f"Task {task_id + 1}/{self.num_tasks}, epochs {epochs}")
        for epoch in range(epochs):
            self.train_mode()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            loss_s = None
            loss_fssd = None
            for videos, labels, texts in loader:
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    loss, loss_s, loss_fssd, video_feature = self.training_loss(videos, labels, texts)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    prototypes = normalize_features(self.encode_texts(class_texts))
                    acc = top1_accuracy((video_feature @ prototypes.t()).float(), labels)
                loss_meter.update(loss.item(), labels.size(0))
                acc_meter.update(acc, labels.size(0))
            message = f"Epoch [{epoch + 1}/{epochs}] loss {loss_meter.avg:.4f} acc {acc_meter.avg:.2f}"
            if loss_fssd is not None and self.num_learned_experts > 0:
                message += f" fssd {(self.fssd_weight * loss_fssd).item():.4f} spatial {loss_s.item():.4f}"
            print(message)
        if statistics_eval_mode:
            self.eval_mode()
        else:
            self.train_mode()
        self._update_after_task(loader)

    @torch.no_grad()
    def _update_after_task(self, loader):
        self.previous_spatial = deepcopy(self.clip_model).requires_grad_(False)
        class_features = {}
        temporal_features = {}
        count = len(loader.dataset.class_names)
        text_bank = normalize_features(self.encode_texts(self.class_texts[-count:])).cpu()
        for videos, labels, _ in loader:
            frame_features = self.encode_frames(videos)
            spatial = normalize_features(frame_features.mean(dim=1))
            spatiotemporal = normalize_features(self.spatiotemporal(frame_features))
            temporal = normalize_features(spatiotemporal - spatial)
            spatial = spatial.cpu()
            temporal = temporal.cpu()
            labels = labels.cpu()
            for index in range(labels.size(0)):
                label = int(labels[index])
                if label not in class_features:
                    class_features[label] = []
                    temporal_features[label] = []
                class_features[label].append(spatial[index])
                temporal_features[label].append(temporal[index])
        if len(class_features) != count:
            raise ValueError("Training loader omitted a class; lower batch_size or add videos for that class")
        for label in range(count):
            feat = torch.stack(class_features[label])
            mean = feat.mean(dim=0)
            var = feat.var(dim=0, unbiased=False)
            var = var / (self.eps + var.max())
            text = text_bank[label]
            mean = (mean * text) / (torch.norm(mean) * torch.norm(text) + self.eps)
            weight = mean / (self.eps + var)
            if self.semantic_importance is None:
                self.semantic_importance = weight
            else:
                self.semantic_importance = self.semantic_importance + weight
            self.temporal_anchors.append(torch.stack(temporal_features[label]).mean(dim=0))
        self.temporal_experts.append(deepcopy(self.spatiotemporal).to(self.device).requires_grad_(False))
        self.task_class_counts.append(count)
        self.num_learned_experts += 1
        self.spatiotemporal = TemporalExpert(
            self.device,
            self.temporal_conf["dim"],
            self.temporal_conf["heads"],
            self.temporal_conf["layers"],
            self.num_frames,
        ).to(self.device)

    @torch.no_grad()
    def evaluate(self, loaders, class_counts, class_texts):
        self.eval_mode()
        total = AverageMeter()
        forgetting = AverageMeter()
        task_scores = []
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            text_features = normalize_features(self.encode_texts(class_texts))
        for task_id, (loader, num_classes) in enumerate(zip(loaders, class_counts)):
            meter = AverageMeter()
            for videos, labels, _ in loader:
                labels = labels.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    logits = self.predict(videos, class_texts, text_features)
                meter.update(top1_accuracy(logits.float(), labels), labels.size(0))
            task_scores.append(meter.avg)
            total.update(meter.avg, num_classes)
            print(f"Task {task_id + 1} accuracy {meter.avg:.3f}")
            if task_id < len(self.initial_task_accuracy) and task_id < len(loaders) - 1:
                forgetting.update(self.initial_task_accuracy[task_id] - meter.avg, num_classes)
            if task_id == len(self.initial_task_accuracy):
                self.initial_task_accuracy.append(meter.avg)
        print(f"Acc {total.avg:.3f}  BWF {forgetting.avg:.3f}")
        return {"accuracy": total.avg, "backward_forgetting": forgetting.avg, "task_accuracy": task_scores}

    def checkpoint_state(self):
        return {
            "pretrained_sha256": self.clip_model.pretrained_sha256,
            "spatial_adapters": {
                key: value.detach().cpu()
                for key, value in self.clip_model.state_dict().items()
                if ".adapter." in key
            },
            "temporal_experts": [
                {key: value.detach().cpu() for key, value in expert.state_dict().items()}
                for expert in self.temporal_experts
            ],
            "temporal_anchors": [anchor.detach().cpu() for anchor in self.temporal_anchors],
            "task_class_counts": self.task_class_counts,
            "class_texts": self.class_texts,
            "initial_task_accuracy": self.initial_task_accuracy,
        }

    def load_checkpoint_state(self, state):
        if state["pretrained_sha256"] != self.clip_model.pretrained_sha256:
            raise ValueError("The evaluation CLIP backbone differs from the training backbone")
        adapters = {
            key: value for key, value in self.clip_model.state_dict().items()
            if ".adapter." in key
        }
        if adapters.keys() != state["spatial_adapters"].keys():
            raise ValueError("Checkpoint spatial adapters do not match the model configuration")
        self.clip_model.load_state_dict(state["spatial_adapters"], strict=False)
        self.temporal_experts = nn.ModuleList()
        for weights in state["temporal_experts"]:
            expert = TemporalExpert(
                self.device, self.temporal_conf["dim"], self.temporal_conf["heads"],
                self.temporal_conf["layers"], self.num_frames,
            ).to(self.device)
            expert.load_state_dict(weights)
            self.temporal_experts.append(expert)
        self.temporal_anchors = state["temporal_anchors"]
        self.task_class_counts = state["task_class_counts"]
        self.class_texts = state["class_texts"]
        self.initial_task_accuracy = state["initial_task_accuracy"]
        self.num_learned_experts = len(self.temporal_experts)
        if (
            len(self.task_class_counts) != self.num_learned_experts
            or sum(self.task_class_counts) != len(self.temporal_anchors)
            or len(self.class_texts) != len(self.temporal_anchors)
        ):
            raise ValueError("Checkpoint experts, class counts, text labels, and anchors do not align")
        self.eval_mode()
