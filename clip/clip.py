import os
from functools import lru_cache
from pathlib import Path

import torch

from .download import download_file, file_checksum
from .model import build_model
from .simple_tokenizer import SimpleTokenizer

_MODEL_URL = 'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt'


@lru_cache(maxsize=1)
def _tokenizer():
    return SimpleTokenizer()


def load(name='ViT-B/16', device='cpu', download_root=None, adapter_config=None):
    if name == 'ViT-B/16':
        root = Path(download_root or '~/.cache/clip').expanduser()
        path = download_file(_MODEL_URL, root / 'ViT-B-16.pt', _MODEL_URL.split('/')[-2])
    elif os.path.isfile(name):
        path = name
    else:
        raise ValueError('clip_model must be ViT-B/16 or a local CLIP ViT-B/16 checkpoint')
    try:
        state = torch.jit.load(path, map_location='cpu').eval().state_dict()
    except RuntimeError:
        state = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(state, adapter_config=adapter_config).to(device)
    model.pretrained_sha256 = _MODEL_URL.split("/")[-2] if name == "ViT-B/16" else file_checksum(path)
    if torch.device(device).type == 'cpu':
        model.float()
    return model


def tokenize(texts, context_length=77):
    if isinstance(texts, str):
        texts = [texts]
    tokenizer = _tokenizer()
    start = tokenizer.encoder['<|startoftext|>']
    end = tokenizer.encoder['<|endoftext|>']
    tokens = torch.zeros(len(texts), context_length, dtype=torch.long)
    for index, text in enumerate(texts):
        encoded = [start] + tokenizer.encode(text) + [end]
        if len(encoded) > context_length:
            raise ValueError(f'Text exceeds CLIP context length: {text}')
        tokens[index, :len(encoded)] = torch.tensor(encoded)
    return tokens
