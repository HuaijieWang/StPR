import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


def download_file(url, destination, expected_sha256):
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_checksum(destination) == expected_sha256:
        return str(destination)
    if destination.exists() and not destination.is_file():
        raise RuntimeError(f"Not a regular file: {destination}")
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url) as source:
            length = source.headers.get("Content-Length")
            with tqdm(total=int(length) if length else None, unit="B", unit_scale=True) as progress:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
                    progress.update(len(chunk))
        if file_checksum(temporary) != expected_sha256:
            raise RuntimeError(f"SHA256 mismatch: {destination.name}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)


def file_checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
