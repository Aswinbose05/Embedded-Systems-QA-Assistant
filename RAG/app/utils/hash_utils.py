import hashlib


def get_bytes_hash(data: bytes) -> str:
    """Computes MD5 hash from raw bytes."""
    return hashlib.md5(data).hexdigest()


def get_file_hash(file_path: str) -> str:
    """Computes MD5 hash from a file path."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
