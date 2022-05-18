import tarfile
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm


def download_file(name: str, url: str, chunk_size: int = 8192) -> Path:
    path = Path("./data") / name.split(".")[0]
    path.mkdir(parents=True, exist_ok=True)
    path = path / name
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        progress = tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=f"downloading {name}",
        )
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                progress.update(len(chunk))
    return path


def download_file_and_unzip(name: str, url: str, chunk_size: int = 8192) -> Path:
    path = download_file(name=name, url=url, chunk_size=chunk_size)
    suffix = "".join(path.suffixes)
    if suffix == ".zip":
        ZipFile(path.as_posix()).extractall(path=path.parent.as_posix())
    elif suffix == ".tar.gz":
        with tarfile.open(path.as_posix()) as file:
            file.extractall(path.parent.as_posix())
    elif suffix == ".tgz":
        with tarfile.open(path.as_posix(), mode="r:gz") as file:
            file.extractall(path.parent.as_posix())
    else:
        raise ValueError("unknown archive format")
    return path.parent
