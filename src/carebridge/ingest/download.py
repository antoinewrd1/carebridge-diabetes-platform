"""Download source datasets from Kaggle into data/raw/."""
import subprocess
import sys

from carebridge.config import RAW, KAGGLE_DATASETS


def download(name: str, slug: str) -> None:
    target = RAW / name
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        print(f"[skip] {name} already present")
        return
    print(f"[get ] {slug} -> {target}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
        check=True,
    )


def main() -> None:
    only = sys.argv[1:] or list(KAGGLE_DATASETS)
    for name in only:
        download(name, KAGGLE_DATASETS[name])


if __name__ == "__main__":
    main()
