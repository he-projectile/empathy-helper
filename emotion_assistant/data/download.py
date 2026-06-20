from pathlib import Path

from datasets import load_dataset


def download_data(output_dir: Path) -> Path:
    dataset = load_dataset("go_emotions")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        output_file = output_dir / f"{split}.csv"
        dataset[split].to_csv(output_file, index=False)

    return output_dir
