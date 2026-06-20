import argparse
from pathlib import Path

from datasets import load_dataset


def download_data(output_dir: Path) -> None:
    dataset = load_dataset("go_emotions")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        output_file = output_dir / f"{split}.csv"
        dataset[split].to_csv(output_file, index=False)

    print(f"Saved GoEmotions CSV files to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the GoEmotions dataset and save it as CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save the CSV files.",
    )
    args = parser.parse_args()
    download_data(args.output_dir)


if __name__ == "__main__":
    main()
