from argparse import ArgumentParser
from pathlib import Path

from emotion_assistant.data.download import download_data


def main() -> None:
    parser = ArgumentParser(description="Download the GoEmotions dataset to a local csv directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where the downloaded CSV files will be saved.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    download_data(output_dir)
    print(f"Downloaded dataset to {output_dir}")


if __name__ == "__main__":
    main()
