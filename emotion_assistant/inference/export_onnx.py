import argparse
from pathlib import Path

import torch

from emotion_assistant.inference.predictor import load_checkpoint


def export_onnx(checkpoint_path: Path, metadata_path: Path, output_path: Path) -> None:
    model, _, metadata = load_checkpoint(checkpoint_path, metadata_path)
    model.eval()

    dummy_input = torch.zeros((1, metadata["max_length"]), dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    print(f"Exported ONNX model to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model to ONNX format.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/last.ckpt"),
        help="Path to the trained Lightning checkpoint",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/checkpoints/metadata.pkl"),
        help="Path to the metadata file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model.onnx"),
        help="Destination path for the exported ONNX model",
    )
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.metadata, args.output)


if __name__ == "__main__":
    main()
