import argparse
import json
from pathlib import Path

from emotion_assistant.inference.predictor import (
    load_checkpoint,
    load_metadata,
    load_onnx_session,
    predict_text,
    predict_text_onnx,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer emotion from a single text input."
    )
    parser.add_argument("text", type=str, help="Text to analyze")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/last.ckpt"),
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/checkpoints/metadata.pkl"),
        help="Path to saved model metadata",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Path to a saved ONNX model. If provided, ONNX inference will be used.",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Top K probabilities to return"
    )
    args = parser.parse_args()

    if args.onnx is not None:
        vocab, metadata = load_metadata(args.metadata)
        session = load_onnx_session(args.onnx)
        result = predict_text_onnx(
            args.text,
            session,
            vocab,
            metadata["max_length"],
            top_k=args.top_k,
        )
    else:
        model, vocab, metadata = load_checkpoint(args.checkpoint, args.metadata)
        result = predict_text(
            args.text, model, vocab, metadata["max_length"], top_k=args.top_k
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
