from argparse import ArgumentParser
from pathlib import Path

try:
    import tensorrt as trt
except ImportError as exc:
    raise ImportError(
        "TensorRT is required for TensorRT export. Install it separately from NVIDIA "
        "or use a matching wheel for your environment."
    ) from exc


def build_engine(onnx_path: Path, engine_path: Path, max_batch_size: int = 1) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open('rb') as model_file:
        if not parser.parse(model_file.read()):
            raise RuntimeError('Failed to parse ONNX model: ' + onnx_path.as_posix())

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    shape = network.get_input(0).shape
    if shape[0] == -1:
        profile.set_shape(input_name, (1, shape[1]), (max_batch_size, shape[1]), (max_batch_size, shape[1]))
        config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError('Failed to build TensorRT engine')

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())


def main() -> None:
    parser = ArgumentParser(description='Convert ONNX model to TensorRT engine.')
    parser.add_argument(
        '--onnx',
        type=Path,
        required=True,
        help='Path to the ONNX model file.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for the TensorRT engine.',
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=4,
        help='Maximum batch size for the TensorRT engine.',
    )
    args = parser.parse_args()
    build_engine(args.onnx, args.output, args.max_batch_size)
    print(f'Exported TensorRT engine to {args.output}')


if __name__ == '__main__':
    main()
