#!/usr/bin/env python3
"""Upload a local model file to a Hugging Face repo using `huggingface_hub`.

Example:
  python scripts/upload_hf.py \
      --repo he-projectile/empathy-helper-model \
      --file outputs/checkpoints/last.ckpt

Requires: pip install huggingface-hub
"""
import argparse
import os
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Hugging Face repo id (e.g. user/repo)")
    p.add_argument("--file", required=True, help="Local path to the file to upload")
    p.add_argument("--path-in-repo", default=None, help="Destination path inside the repo")
    p.add_argument("--token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if token is None:
        raise SystemExit("Provide a token via --token or set HF_TOKEN environment variable")

    api = HfApi()
    path_in_repo = args.path_in_repo or os.path.basename(args.file)

    print(f"Uploading {args.file} -> {args.repo}/{path_in_repo}")
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=path_in_repo,
        repo_id=args.repo,
        token=token,
    )

    print("Upload finished.")


if __name__ == "__main__":
    main()
