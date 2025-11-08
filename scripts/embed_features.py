#!/usr/bin/env python3
"""Generate sequence embeddings using a protein language model."""

from __future__ import annotations

import argparse
import math
import pathlib

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from crispr_design_agent.utils.io import ensure_dir
from crispr_design_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path("data/processed/dms.parquet"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("features/protT5"))
    parser.add_argument("--model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def batched(iterable, batch_size):
    total = len(iterable)
    for start in range(0, total, batch_size):
        yield iterable[start : start + batch_size], start


def main() -> None:
    args = parse_args()
    configure_logging()
    df = pd.read_parquet(args.input)
    sequences = df["sequence"].dropna().astype(str).tolist()
    if args.limit:
        sequences = sequences[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ensure_dir(args.output_dir)
    total_batches = math.ceil(len(sequences) / args.batch_size)
    for batch, start_idx in tqdm(
        batched(sequences, args.batch_size),
        total=total_batches,
        desc="Embedding",
    ):
        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        chunk_path = args.output_dir / f"embeddings_{start_idx:07d}.pt"
        torch.save({"start": start_idx, "embeddings": embeddings}, chunk_path)


if __name__ == "__main__":
    main()
