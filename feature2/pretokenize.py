#!/usr/bin/env python3
"""
pretokenize.py
--------------

Convert a raw-text Parquet file (column “text”) into a new Parquet file
(column “input_ids”) where every row is a *fixed-length* list of
(sequence_length + 1) token-ids:

    python3 pretokenize.py \
        --input  /capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet \
        --output /iopsstor/scratch/cscs/$USER/datasets/train_data_tok.parquet \
        --tokenizer unsloth/Mistral-Nemo-Base-2407-bnb-4bit \
        --sequence-length 200 \
        --batch-size 4096 \
        --flush-rows 25000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="Parquet file with a string column named 'text'")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination Parquet file (overwritten if exists!)")
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer name or path")
    p.add_argument("--sequence-length", type=int, required=True,
                   help="Tokens per row *excluding* the leading BOS")
    p.add_argument("--batch-size", type=int, default=8192,
                   help="Rows read from the source Parquet in each Arrow batch")
    p.add_argument("--flush-rows", type=int, default=25000,
                   help="Write to disk after compiling this many chunks")
    return p


def main() -> None:
    args = get_parser().parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tok.model_max_length = 10**9              # disable built-in truncation
    transformers.logging.set_verbosity_error()

    bos_id = tok.bos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer doesn't have a BOS token. Can't proceed!")

    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    pad_id = tok.pad_token_id
    assert pad_id is not None


    schema = pa.schema([("input_ids", pa.list_(pa.int32()))])
    if args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.output, schema, use_dictionary=False)


    src = pq.ParquetFile(args.input)
    buf: List[List[int]] = []
    total_rows = 0
    seq_plus_one = args.sequence_length + 1

    for rec_batch in tqdm(
        src.iter_batches(batch_size=args.batch_size, columns=["text"]),
        desc="Tokenising",
    ):
        # Arrow StringArray -> list[str] without extra copy
        texts = [s.as_py() for s in rec_batch.column(0)]

        # Batch-tokenise -> List[List[int]]
        tokenised = tok(texts, add_special_tokens=False)["input_ids"]

        for ids in tokenised:
            # BOS
            tokens = [bos_id] + ids
            # fixed-length chunking
            for i in range(0, len(tokens), seq_plus_one):
                chunk = tokens[i : i + seq_plus_one]
                if len(chunk) < seq_plus_one:
                    chunk.extend([pad_id] * (seq_plus_one - len(chunk)))
                buf.append(chunk)

            # flush
            if len(buf) >= args.flush_rows:
                _flush(writer, buf)
                total_rows += len(buf)
                buf.clear()

    # final flush
    if buf:
        _flush(writer, buf)
        total_rows += len(buf)

    writer.close()
    print(f"Wrote {total_rows:,} rows to {args.output}", file=sys.stderr)


def _flush(writer: pq.ParquetWriter, rows: List[List[int]]) -> None:
    """Write the buffered rows to Parquet and return."""
    arr = pa.array(rows, pa.list_(pa.int32()))
    tbl = pa.Table.from_arrays([arr], names=["input_ids"])
    writer.write_table(tbl)


if __name__ == "__main__":
    main()
