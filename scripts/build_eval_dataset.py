#!/usr/bin/env python3
"""Build a local-KB-backed agricultural QA evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raganything.eval_dataset import EvalDatasetBuildConfig, EvalDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local knowledge-base backed QA evaluation dataset."
    )
    parser.add_argument("--rag-dir", default="rag_storage_whole_book_gemini")
    parser.add_argument("--image-manifest", default=None)
    parser.add_argument("--output-dir", default="eval_dataset")
    parser.add_argument("--target-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--image-min-ratio", type=float, default=0.30)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalDatasetBuildConfig(
        rag_dir=args.rag_dir,
        output_dir=args.output_dir,
        image_manifest=args.image_manifest,
        target_size=args.target_size,
        seed=args.seed,
        candidate_multiplier=args.candidate_multiplier,
        image_min_ratio=args.image_min_ratio,
        max_workers=args.max_workers,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        vision_model=args.vision_model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    report = EvalDatasetBuilder(config).build()
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"Dataset written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
