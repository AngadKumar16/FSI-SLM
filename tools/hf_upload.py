"""Upload trained LoRA adapters to a gated (restricted-access) HuggingFace repo.

Creates the repo if needed, sets gated=manual so every access request requires
your approval, uploads the adapters + model card.

Usage:
    python tools/hf_upload.py --repo-id AngadKumar/fsi-slm-phi4-mini-lora
    python tools/hf_upload.py --repo-id ... --private   # fully private instead of gated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "artifacts" / "adapters"
MODEL_CARD = ROOT / "tools" / "hf_model_card.md"
EVAL_RESULTS = ROOT / "artifacts" / "eval_results" / "eval_results.json"


def build_readme() -> str:
    """Model card with eval results table injected."""
    card = MODEL_CARD.read_text()
    if EVAL_RESULTS.exists():
        results = json.loads(EVAL_RESULTS.read_text())
        summary = results.get("summary", {})
        lines = ["| Metric | Value |", "|---|---|"]
        for key, val in summary.items():
            lines.append(
                f"| {key} | {val:.4f} |" if isinstance(val, float) else f"| {key} | {val} |"
            )
        for fld, stats in results.get("per_field_accuracy", {}).items():
            lines.append(f"| field accuracy: {fld} | {stats['accuracy']:.4f} |")
        pred = results.get("meta", {}).get("predictor")
        if pred:
            lines.append(f"| predictor | {pred} |")
        card = card.replace("<!-- EVAL_RESULTS -->", "\n".join(lines))
    else:
        card = card.replace(
            "<!-- EVAL_RESULTS -->", "_Evaluation results pending._"
        )
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--private", action="store_true",
        help="Make repo fully private instead of gated-public.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(
        p for p in ADAPTER_DIR.iterdir()
        if p.name in ("adapters.safetensors", "adapter_config.json")
    )
    if not any(p.name == "adapters.safetensors" for p in files):
        raise SystemExit(f"No adapters.safetensors in {ADAPTER_DIR} — train first.")

    print(f"Repo: {args.repo_id} ({'private' if args.private else 'public + gated=manual'})")
    for p in files:
        print(f"  upload {p.relative_to(ROOT)} -> adapters/{p.name}")
    if args.dry_run:
        print("Dry run — nothing uploaded.")
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    if not args.private:
        # Manual gating: every download request needs your approval.
        api.update_repo_settings(args.repo_id, gated="manual")

    for p in files:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f"adapters/{p.name}",
            repo_id=args.repo_id,
        )
    api.upload_file(
        path_or_fileobj=build_readme().encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")
    print("Review access requests: https://huggingface.co/"
          f"{args.repo_id}/settings#gated-access")


if __name__ == "__main__":
    main()
