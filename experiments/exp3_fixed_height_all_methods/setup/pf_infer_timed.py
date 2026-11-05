"""
Phyloformer inference with per-alignment timing and GPU-memory measurement.

This replaces `infer_alns.py` for the evaluation notebooks. It does the same job
(alignment in -> pairwise distance matrix out) but ALSO records, for every alignment:
  * infer_seconds      - time of the model's forward pass only (model loading excluded,
                         GPU synchronised before and after the clock is read)
  * peak_alloc_mb      - peak GPU memory the tensors actually needed (includes model weights)
  * peak_reserved_mb   - peak GPU memory PyTorch held (what decides whether a run fits)
  * status             - ok / oom / error

It lives next to infer_alns.py (setup_pipeline.py copies it there) so it can import
from it and from the `phyloformer` package.

Usage (run from the Phyloformer folder):
  python pf_infer_timed.py WEIGHTS ALNDIR -o MATRIXDIR --stats STATS.csv
"""
import argparse
import csv
import os
import time
from glob import glob
from pathlib import Path

import torch  # type:ignore

from infer_alns import has_fasta_ext, vec_to_phylip
from phyloformer.data import load_alignment
from phyloformer.model import Phyloformer

MB = 1024 * 1024


def is_oom(err):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_cls is not None and isinstance(err, oom_cls)) or "out of memory" in str(err).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("alndir")
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--stats", required=True, help="CSV file to write the per-alignment stats to")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"

    # --- load model exactly as infer_alns.py does ---
    ckpt = torch.load(args.weights, map_location=device)
    params = ckpt["hyper_parameters"]
    params["device"] = device
    model = Phyloformer(**params)
    model.load_state_dict(
        {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k != "model.seq2pair"},
        strict=False,
    )
    model = model.to(device)
    # Same fix as the old sed patch on infer_alns.py: seq2pair is not always moved with the model.
    if hasattr(model, "seq2pair"):
        model.seq2pair = model.seq2pair.to(device)
    model.eval()

    os.makedirs(args.outdir, exist_ok=True)
    paths = sorted(glob(os.path.join(os.path.abspath(args.alndir), "*")))
    for p in paths:
        if not has_fasta_ext(p):
            raise ValueError(f"Input files must be fasta files (.fa or .fasta). Got {p}")

    rows = []

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        # Warm-up on the first alignment (not recorded): the very first GPU call pays one-off
        # start-up costs that would otherwise be charged to alignment number 1.
        if paths:
            try:
                aln0, _ = load_alignment(paths[0])
                model(aln0[None, :].to(device).float())
                sync()
            except Exception as e:  # noqa: BLE001 - OOM here is recorded properly in the real pass
                if use_cuda:
                    torch.cuda.empty_cache()

        for p in paths:
            stem = Path(p).stem
            row = {"id": stem, "status": "ok", "infer_seconds": float("nan"),
                   "peak_alloc_mb": float("nan"), "peak_reserved_mb": float("nan")}
            preds = None
            try:
                aln, ids = load_alignment(p)
                x = aln[None, :].to(device).float()
                if use_cuda:
                    torch.cuda.reset_peak_memory_stats()
                sync()
                t0 = time.perf_counter()
                preds = model(x)
                sync()
                row["infer_seconds"] = time.perf_counter() - t0
                if use_cuda:
                    row["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / MB
                    row["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / MB
                _, phylip = vec_to_phylip(preds, ids)
                with open(os.path.join(args.outdir, f"{stem}.phy"), "w") as fh:
                    fh.write(phylip)
            except Exception as e:  # noqa: BLE001
                row["status"] = "oom" if is_oom(e) else "error: " + str(e)[:200].replace("\n", " ")
                if use_cuda:
                    row["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / MB
                    row["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / MB
            finally:
                preds = None
                if use_cuda:
                    torch.cuda.empty_cache()
            rows.append(row)

    with open(args.stats, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "status", "infer_seconds", "peak_alloc_mb", "peak_reserved_mb"])
        w.writeheader()
        w.writerows(rows)
    print(f"Phyloformer: {sum(r['status']=='ok' for r in rows)}/{len(rows)} alignments done on {device}")


if __name__ == "__main__":
    main()
