"""
Shared helper functions for the fixed-height, all-methods study.

Used by the three notebooks:
  1_Generate_Trees_and_Sequences  (CPU)  - simulate + save trees and sequences
  2_Run_Phyloformer               (GPU)  - Phyloformer on the saved data
  3_Run_Other_Methods             (CPU)  - FastME, FastTree, IQ-TREE on the saved data

Folder layout inside the team repo (everything for this study lives under *_all_methods,
so earlier rounds are never touched):

  simulated_data/fixed_height_all_methods/
      trees/true/clock_rate=R/tree_size=N/<i>_<N>_tips.nwk
      sequences/clock_rate=R/model=LG|Cherry/tree_size=N/seq_length=L/<i>_<N>_tips.fa|fasta
      trees/inferred/clock_rate=R/model=M/tree_size=N/seq_length=L/method=X/<i>_<N>_tips.nwk
  results/fixed_height_all_methods/
      treesize_seqlength_rate{R}_{lg|cherry}_{phyloformer|baselines}_results.csv
      failures_rate{R}_{generate|phyloformer|baselines}.csv

Only needs pandas + numpy (both preinstalled on Colab). Everything else is the repo's own tools.
"""
import csv
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

STUDY = "fixed_height_all_methods"


class GpuNotRecordedError(RuntimeError):
    """Raised when GPU memory could not be measured. The notebooks let this one stop the whole run:
    if the GPU is not being measured, every remaining cell would be wasted."""


TREE_EXTS = (".nwk", ".newick")
ALN_EXTS = (".fa", ".fasta")


# ----------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------
def data_root(team_repo):
    return os.path.join(team_repo, "simulated_data", STUDY)


def true_trees_dir(team_repo, rate, size):
    return os.path.join(data_root(team_repo), "trees", "true", f"clock_rate={rate}", f"tree_size={size}")


def sequences_dir(team_repo, rate, model, size, seqlen):
    return os.path.join(data_root(team_repo), "sequences", f"clock_rate={rate}", f"model={model}",
                        f"tree_size={size}", f"seq_length={seqlen}")


def inferred_dir(team_repo, rate, model, size, seqlen, method):
    return os.path.join(data_root(team_repo), "trees", "inferred", f"clock_rate={rate}", f"model={model}",
                        f"tree_size={size}", f"seq_length={seqlen}", f"method={method}")


def results_dir(team_repo):
    return os.path.join(team_repo, "results", STUDY)


def results_filename(rate, model, stage):
    """stage is 'phyloformer' or 'baselines'. Separate files per stage so two sessions running
    at the same time never write to (and never git-conflict on) the same CSV."""
    return f"treesize_seqlength_rate{rate}_{model.lower()}_{stage}_results.csv"


# ----------------------------------------------------------------------------------------------
# Small file helpers
# ----------------------------------------------------------------------------------------------
def clean_dir(path):
    """Wipe and recreate a folder (stops leftovers from an earlier run leaking into this one)."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def clean_dir_no_create(path):
    """Wipe a folder but do NOT recreate it (alisim.py insists on creating its own output folder)."""
    shutil.rmtree(path, ignore_errors=True)


def list_files(folder, exts):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(tuple(exts)))


def stem(filename):
    return os.path.splitext(filename)[0]


def _run(cmd, cwd=None, timeout=None, stdout=None):
    """Run a command quietly; returns (returncode, stderr_text). Never raises on non-zero exit."""
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, stdout=stdout if stdout else subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
        return p.returncode, (p.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        return -9, "timeout"


# ----------------------------------------------------------------------------------------------
# Newick handling (no ete3 needed: only branch lengths are read or rescaled)
# ----------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"\(|\)|,|;|:[0-9eE+\-.]+|[^(),:;]+")
_LEN = re.compile(r":([0-9eE+\-.]+)")


def newick_stats(newick):
    """
    Returns (tip_depths, total_length) for a newick string.
    tip_depths = distance from the root to every tip (the root's own branch, if any, is not counted).
    total_length = sum of all branch lengths below the root.
    """
    root = {"c": [], "d": 0.0}
    cur, stack = root, []
    for tok in _TOKEN.findall(newick.strip()):
        if tok == "(":
            child = {"c": [], "d": 0.0}
            cur["c"].append(child)
            stack.append(cur)
            cur = child
        elif tok == ",":
            child = {"c": [], "d": 0.0}
            stack[-1]["c"].append(child)
            cur = child
        elif tok == ")":
            cur = stack.pop()
        elif tok == ";":
            break
        elif tok.startswith(":"):
            cur["d"] = float(tok[1:])
    depths, total = [], 0.0
    todo = [(root, 0.0)]
    while todo:
        node, depth = todo.pop()
        if not node["c"]:
            depths.append(depth)
        for ch in node["c"]:
            total += ch["d"]
            todo.append((ch, depth + ch["d"]))
    return depths, total


def tree_stats(path):
    with open(path) as fh:
        depths, total = newick_stats(fh.read())
    return {"true_tree_height": max(depths), "true_total_length": total}


def rescale_newick_file(path_in, path_out, factor):
    """Multiply every branch length in a newick file by `factor` (strict clock = one constant)."""
    with open(path_in) as fh:
        text = fh.read()
    text = _LEN.sub(lambda m: ":" + format(float(m.group(1)) * factor, ".10g"), text)
    with open(path_out, "w") as fh:
        fh.write(text)


def normalise_and_scale_trees(in_dir, out_dir, target_height, clock_rate):
    """
    For every tree: rescale so its height (longest root-to-tip distance) equals target_height,
    then multiply by clock_rate. Done in one pass: factor = target_height / height * clock_rate.
    The final root-to-tip distance is therefore target_height x clock_rate.
    """
    os.makedirs(out_dir, exist_ok=True)
    for f in list_files(in_dir, TREE_EXTS):
        height = tree_stats(os.path.join(in_dir, f))["true_tree_height"]
        if height <= 0:
            raise ValueError(f"{f} has zero height - cannot normalise it")
        rescale_newick_file(os.path.join(in_dir, f), os.path.join(out_dir, f),
                            target_height / height * clock_rate)


def check_tree_heights(trees_dir, expected_height, rel_tol=1e-3):
    """
    Stops the run (raises) if any tree's height (longest root-to-tip distance) is not the expected
    value. Call before saving/pushing a batch of trees. Also reports how many trees are not
    ultrametric (tips at different distances from the root) - information only, not an error.
    """
    heights, bad, non_ultra = [], [], 0
    for f in list_files(trees_dir, TREE_EXTS):
        with open(os.path.join(trees_dir, f)) as fh:
            depths, _ = newick_stats(fh.read())
        h = max(depths)
        heights.append(h)
        if abs(h - expected_height) > rel_tol * expected_height:
            bad.append(f)
        if (h - min(depths)) > rel_tol * h:
            non_ultra += 1
    if not heights:
        raise RuntimeError(f"No trees found in {trees_dir}")
    print(f"  tree heights: {len(heights)} trees, min={min(heights):.6f} max={max(heights):.6f} "
          f"(expected {expected_height:.6f}); non-ultrametric trees: {non_ultra}")
    if bad:
        raise ValueError(f"{len(bad)} trees have the wrong height: {bad[:5]}")


# ----------------------------------------------------------------------------------------------
# Stage 1: trees and sequences
# ----------------------------------------------------------------------------------------------
def get_or_make_true_trees(pf_dir, team_repo, rate, size, n_trees, tree_height):
    """
    Returns the folder of saved true trees for (rate, size). If a complete set is already saved
    (e.g. after a disconnect) it is reused, so sequences are always simulated from the SAME trees
    that are saved. Otherwise fresh trees are simulated, rescaled to the fixed height, multiplied
    by the clock rate, checked, and saved. If trees are regenerated, any sequences saved for the
    old trees at this size are deleted (they would no longer match).
    """
    out = true_trees_dir(team_repo, rate, size)
    if len(list_files(out, TREE_EXTS)) == n_trees:
        print(f"  true trees for tree_size={size}: already saved ({n_trees}), reusing them")
        return out

    for model in ("LG", "Cherry"):  # old sequences no longer match the new trees
        shutil.rmtree(os.path.join(data_root(team_repo), "sequences", f"clock_rate={rate}",
                                   f"model={model}", f"tree_size={size}"), ignore_errors=True)

    raw = os.path.join(pf_dir, "data", "fh_tmp", "trees_raw")
    clean_dir(raw)
    rc, err = _run(["python", "simulate_trees.py", "-n", str(n_trees), "-t", str(size),
                    "--type", "birth-death", "-o", raw], cwd=pf_dir)
    if len(list_files(raw, TREE_EXTS)) != n_trees:
        raise RuntimeError(f"simulate_trees.py made {len(list_files(raw, TREE_EXTS))} of {n_trees} trees. {err}")
    clean_dir(out)
    normalise_and_scale_trees(raw, out, tree_height, rate)
    check_tree_heights(out, tree_height * rate)
    return out


def alignment_length(aln_dir):
    """Length of the first sequence of the first alignment (what was ACTUALLY produced)."""
    files = list_files(aln_dir, ALN_EXTS)
    if not files:
        return None
    seq, seen = "", False
    for line in open(os.path.join(aln_dir, files[0])):
        if line.startswith(">"):
            if seen:
                break
            seen = True
        else:
            seq += line.strip()
    return len(seq)


def generate_sequences(pf_dir, team_repo, rate, model, size, seqlen, n_trees):
    """
    Simulate alignments for the saved true trees of this tree size, for one model and one length,
    and save them. Skips if a complete set is already saved. Returns a list of tree ids that did
    NOT get an alignment (empty list = all good).
    """
    out = sequences_dir(team_repo, rate, model, size, seqlen)
    if len(list_files(out, ALN_EXTS)) == n_trees:
        print(f"  {model} tree_size={size} seq_length={seqlen}: already saved, skipping")
        return []

    true_dir = true_trees_dir(team_repo, rate, size)
    # Same invocation as the proven notebook: working copies inside the Phyloformer folder.
    work_trees = os.path.join(pf_dir, "data", "fh_tmp", "trees")
    work_aln = os.path.join(pf_dir, "data", "fh_tmp", "alignments")
    clean_dir(work_trees)
    for f in list_files(true_dir, TREE_EXTS):
        shutil.copy(os.path.join(true_dir, f), work_trees)
    clean_dir_no_create(work_aln)  # the simulators create this folder themselves
    rel_trees = os.path.relpath(work_trees, pf_dir)
    rel_aln = os.path.relpath(work_aln, pf_dir)

    if model == "LG":
        cmd = ["python", "alisim.py", rel_trees, "-o", rel_aln, "-l", str(seqlen), "-g", "GC", "-s", "LG",
               "-i", "bin/bin_linux/iqtree_2.2.0", "-p", "2", "--allow-duplicate-sequences"]
    elif model == "Cherry":
        cmd = ["bash", "bin/simcherry.sh", rel_trees, rel_aln, str(seqlen)]
    else:
        raise ValueError(f"Unknown model {model}")
    rc, err = _run(cmd, cwd=pf_dir)

    made = list_files(work_aln, ALN_EXTS)
    clean_dir(out)
    for f in made:
        shutil.copy(os.path.join(work_aln, f), out)
    missing = sorted(set(stem(f) for f in list_files(true_dir, TREE_EXTS)) - set(stem(f) for f in made))
    actual = alignment_length(out)
    if actual is not None and actual != seqlen:
        print(f"  NOTE: requested length {seqlen}, alignments actually have {actual} (Cherry rounds odd lengths down)")
    if missing:
        print(f"  WARNING: {len(missing)} of {n_trees} alignments were not produced. {err}")
    return missing


# ----------------------------------------------------------------------------------------------
# Alignment statistics (duplicates etc.)
# ----------------------------------------------------------------------------------------------
def read_fasta(path):
    seqs, name = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                seqs[name] = []
            elif line and name is not None:
                seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def sequence_stats(path):
    """n_seqs, n_unique_sequences, n_duplicates (= extra copies = n_seqs - n_unique), seq_length_actual."""
    seqs = read_fasta(path)
    n, u = len(seqs), len(set(seqs.values()))
    length = len(next(iter(seqs.values()))) if seqs else 0
    return {"n_seqs": n, "n_unique_sequences": u, "n_duplicates": n - u, "seq_length_actual": length}


def add_context_columns(df, true_dir, aln_dir):
    """Adds, per tree id: true tree height / total length, and alignment duplicate counts."""
    aln_by_id = {stem(f): f for f in list_files(aln_dir, ALN_EXTS)}
    rows = []
    for i in df["id"].astype(str).unique():  # unique: the same tree appears once per method
        r = {"id": i}
        tpath = os.path.join(true_dir, i + ".nwk")
        r.update(tree_stats(tpath) if os.path.exists(tpath) else {})
        if i in aln_by_id:
            r.update(sequence_stats(os.path.join(aln_dir, aln_by_id[i])))
        rows.append(r)
    ctx = pd.DataFrame(rows)
    return df.astype({"id": str}).merge(ctx, on="id", how="left")


# ----------------------------------------------------------------------------------------------
# Metrics: KF, RF (and normalised / weighted RF) and MAE of pairwise distances
# ----------------------------------------------------------------------------------------------
def compute_metrics(pf_dir, true_dir, pred_dir, prefix):
    """
    Compares predicted trees with the true trees using the repo's phylocompare tool.
    Returns one row per tree: rf, norm_rf, weighted_rf, kf_score, n_tips, and
    mae = mean absolute error over all pairwise tip-to-tip distances (true vs predicted).
    Trees are matched by file name; a predicted tree with no true partner (or vice versa) is skipped.
    Returns None if nothing could be compared.
    """
    topo, dist = f"{prefix}_topo.csv", f"{prefix}_dist.csv"
    for f in (topo, dist):
        if os.path.exists(f):
            os.remove(f)
    _run([os.path.join(pf_dir, "bin", "bin_linux", "phylocompare"), "-t", "-d", "-n", "-o", prefix,
          true_dir, pred_dir])
    try:
        t = pd.read_csv(topo)
        d = pd.read_csv(dist)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    if t.empty:
        return None
    d["abs_err"] = (d["ref_dist"] - d["cmp_dist"]).abs()
    mae = d.groupby("id")["abs_err"].mean().rename("mae").reset_index()
    out = t.merge(mae, on="id", how="left").drop(columns=["marker"], errors="ignore")
    out["id"] = out["id"].astype(str)
    for f in (topo, dist):  # the distance file is large - not needed any more
        os.remove(f)
    return out


# ----------------------------------------------------------------------------------------------
# Results files (upsert) and failures
# ----------------------------------------------------------------------------------------------
def save_results(df, params, filename, out_dir):
    """
    Saves rows tagged with the run parameters. If the file already holds rows with EXACTLY the same
    parameters (a re-run), those old rows are replaced, not duplicated. Other rows are untouched.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df = df.copy()
    for k, v in params.items():
        df[k] = v
    df["saved_at"] = pd.Timestamp.now().isoformat()
    if os.path.exists(path):
        old = pd.read_csv(path)
        mask = pd.Series(True, index=old.index)
        for k, v in params.items():
            mask &= (old[k] == v) if k in old.columns else False
        if mask.sum():
            print(f"  replacing {int(mask.sum())} old rows for {params}")
        df = pd.concat([old[~mask], df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"  saved -> {filename} ({len(df)} rows in file)")


def save_failures(out_dir, rate, stage, cell, rows):
    """
    Writes this grid cell's failures to failures_rate{R}_{stage}.csv (replacing the cell's old rows,
    so a successful re-run clears them). `cell` = dict(model, tree_size, seq_length);
    `rows` = list of dict(method, tree_id, reason).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"failures_rate{rate}_{stage}.csv")
    new = pd.DataFrame([{**cell, "stage": stage, **r} for r in rows])
    if os.path.exists(path):
        old = pd.read_csv(path)
        mask = pd.Series(True, index=old.index)
        for k, v in cell.items():
            mask &= (old[k] == v)
        old = old[~mask]
        new = pd.concat([old, new], ignore_index=True) if len(new) else old
    if len(new) or os.path.exists(path):
        new.to_csv(path, index=False)
    if rows:
        print(f"  !! {len(rows)} failure(s) logged -> {os.path.basename(path)}")


def cell_done(out_dir, filename, params, methods, n_trees):
    """True if the results file already has n_trees rows for every method at these parameters."""
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path)
    mask = pd.Series(True, index=d.index)
    for k, v in params.items():
        mask &= (d[k] == v) if k in d.columns else False
    d = d[mask]
    return all((d["method"] == m).sum() >= n_trees for m in methods)


# ----------------------------------------------------------------------------------------------
# Stage 2: Phyloformer (GPU)
# ----------------------------------------------------------------------------------------------
def _gpu_used_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return float(r.stdout.strip().split("\n")[0])
    except Exception:  # noqa: BLE001
        return None


class GpuPoller:
    """Reads nvidia-smi ~10x a second in a background thread and keeps the peak. Used as an
    outside cross-check on the in-process PyTorch measurements."""

    def __init__(self, interval=0.1):
        self.interval, self.peak, self.baseline, self._stop = interval, None, None, threading.Event()

    def __enter__(self):
        self.baseline = _gpu_used_mb()
        self.peak = self.baseline
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            v = _gpu_used_mb()
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    @property
    def peak_minus_baseline(self):
        if self.peak is None or self.baseline is None:
            return None
        return self.peak - self.baseline


def fastme_from_matrix(pf_dir, matrix_path, tree_path):
    """Phyloformer's last step: build a tree from its distance matrix. Returns seconds taken (or None)."""
    t0 = time.perf_counter()
    rc, _ = _run([os.path.join(pf_dir, "bin", "bin_linux", "fastme"), "-i", matrix_path, "-o", tree_path,
                  "--nni", "--spr"])
    dt = time.perf_counter() - t0
    return dt if rc == 0 and os.path.exists(tree_path) else None


def run_phyloformer_on_dirs(pf_dir, aln_dir, true_dir, out_trees, checkpoint, work_dir, allow_cpu=False):
    """
    Phyloformer on one folder of alignments. Returns (df, failures).

    Per tree the results hold:
      infer_seconds   forward pass only (model load excluded)
      fastme_seconds  FastME building the tree from Phyloformer's distances
      runtime_seconds infer + fastme  (this is what the paper's Phyloformer timing is: PF + FastME)
      peak_alloc_mb, peak_reserved_mb   from inside the inference process (PyTorch)
      gpu_nvsmi_peak_mb   from nvidia-smi for the whole cell (peak minus the level before starting)

    Raises GpuNotRecordedError if the GPU numbers come back missing or zero - a silent 0.0 must
    never be saved again. (allow_cpu=True skips that check; it exists only for testing off-GPU.)
    """
    matrices = os.path.join(work_dir, "pf_matrices")
    stats_csv = os.path.join(work_dir, "pf_stats.csv")
    clean_dir(matrices)
    clean_dir(out_trees)
    if os.path.exists(stats_csv):
        os.remove(stats_csv)

    with GpuPoller() as poll:
        rc, err = _run(["python", "pf_infer_timed.py", checkpoint, aln_dir, "-o", matrices, "--stats", stats_csv],
                       cwd=pf_dir)
    if not os.path.exists(stats_csv):
        raise RuntimeError(f"Phyloformer inference did not run: {err}")
    stats = pd.read_csv(stats_csv)

    failures, fm_secs = [], {}
    for _, r in stats.iterrows():
        if r["status"] != "ok":
            failures.append({"method": "Phyloformer", "tree_id": r["id"],
                             "reason": ("out of GPU memory" if r["status"] == "oom" else r["status"])})
            continue
        secs = fastme_from_matrix(pf_dir, os.path.join(matrices, r["id"] + ".phy"),
                                  os.path.join(out_trees, r["id"] + ".nwk"))
        if secs is None:
            failures.append({"method": "Phyloformer", "tree_id": r["id"], "reason": "FastME produced no tree"})
        else:
            fm_secs[r["id"]] = secs

    # The GPU check comes BEFORE anything is saved or scored.
    if not allow_cpu:
        ok = stats[stats["status"] == "ok"]
        bad = (ok["peak_alloc_mb"].isna() | (ok["peak_alloc_mb"] <= 0) | ok["peak_reserved_mb"].isna()
               | (ok["peak_reserved_mb"] <= 0)).any()
        smi = poll.peak_minus_baseline
        if bad or smi is None or smi <= 0:
            raise GpuNotRecordedError(
                f"GPU memory was NOT recorded properly (PyTorch numbers ok={not bad}, nvidia-smi peak={smi}). "
                "Is this notebook on a GPU runtime? Nothing was saved.")

    df = compute_metrics(pf_dir, true_dir, out_trees, os.path.join(work_dir, "verify_pf"))
    if df is None:
        return None, failures or [{"method": "Phyloformer", "tree_id": "ALL", "reason": "no trees produced"}]
    df = df.merge(stats, on="id", how="left")
    df["fastme_seconds"] = df["id"].map(fm_secs)
    df["runtime_seconds"] = df["infer_seconds"] + df["fastme_seconds"]
    df["gpu_nvsmi_peak_mb"] = poll.peak_minus_baseline
    df["method"] = "Phyloformer"
    df = add_context_columns(df, true_dir, aln_dir)
    return df, failures


def run_phyloformer_cell(pf_dir, team_repo, rate, model, size, seqlen, n_trees, checkpoint, work_dir,
                         allow_cpu=False):
    """Phyloformer on the saved alignments of one grid cell (see run_phyloformer_on_dirs)."""
    return run_phyloformer_on_dirs(
        pf_dir,
        sequences_dir(team_repo, rate, model, size, seqlen),
        true_trees_dir(team_repo, rate, size),
        inferred_dir(team_repo, rate, model, size, seqlen, "Phyloformer"),
        checkpoint, work_dir, allow_cpu)


def sanity_check_phyloformer(pf_dir, work_dir):
    """
    Runs Phyloformer on the repo's own 20 test alignments and checks the whole chain works:
    the mean KF distance must come out near 0.333 (the repo's published benchmark) and the
    GPU numbers must be real. Raises if not - fix the environment before starting any real run.
    """
    td = os.path.join(pf_dir, "data", "testdata")
    df, failures = run_phyloformer_on_dirs(
        pf_dir, os.path.join(td, "msas"), os.path.join(td, "trees"),
        os.path.join(work_dir, "sanity_trees"), "models/pf.ckpt", os.path.join(work_dir, "sanity"))
    kf = df["kf_score"].mean()
    print(f"Sanity check: {len(df)} trees, mean KF = {kf:.3f} (expect about 0.333), mean MAE = {df['mae'].mean():.3f}")
    print(f"  per-tree time: inference {df['infer_seconds'].mean():.3f}s + FastME {df['fastme_seconds'].mean():.3f}s")
    print(f"  GPU peak (PyTorch): allocated {df['peak_alloc_mb'].max():.0f} MB, reserved {df['peak_reserved_mb'].max():.0f} MB; "
          f"nvidia-smi peak {df['gpu_nvsmi_peak_mb'].iloc[0]:.0f} MB")
    if failures or not (0.30 < kf < 0.36):
        raise RuntimeError(f"Sanity check FAILED (KF {kf:.3f}, failures {failures[:3]}). Do not start the study.")
    print("Sanity check passed.")


# ----------------------------------------------------------------------------------------------
# Stage 3: other methods (CPU)
# ----------------------------------------------------------------------------------------------
def fasta_to_phylip(fasta_path, phylip_path):
    seqs = read_fasta(fasta_path)
    with open(phylip_path, "w") as fh:
        fh.write(f"{len(seqs)} {len(next(iter(seqs.values())))}\n")
        for name, s in seqs.items():
            fh.write(f"{name} {s}\n")


def iqtree_method_name(model):
    """The paper's labels: LG+GC on the LG data, ModelFinder on the Cherry data."""
    return "IQTree_LG+GC" if model == "LG" else "IQTree_MF"


def baseline_methods(model):
    return ["FastME", "FastTree", iqtree_method_name(model)]


def _run_one_baseline(pf_dir, method, model, aln_path, out_tree, tmp, iqtree_timeout, mf_models):
    """Runs one method on one alignment. Returns (seconds or None, error text)."""
    b = os.path.join(pf_dir, "bin", "bin_linux")
    tid = stem(os.path.basename(aln_path))
    if method == "FastME":
        # Distances computed from the alignment with the LG model, then the same NNI+SPR tree search
        # that is used on Phyloformer's distances. Converting the file format is not timed.
        phy = os.path.join(tmp, tid + ".phy")
        fasta_to_phylip(aln_path, phy)
        cmd, so = [os.path.join(b, "fastme"), "-i", phy, "-pL", "-o", out_tree, "--nni", "--spr"], None
    elif method == "FastTree":
        cmd, so = [os.path.join(b, "FastTree"), "-lg", "-gamma", "-quiet", aln_path], out_tree
    else:  # IQ-TREE
        prefix = os.path.join(tmp, tid)
        m = "LG+GC" if model == "LG" else "MFP"
        cmd = [os.path.join(b, "iqtree_2.2.0"), "-s", aln_path, "-m", m, "-T", "1", "-keep-ident",
               "--prefix", prefix, "-quiet", "--redo"]
        if model != "LG" and mf_models:
            cmd += ["-mset", mf_models]
        so = None

    fh = open(out_tree, "w") if so else None
    t0 = time.perf_counter()
    rc, err = _run(cmd, cwd=pf_dir, timeout=(iqtree_timeout if method.startswith("IQTree") else None), stdout=fh)
    dt = time.perf_counter() - t0
    if fh:
        fh.close()
    if method.startswith("IQTree"):
        tf = prefix + ".treefile"
        if rc == 0 and os.path.exists(tf):
            shutil.copy(tf, out_tree)
    ok = rc == 0 and os.path.exists(out_tree) and os.path.getsize(out_tree) > 0
    if not ok and os.path.exists(out_tree):
        os.remove(out_tree)
    return (dt if ok else None), ("" if ok else ("timeout" if err == "timeout" else f"exit code {rc}: {err[-150:]}"))


def run_baselines_cell(pf_dir, team_repo, rate, model, size, seqlen, n_trees, work_dir,
                       iqtree_timeout=3600, mf_models=None, n_workers=1):
    """
    FastME, FastTree and IQ-TREE on the saved alignments of one grid cell. Returns (df, failures).
    Each method is timed per tree (wall-clock, one CPU thread each). n_workers > 1 runs several trees
    at once, which is faster but makes the timings less clean - keep 1 for the final runs.
    """
    aln_dir = sequences_dir(team_repo, rate, model, size, seqlen)
    true_dir = true_trees_dir(team_repo, rate, size)
    tmp = os.path.join(work_dir, "baseline_tmp")
    clean_dir(tmp)
    methods = baseline_methods(model)
    aln_files = list_files(aln_dir, ALN_EXTS)
    for m in methods:
        clean_dir(inferred_dir(team_repo, rate, model, size, seqlen, m))

    jobs = [(m, f) for f in aln_files for m in methods]

    def work(job):
        m, f = job
        out_tree = os.path.join(inferred_dir(team_repo, rate, model, size, seqlen, m), stem(f) + ".nwk")
        job_tmp = os.path.join(tmp, f"{m}_{stem(f)}")
        os.makedirs(job_tmp, exist_ok=True)
        secs, err = _run_one_baseline(pf_dir, m, model, os.path.join(aln_dir, f), out_tree, job_tmp,
                                      iqtree_timeout, mf_models)
        shutil.rmtree(job_tmp, ignore_errors=True)  # IQ-TREE / FastME side files are not needed
        return m, stem(f), secs, err

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(work, jobs))
    else:
        results = [work(j) for j in jobs]

    seconds, failures = {}, []
    for m, tid, secs, err in results:
        if secs is None:
            failures.append({"method": m, "tree_id": tid, "reason": err})
        else:
            seconds[(m, tid)] = secs

    frames = []
    for m in methods:
        df = compute_metrics(pf_dir, true_dir, inferred_dir(team_repo, rate, model, size, seqlen, m),
                             os.path.join(work_dir, f"verify_{m}"))
        if df is None:
            continue
        df["method"] = m
        df["runtime_seconds"] = df["id"].map(lambda i, m=m: seconds.get((m, i)))
        frames.append(df)
    if not frames:
        return None, failures
    df = pd.concat(frames, ignore_index=True)
    df = add_context_columns(df, true_dir, aln_dir)
    return df, failures


def sanity_check_baselines(pf_dir, work_dir):
    """
    Runs FastME, FastTree and IQ-TREE (LG+GC) once on one of the repo's test alignments (about
    half a minute in total) and prints their scores, so a broken program is found before hours of
    running. Raises if any of them fails.
    """
    td = os.path.join(pf_dir, "data", "testdata")
    one = "0_20_tips"
    tmp = os.path.join(work_dir, "sanity_baselines")
    clean_dir(tmp)
    true_dir = os.path.join(tmp, "true")
    os.makedirs(true_dir)
    shutil.copy(os.path.join(td, "trees", one + ".nwk"), true_dir)
    aln = os.path.join(td, "msas", one + ".fa")
    rows = []
    for m in ("FastME", "FastTree", "IQTree_LG+GC"):
        outd = os.path.join(tmp, m)
        os.makedirs(outd)
        job_tmp = os.path.join(tmp, "job_" + m)
        os.makedirs(job_tmp)
        secs, err = _run_one_baseline(pf_dir, m, "LG", aln, os.path.join(outd, one + ".nwk"), job_tmp, 600, None)
        if secs is None:
            raise RuntimeError(f"{m} failed on the test alignment: {err}")
        df = compute_metrics(pf_dir, true_dir, outd, os.path.join(tmp, "verify_" + m))
        rows.append((m, df["kf_score"].iloc[0], df["mae"].iloc[0], secs))
    for m, kf, mae, secs in rows:
        print(f"  {m:14s} KF={kf:.3f}  MAE={mae:.3f}  time={secs:.2f}s")
    print("Sanity check passed: all three programs run and produce scoreable trees.")


# ----------------------------------------------------------------------------------------------
# Completeness report
# ----------------------------------------------------------------------------------------------
def completeness_report(team_repo, rate, stage, models, sizes, seqlens, methods_for_model, n_trees):
    """Prints, per model, which grid cells/methods are short of n_trees rows, and how many failures are logged."""
    rdir = results_dir(team_repo)
    for model in models:
        path = os.path.join(rdir, results_filename(rate, model, stage))
        print(f"--- {os.path.basename(path)}")
        if not os.path.exists(path):
            print("    not found (not run yet)")
            continue
        d = pd.read_csv(path)
        d = d[d["clock_rate"] == rate]
        short = []
        for s in sizes:
            for l in seqlens:
                for m in methods_for_model(model):
                    n = int(((d["tree_size"] == s) & (d["seq_length"] == l) & (d["method"] == m)).sum())
                    if n < n_trees:
                        short.append((s, l, m, n))
        print(f"    {len(d)} rows; cells/methods below {n_trees} trees: {len(short)}")
        for s, l, m, n in short[:40]:
            print(f"      tree_size={s} seq_length={l} {m}: {n}/{n_trees}")
    fpath = os.path.join(rdir, f"failures_rate{rate}_{stage}.csv")
    if os.path.exists(fpath):
        f = pd.read_csv(fpath)
        print(f"--- {os.path.basename(fpath)}: {len(f)} logged failures")
        print(f.groupby(["model", "method", "reason"]).size().to_string() if len(f) else "    none")


# ----------------------------------------------------------------------------------------------
# Git push (used after every tree size, and once more at the end)
# ----------------------------------------------------------------------------------------------
def ask_github_login(team_repo):
    """Asks for the GitHub username and token ONCE per session. The token is held in memory only."""
    from getpass import getpass
    username = input("GitHub username: ").strip()
    token = getpass("GitHub Personal Access Token (hidden): ").strip()
    subprocess.run(["git", "-C", team_repo, "config", "user.name", username])
    subprocess.run(["git", "-C", team_repo, "config", "user.email", f"{username}@users.noreply.github.com"])
    return {"username": username, "token": token}


def _git(team_repo, *args, creds=None):
    p = subprocess.run(["git", "-C", team_repo, *args], capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if creds:
        out = out.replace(creds["token"], "***")
    return p.returncode, out


def push_to_github(team_repo, allowed_prefixes, message, creds, owner="ACKarl", repo="EPTI", retries=3,
                   remote_url=None):
    """
    Stages ONLY the folders in allowed_prefixes, checks nothing else is staged, commits, pulls (merge)
    and pushes. If another push lands in between, pulls again and retries.
    """
    for pre in allowed_prefixes:
        os.makedirs(os.path.join(team_repo, pre), exist_ok=True)
    rc, out = _git(team_repo, "add", "--", *allowed_prefixes)
    if rc != 0:
        raise RuntimeError("git add failed (is team_repo a git clone, and are these folders in the sparse checkout?):\n" + out)
    rc, names = _git(team_repo, "diff", "--cached", "--name-status")
    if rc != 0:
        raise RuntimeError("git diff failed:\n" + names)
    staged = [l.split("\t", 1)[-1] for l in names.strip().splitlines() if l.strip()]
    if not staged:
        print("Push: nothing new to push.")
        return True
    outside = [p for p in staged if not any(p.startswith(pre) for pre in allowed_prefixes)]
    if outside:
        _git(team_repo, "reset")
        raise RuntimeError(f"Refusing to push: files outside this study are staged, e.g. {outside[:3]}")
    print(f"Push: {len(staged)} files staged (all inside {', '.join(allowed_prefixes)})")

    rc, out = _git(team_repo, "commit", "-m", message)
    if rc != 0:
        raise RuntimeError("git commit failed:\n" + out)
    url = remote_url or f"https://{creds['username']}:{creds['token']}@github.com/{owner}/{repo}.git"
    for attempt in range(1, retries + 1):
        rc, out = _git(team_repo, "pull", "--no-rebase", "--no-edit", url, "main", creds=creds)
        if rc != 0:
            raise RuntimeError("git pull failed (merge conflict?):\n" + out)
        rc, out = _git(team_repo, "push", url, "HEAD:main", creds=creds)
        if rc == 0:
            print("Push: done.")
            return True
        print(f"Push attempt {attempt} rejected, pulling again...\n{out[-300:]}")
    raise RuntimeError("git push failed after retries:\n" + out)


def study_prefixes(rate):
    """Folders this study may write to (used by the push step)."""
    return [f"simulated_data/{STUDY}/", f"results/{STUDY}/"]
