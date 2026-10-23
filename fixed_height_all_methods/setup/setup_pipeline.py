"""
One-time setup for the three evaluation notebooks. Run it as the first thing in each notebook:

    python team_repo/fixed_height_all_methods/setup/setup_pipeline.py --stage 1|2|3 --clock-rate 0.4

Stage 1 = generate trees and sequences   (CPU runtime)
Stage 2 = run Phyloformer                (GPU runtime)
Stage 3 = run FastME / FastTree / IQ-TREE (CPU runtime)

What it does (safe to run again - every step checks before it changes anything):
  1. clones Phyloformer (the tool being evaluated)
  2. installs the few Python packages the pipeline needs
  3. makes the bundled programs executable
  4. Stage 2 only: applies the one-line device fix to Phyloformer's model.py and copies in the
     timed inference script
  5. Stage 1 only: copies the two Cherry model data files from the team repo into place
  6. pulls the part of the team repo this stage needs (only YOUR clock rate's data)
  7. checks the runtime type: Stage 2 stops if there is no GPU; Stages 1/3 warn if a GPU is attached
     (an attached GPU still uses up quota even when idle)
Details of the installs are written to setup_log.txt, not printed.
"""
import argparse
import os
import shutil
import subprocess
import sys

STUDY = "fixed_height_all_methods"
STAGE_NAMES = {1: "generate trees and sequences (CPU runtime)",
               2: "run Phyloformer (GPU runtime)",
               3: "run FastME / FastTree / IQ-TREE (CPU runtime)"}
failed = []


def log_run(cmd, logfile, cwd=None):
    with open(logfile, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode


def report(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        failed.append(name)


def gpu_attached():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--clock-rate", type=float, required=True)
    ap.add_argument("--base", default=os.getcwd(), help="folder holding Phyloformer/ and team_repo/")
    a = ap.parse_args()

    base = os.path.abspath(a.base)
    pf = os.path.join(base, "Phyloformer")
    team = os.path.join(base, "team_repo")
    log = os.path.join(base, "setup_log.txt")
    rate = a.clock_rate
    print(f"Setup for Stage {a.stage}: {STAGE_NAMES[a.stage]}, clock rate {rate}")

    # --- runtime check first, so a wrong runtime is caught before any time is spent ---
    gpu = gpu_attached()
    if a.stage == 2 and not gpu:
        report("GPU runtime", False, "Stage 2 needs a GPU. Runtime > Change runtime type > T4 GPU, then run again.")
        sys.exit(1)
    if a.stage == 2:
        report("GPU runtime", True)
    elif gpu:
        print("  [WARNING] A GPU is attached to this runtime but Stage "
              f"{a.stage} does not use it. Switch to a CPU runtime (Runtime > Change runtime type > CPU) "
              "to save your GPU quota.")
    else:
        report("CPU runtime (no GPU attached - correct for this stage)", True)

    # --- 1. Phyloformer ---
    if not os.path.isdir(pf):
        rc = log_run(["git", "clone", "-q", "https://github.com/lucanest/Phyloformer.git", pf], log)
        report("Phyloformer cloned", rc == 0 and os.path.isdir(pf), "git clone failed, see setup_log.txt")
    else:
        report("Phyloformer already cloned", True)
    if not os.path.isdir(pf):
        sys.exit(1)

    # --- 2. packages ---
    if a.stage == 3:
        # Stage 3 only runs the bundled programs and reads results with pandas (preinstalled).
        r = subprocess.run([sys.executable, "-c", "import pandas, numpy"], capture_output=True, text=True)
        report("Python packages (pandas)", r.returncode == 0, (r.stderr or "").strip().splitlines()[-1:][0:1] and
               (r.stderr or "").strip().splitlines()[-1])
    else:
        rc = log_run([sys.executable, "-m", "pip", "install", "-q", "biopython", "ete3", "dendropy", "tqdm",
                      "numpy", "scipy", "legacy-cgi"], log)
        r = subprocess.run([sys.executable, "-c", "import Bio, ete3, dendropy, numpy, scipy, tqdm, pandas"],
                           capture_output=True, text=True)
        last = (r.stderr or "").strip().splitlines()[-1:] or [""]
        report("Python packages", rc == 0 and r.returncode == 0, last[0] + " (details in setup_log.txt)")

    # --- 3. executables ---
    for d in ("bin/bin_linux", "bin"):
        folder = os.path.join(pf, d)
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p):
                os.chmod(p, os.stat(p).st_mode | 0o111)
    need = ["fastme", "FastTree", "iqtree_2.2.0", "phylocompare"]
    missing = [b for b in need if not os.access(os.path.join(pf, "bin", "bin_linux", b), os.X_OK)]
    report("bundled programs (FastME, FastTree, IQ-TREE, phylocompare)", not missing, f"missing: {missing}")

    # --- 4. Stage 2: device fix + timed inference script ---
    if a.stage == 2:
        model_py = os.path.join(pf, "phyloformer", "model.py")
        old, new = "device = self.seq2pair.device", "device = next(self.parameters()).device"
        text = open(model_py).read()
        if old in text:
            open(model_py, "w").write(text.replace(old, new))
        text = open(model_py).read()
        report("Phyloformer device fix (model.py)", new in text and old not in text)
        src = os.path.join(team, STUDY, "setup", "pf_infer_timed.py")
        if not os.path.exists(src):
            report("timed inference script", False, f"{src} not found - is setup/ pushed to the team repo?")
        else:
            shutil.copy(src, os.path.join(pf, "pf_infer_timed.py"))
            rc3 = subprocess.run([sys.executable, "-c", "import torch; "
                                  "from phyloformer.model import Phyloformer; import infer_alns"],
                                 cwd=pf, capture_output=True, text=True)
            report("Phyloformer imports (torch, model)", rc3.returncode == 0, (rc3.stderr or "")[-300:])
            report("PyTorch sees the GPU", subprocess.run(
                [sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"]
            ).returncode == 0, "torch.cuda.is_available() is False")

    # --- 6. team repo: pull only what this stage needs ---
    d = f"simulated_data/{STUDY}"
    wanted = [f"{STUDY}/setup", f"results/{STUDY}",
              f"{d}/trees/true/clock_rate={rate}",
              f"{d}/sequences/clock_rate={rate}",
              f"{d}/trees/inferred/clock_rate={rate}"]
    if a.stage == 1:
        wanted.append("CherryML_data(CoreCode)")
    rc = log_run(["git", "-C", team, "sparse-checkout", "set", "--cone", *wanted], log)
    rc_pull = log_run(["git", "-C", team, "pull", "-q", "--no-rebase", "--no-edit", "origin", "main"], log)
    report("team repo (only the folders this stage needs)", rc == 0 and rc_pull == 0,
           "see setup_log.txt (is team_repo cloned? run the first cell of the notebook)")

    # --- 5. Stage 1: Cherry data files ---
    if a.stage == 1:
        src_dir = os.path.join(team, "CherryML_data(CoreCode)")
        dest = os.path.join(pf, "bin", "simulateWithCoevolution", "data")
        os.makedirs(dest, exist_ok=True)
        files = ["coevolution.txt", "coevolution_stationary.txt"]
        miss = [f for f in files if not os.path.exists(os.path.join(src_dir, f))]
        if miss:
            report("Cherry model data files", False, f"not found in team repo: {miss}")
        else:
            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dest, f))
            report("Cherry model data files copied", True)

    print("SETUP FAILED - fix the [FAIL] items above, then run this cell again." if failed
          else "SETUP OK - continue with the next cell.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
