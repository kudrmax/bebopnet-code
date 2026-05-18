"""Build training/colab_train.ipynb from inline source.

Run once after editing the cell source below:
    python training/_build_notebook.py
The notebook is committed to the repo for direct opening in Colab.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).parent / "colab_train.ipynb"

CELL_SOURCE = r'''# === Configuration ===
MAX_STEP = 500_000        # paper recipe (configs/train_model.yml)
EVAL_INTERVAL = 4000      # also paper recipe; controls best-checkpoint cadence
SMOKE_MAX_STEP = None     # set to ~2000 for quick pipeline check; None = full recipe

# === Paths ===
REPO_ROOT = "/content/repo"
GIT_URL = "https://github.com/kudrmax/bebopnet-code.git"
GIT_BRANCH = "master"   # NOTE: switch to feat/colab-training while iterating

DIPLOMA_REPO_ROOT = "/content/diploma2"
DIPLOMA_GIT_URL = "https://github.com/kudrmax/jazz-generation-research.git"
DIPLOMA_BRANCH = "master"

DRIVE_ROOT = "/content/drive/MyDrive/bebopnet-training"
RUN_NAME = "result-paper-default"  # change for ablation runs (e.g. "result-paper-pretrained")
RESULT_ROOT = f"{DRIVE_ROOT}/{RUN_NAME}"  # work_dir for train.py
PREP_ROOT = "/content/prep"               # transient: routed xml + pkls (not on Drive)
PKL_DIR = f"{PREP_ROOT}/pkls"

import threading, time, os, re, subprocess
from datetime import datetime


def run(cmd, **kw):
    """Popen + line-by-line stdout proxy (Colab IOPub buffering workaround)."""
    print(f">>> {cmd}", flush=True)
    env = kw.pop("env", None) or os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, shell=isinstance(cmd, str), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True, **kw,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def cache_restore(name, dest_path):
    """Restore <DRIVE_ROOT>/cache/<name>.tar.gz to dest_path. Returns True if restored."""
    tar_path = f"{DRIVE_ROOT}/cache/{name}.tar.gz"
    if os.path.exists(dest_path):
        return False
    if not os.path.exists(tar_path):
        return False
    print(f"==> Restoring {name} from Drive cache...", flush=True)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    run(["tar", "-xzf", tar_path, "-C", os.path.dirname(dest_path)])
    return True


def cache_save(name, src_path):
    """Cache src_path to <DRIVE_ROOT>/cache/<name>.tar.gz. No-op if missing or already cached."""
    tar_path = f"{DRIVE_ROOT}/cache/{name}.tar.gz"
    if not os.path.exists(src_path):
        return False
    if os.path.exists(tar_path):
        return False
    print(f"==> Caching {name} to Drive...", flush=True)
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    run(["tar", "-czf", tar_path, "-C", os.path.dirname(src_path), os.path.basename(src_path)])
    return True


# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')
os.makedirs(DRIVE_ROOT, exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/cache", exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

# 2. Fresh clone bebopnet fork
if os.path.isdir(REPO_ROOT):
    backup = f"{REPO_ROOT}.old.{int(time.time())}"
    os.rename(REPO_ROOT, backup)
    print(f"==> Moved old repo to {backup}", flush=True)
run(["git", "clone", "-b", GIT_BRANCH, GIT_URL, REPO_ROOT])

# 3. Clone diploma2 main repo for split.json
if os.path.isdir(DIPLOMA_REPO_ROOT):
    os.rename(DIPLOMA_REPO_ROOT, f"{DIPLOMA_REPO_ROOT}.old.{int(time.time())}")
run(["git", "clone", "--depth", "1", "-b", DIPLOMA_BRANCH, DIPLOMA_GIT_URL, DIPLOMA_REPO_ROOT])
os.chdir(REPO_ROOT)

# 4. Install deps (training stack on top of inference reqs)
run(["pip", "install", "-q", "-r", "requirements-py312.txt"])
run(["pip", "install", "-q", "tensorboard"])

# 5. Sanity: split.json + 422 xml committed
SPLIT_JSON = f"{DIPLOMA_REPO_ROOT}/pipelines/training-pipeline/wjazzd_split.json"
assert os.path.exists(SPLIT_JSON), f"split.json missing at {SPLIT_JSON}"

xml_pool = os.path.join(REPO_ROOT, "resources/dataset/wjazzd")
n_xml = sum(
    1 for f in os.listdir(xml_pool)
    if f.endswith(".xml") and len(f) >= 4 and f[:3].isdigit() and f[3] == "_"
) if os.path.isdir(xml_pool) else 0
assert n_xml >= 420, (
    f"resources/dataset/wjazzd/ missing or incomplete ({n_xml} files); "
    "ensure 422 xml are committed in the bebopnet fork."
)
print(f"==> Input xml OK: {n_xml} files in {xml_pool}", flush=True)

# 6. Routing always — fast (copy 422 xml from REPO_ROOT to PREP_ROOT) and
#    needed even after pkl cache restore so test_canonical/wjazzd/ exists
#    for evaluate_canonical (step 9). Idempotent: re-copying overwrites.
print("==> Routing xml via wjazzd_split_prep...", flush=True)
os.makedirs(PREP_ROOT, exist_ok=True)
run(["python", "-m", "jazz_rnn.A_data_prep.wjazzd_split_prep",
     "--xml-source", xml_pool,
     "--split-json", SPLIT_JSON,
     "--out-root", PREP_ROOT])

# 7. Restore-or-prep pkls (the heavy part; cached on Drive after first run)
cache_restore("bebopnet_data", PKL_DIR)
if not (os.path.exists(f"{PKL_DIR}/train.pkl") and os.path.exists(f"{PKL_DIR}/val.pkl")):
    # num_processes=1 forces our sequential path (gather_data_from_xml.py
    # else-branch). mp.Pool with N>1 deadlocks on Python 3.12 + music21 9.x:
    # parent imports music21 → it spawns background threads → mp.Pool fork
    # inherits dead lock state → child hangs in extract_vectors at 0% CPU.
    # Authors used Python 3.7 + music21 5.x where this didn't happen.
    # Sequential takes ~5-7 minutes on 380+40 files; pkls cache to Drive
    # after, so subsequent Colab runs skip this step entirely.
    print("==> Running gather_data_from_xml (sequential)...", flush=True)
    run(["python", "-m", "jazz_rnn.A_data_prep.gather_data_from_xml",
         "--xml_dir", PREP_ROOT,
         "--out_dir", PKL_DIR,
         "--num_processes", "1"])
    assert os.path.exists(f"{PKL_DIR}/train.pkl") and os.path.exists(f"{PKL_DIR}/val.pkl")
cache_save("bebopnet_data", PKL_DIR)

# 7. ETA monitor — parses author's per-log-interval line in <work_dir>/log.txt
def monitor_loop():
    log_re = re.compile(r"\| epoch\s+\d+\s+step\s+(\d+)\s+\|\s+\d+\s+batches\s+\|\s+lr\s+[\d\.eE+\-]+\s+\|\s+ms/batch\s+([\d\.]+)")
    last_reported = -1
    while True:
        time.sleep(60)
        try:
            log_path = f"{RESULT_ROOT}/log.txt"
            if not os.path.exists(log_path):
                continue
            text = open(log_path).read()
            matches = log_re.findall(text)
            if not matches:
                continue
            last_step = int(matches[-1][0])
            if last_step <= last_reported:
                continue
            last_reported = last_step
            ms_per_batch = float(matches[-1][1])
            target = SMOKE_MAX_STEP or MAX_STEP
            steps_left = max(0, target - last_step)
            eta_min = steps_left * ms_per_batch / 1000 / 60
            print(f"[ETA] step {last_step}/{target}, ms/batch {ms_per_batch:.1f}, ETA {eta_min:.1f} min", flush=True)
        except Exception:
            pass

threading.Thread(target=monitor_loop, daemon=True).start()

# 8. Train (resumes from RESULT_ROOT if state present)
target_max_step = SMOKE_MAX_STEP or MAX_STEP
state_file = os.path.join(RESULT_ROOT, "train_state.json")
restart_args = ["--restart", "--restart_dir", RESULT_ROOT] if os.path.exists(state_file) else []
if restart_args:
    print(f"==> Resuming from {state_file}", flush=True)

train_cmd = ["python", "-m", "jazz_rnn.B_next_note_prediction.transformer.train",
             "--config", "configs/train_model.yml",
             "--data_pkl", PKL_DIR,
             "--work_dir", RESULT_ROOT,
             "--no_timestamp",
             "--save_name", "",
             "--max_step", str(target_max_step),
             "--eval_interval", str(EVAL_INTERVAL)] + restart_args
run(train_cmd)

# 9. Evaluate model_best.pt on canonical test=40 (held out from training
#    AND from best-checkpoint selection — see evaluate_canonical.py docstring).
test_canonical_xml = f"{PREP_ROOT}/test_canonical/wjazzd"
assert os.path.isdir(test_canonical_xml), (
    f"test_canonical xml dir missing at {test_canonical_xml}; "
    "rerun cell — wjazzd_split_prep should have created it"
)
run(["python", "-m", "jazz_rnn.B_next_note_prediction.transformer.evaluate_canonical",
     "--work-dir", RESULT_ROOT,
     "--test-canonical-xml-dir", test_canonical_xml,
     "--num-processes", "1"])

# 10. Confirm final artefacts
run(["ls", "-la", RESULT_ROOT])
assert os.path.exists(f"{RESULT_ROOT}/model.pt")
assert os.path.exists(f"{RESULT_ROOT}/model_best.pt")
assert os.path.exists(f"{RESULT_ROOT}/summary.json")
assert os.path.exists(f"{RESULT_ROOT}/epochs.csv")
assert os.path.exists(f"{RESULT_ROOT}/log.txt")
import json as _json
with open(f"{RESULT_ROOT}/summary.json") as _f:
    _s = _json.load(_f)
assert "final_test_ppl" in _s, "evaluate_canonical did not append final_test_*"
print(f"==> Done. final_test_ppl={_s['final_test_ppl']:.3f}, "
      f"best_val_ppl={_s.get('best_val_ppl'):.3f}", flush=True)
'''


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell(CELL_SOURCE)]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["accelerator"] = "GPU"
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
