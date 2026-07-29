import os
import time
from pathlib import Path
import numpy as np

from case_io import save_case_data_bundle


def make_run_dir(results_root="results", prefix="run"):
    os.makedirs(results_root, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"{prefix}_{run_tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_tag


def save_npz_results(path, **arrays):
    np.savez_compressed(path, **arrays)


def write_metadata(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def save_case_with_results(case_npz_path, geom, run_tag, solution_fields, histories=None, coordinates=None, extra_meta=None):
    case_npz_path = Path(case_npz_path)
    case_dir = case_npz_path.parent
    case_name = geom["case_name"]

    merged_path = case_dir / f"{case_name}__steady_{run_tag}.npz"

    data_fields = {
        "solved_u": solution_fields["u"],
        "solved_v": solution_fields["v"],
        "solved_p": solution_fields["p"],
        "solved_T": solution_fields["T"],
        "solution_state": "steady_solution",
        "source_case_file": str(case_npz_path.name),
    }

    if histories:
        data_fields.update(histories)
    if coordinates:
        data_fields.update(coordinates)

    save_case_data_bundle(
        merged_path,
        geom,
        data_fields=data_fields,
        extra_meta=extra_meta,
    )
    return merged_path
