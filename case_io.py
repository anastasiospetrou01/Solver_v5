import os
from pathlib import Path
import numpy as np


DEFAULT_SOURCES = {
    "energy": {"default": 0.0, "by_region": {}},
    "momentum_x": {"default": 0.0, "by_region": {}},
    "momentum_y": {"default": 0.0, "by_region": {}},
}


def _case_payload(geom):
    return {
        "nx": geom["nx"],
        "ny": geom["ny"],
        "Lx": geom["Lx"],
        "Ly": geom["Ly"],
        "region": geom["region"],
        "region_defs": geom["region_defs"],
        "case_name": geom["case_name"],
        "boundaries": geom["boundaries"],
        "initial_temperature": geom["initial_temperature"],
        "initial_flow": geom["initial_flow"],
        "sources": geom.get("sources", DEFAULT_SOURCES),
    }


def _format_mapping(title, mapping, lines, indent=0):
    prefix = " " * indent
    lines.append(f"{prefix}{title}:")
    if not mapping:
        lines.append(f"{prefix}  <empty>")
        return
    for key, value in mapping.items():
        if isinstance(value, dict):
            _format_mapping(str(key), value, lines, indent=indent + 2)
        else:
            lines.append(f"{prefix}  {key}: {value}")


def build_case_info_lines(geom):
    lines = []
    lines.append("================ CASE INFORMATION ================")
    lines.append(f"case_name: {geom['case_name']}")
    lines.append(f"nx: {geom['nx']}")
    lines.append(f"ny: {geom['ny']}")
    lines.append(f"Lx: {geom['Lx']}")
    lines.append(f"Ly: {geom['Ly']}")
    lines.append(f"dx: {geom['Lx'] / geom['nx']:.8e}")
    lines.append(f"dy: {geom['Ly'] / geom['ny']:.8e}")
    lines.append("")

    lines.append("region_defs:")
    for region_id, info in geom["region_defs"].items():
        lines.append(
            f"  id={region_id} | name={info.get('name')} | material={info.get('material')} | phase={info.get('phase')}"
        )
    lines.append("")

    _format_mapping("boundaries", geom["boundaries"], lines)
    lines.append("")
    _format_mapping("initial_temperature", geom["initial_temperature"], lines)
    lines.append("")
    _format_mapping("initial_flow", geom["initial_flow"], lines)
    lines.append("")
    _format_mapping("sources", geom.get("sources", DEFAULT_SOURCES), lines)
    lines.append("==================================================")
    return lines


def write_case_info(path, geom):
    lines = build_case_info_lines(geom)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def prepare_case_directory(cases_root, case_name):
    case_dir = Path(cases_root) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def save_case(filename, geom):
    np.savez_compressed(filename, **_case_payload(geom))


def save_case_bundle(cases_root, geom):
    case_dir = prepare_case_directory(cases_root, geom["case_name"])
    case_npz_path = case_dir / f"{geom['case_name']}.npz"
    case_info_path = case_dir / "case_info.txt"
    save_case(case_npz_path, geom)
    write_case_info(case_info_path, geom)
    return {
        "case_dir": case_dir,
        "case_npz_path": case_npz_path,
        "case_info_path": case_info_path,
    }


def save_case_data_bundle(path, geom, data_fields=None, extra_meta=None):
    payload = _case_payload(geom)

    if data_fields:
        payload.update(data_fields)

    if extra_meta:
        payload.update(extra_meta)

    np.savez_compressed(path, **payload)


def load_case(filename):
    data = np.load(filename, allow_pickle=True)

    geom = {
        "nx": int(data["nx"]),
        "ny": int(data["ny"]),
        "Lx": float(data["Lx"]),
        "Ly": float(data["Ly"]),
        "region": data["region"],
        "region_defs": data["region_defs"].item(),
        "case_name": str(data["case_name"]),
        "boundaries": data["boundaries"].item(),
        "initial_temperature": data["initial_temperature"].item(),
        "initial_flow": data["initial_flow"].item(),
        "sources": (
            data["sources"].item()
            if "sources" in data.files
            else DEFAULT_SOURCES
        ),
    }

    if "solved_u" in data.files and "solved_v" in data.files and "solved_p" in data.files:
        geom["solved_flow"] = {
            "u": data["solved_u"],
            "v": data["solved_v"],
            "p": data["solved_p"],
        }

    if "solved_T" in data.files:
        geom["solved_temperature"] = data["solved_T"]

    if "solution_state" in data.files:
        value = data["solution_state"]
        geom["solution_state"] = value.item() if hasattr(value, "item") else value

    if "source_case_file" in data.files:
        geom["source_case_file"] = str(data["source_case_file"])

    return geom
