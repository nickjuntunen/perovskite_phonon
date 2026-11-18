"""Read back what src/mlip/phonon_pipeline.py writes: <root>/<model label>/
<name>/ with meta.json, band.npz, <name>.dos, phonopy_params.yaml, and the
pre_rattle/ and soft_mode/ subdirectories. The on-disk schema is defined by
the pipeline; everything that depends on it is here, next to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def is_finished(run_dir: Path) -> bool:
    """Whether the pipeline completed for this run directory: meta.json is
    the last file it writes (what run_one's own skip check tests)."""
    return (Path(run_dir) / "meta.json").exists()


def unfinished(names: list[str], sweep_root: Path) -> list[str]:
    """The `names` without a finished run under one model's output directory.
    NotADirectoryError if that directory isn't there (an unmounted sweep would
    otherwise read as "nothing finished")."""
    sweep_root = Path(sweep_root)
    if not sweep_root.is_dir():
        raise NotADirectoryError(f"{sweep_root} is not a directory (is the sweep mounted?)")
    return [n for n in names if not is_finished(sweep_root / n)]


def discover_models(root: Path) -> list[str]:
    """Model labels under an output root: subdirectories with at least one run's force constants."""
    return sorted(d.name for d in Path(root).iterdir() if any(d.glob("*/phonopy_params.yaml")))


def load_band_run(run_dir: Path, name: str) -> dict | None:
    """band.npz (and the vDOS, if written) of one run directory, or None if
    there is no band.npz."""
    band_path = run_dir / "band.npz"
    if not band_path.exists():
        return None
    from src.qe.phonon_dos import parse_matdyn_dos

    band = np.load(band_path)
    dos_path = run_dir / f"{name}.dos"
    return {
        "q_frac": band["q_frac"],
        "x": band["x"],
        "special_x": band["special_x"],
        "labels": [str(label) for label in band["labels"]],
        "freqs": band["freqs_cm1"],
        "dos": parse_matdyn_dos(dos_path) if dos_path.exists() else None,
    }


def path_min_q(band_npz: Path) -> list[float] | None:
    """Fractional q (reference cell) of the lowest frequency on the band path."""
    if not band_npz.exists():
        return None
    band = np.load(band_npz)
    i = np.unravel_index(np.argmin(band["freqs_cm1"]), band["freqs_cm1"].shape)[0]
    return [round(float(x), 4) for x in band["q_frac"][i]]


def run_states(run_dir: Path) -> list[dict]:
    """The candidate states of one run, with the meta.json fields they share."""
    meta = json.loads((run_dir / "meta.json").read_text())
    mp_id, formula = run_dir.name.split("_", 1)
    rattle, soft = meta.get("rattle_retry"), meta.get("soft_mode")
    retried_ok = rattle is not None and "error" not in rattle
    first = rattle["first_attempt"] if rattle is not None else meta

    def attempt(state: str, subdir: str, m: dict, energy: float, **extra) -> dict:
        return {
            "state": state, "dir": run_dir / subdir, "energy_eV_per_atom": energy,
            "omega_min_cm1": m["min_freq_cm1"], "omega_min_path_cm1": m["min_freq_on_path_cm1"],
            "relax": m.get("relax", "full"), "keep_symmetry": m.get("keep_symmetry"),
            "max_residual_force_eV_per_A": m["max_residual_force_eV_per_A"],
            "pressure_GPa": m.get("pressure_GPa"), "phonon_supercell": "x".join(map(str, m["supercell"])),
            "n_displaced_supercells": m["n_displaced_supercells"], "nac": m["nac"], **extra,
        }

    states = [attempt("symmetric", "pre_rattle" if rattle is not None else ".", first, first["energy_eV_per_atom"])]
    if retried_ok:
        states.append(attempt("rattled", ".", meta, meta["energy_eV_per_atom"]))
    if soft is not None and "error" not in soft:
        states.append(attempt(
            "soft_mode", "soft_mode", soft,
            # energy_gain is vs the symmetric first attempt, per atom
            first["energy_eV_per_atom"] + soft["energy_gain_meV_per_atom"] / 1000,
            soft_mode_seeds="+".join(f"{s['q']}({s['frequency_cm1']:.1f})" for s in soft["seeds"]),
            soft_mode_spacegroup_run=soft["spacegroup"],
        ))

    for s in states:
        s.update(material_id=mp_id, run_name=run_dir.name, formula_mp=formula, model_label=meta["model_label"],
                 rattle_error=rattle.get("error") if rattle else None,
                 soft_mode_error=soft.get("error") if soft else None)
    return states
