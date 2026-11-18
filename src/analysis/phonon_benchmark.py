"""Metrics for comparing an MLIP's phonons against a DFPT reference, for
scripts/mlip/compare_phonons.py.

Dispersion metrics compare frequencies branch-by-branch at identical q-points
(the MLIP force constants are re-evaluated at QE's own matdyn.x q-list), with
both sets sorted ascending at each q. Sorting means band crossings are
compared by energy ordering rather than by mode character -- the standard
choice for frequency-error benchmarks, and it never inflates the error (the
sorted pairing minimizes the per-q L1 distance).

vDOS metrics put both spectra on one grid, normalize them to unit area, and
apply the same Gaussian broadening (vdos_features.smooth_dos), so what's
compared is spectral shape, independent of each code's bin width or
tetrahedron-vs-histogram noise. Integrated harmonic thermodynamics (Cv, S, F,
ZPE) are computed from each DOS rescaled to 3*nat states, since those
quantities weight the spectrum in physically meaningful ways (Cv and S at
300 K are dominated by the low-frequency modes, ZPE by the high ones).
Imaginary (negative) frequencies are excluded from the thermodynamics: a
harmonic free energy isn't defined for them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from src.mlip import mlip_phonons
from src.qe.phonon_dos import ensure_qe_dos, parse_matdyn_dos
from src.qe.qe_phonon_io import parse_fc_header, parse_matdyn_band_path, parse_matdyn_freq, path_distances
from src.qe.unstable_modes import label_q
from src.qe.vdos_features import extract_features, smooth_dos

# 1 cm-1 in kelvin (h c / k_B).
CM1_TO_K = 1.438776877
# 1 cm-1 in meV.
CM1_TO_MEV = 0.1239841984

# A branch counts as unstable only below this: DFPT/finite-difference noise
# and imperfect ASR routinely leave acoustic branches near Gamma a few cm-1
# negative, which isn't a physical instability.
UNSTABLE_TOL_CM1 = -5.0


def band_metrics(ref_cm1: np.ndarray, mlip_cm1: np.ndarray, unstable_tol_cm1: float = UNSTABLE_TOL_CM1) -> dict:
    """Errors between two (nq, nbands) frequency arrays on the same q-points."""
    if ref_cm1.shape != mlip_cm1.shape:
        raise ValueError(f"band arrays differ in shape: {ref_cm1.shape} vs {mlip_cm1.shape}")
    err = np.sort(mlip_cm1, axis=1) - np.sort(ref_cm1, axis=1)
    ref_unstable = bool(ref_cm1.min() < unstable_tol_cm1)
    mlip_unstable = bool(mlip_cm1.min() < unstable_tol_cm1)
    return {
        "band_mae_cm1": float(np.abs(err).mean()),
        "band_rmse_cm1": float(np.sqrt((err**2).mean())),
        "band_max_abs_err_cm1": float(np.abs(err).max()),
        # Negative = the MLIP is systematically too soft (the usual MLIP failure mode).
        "band_mean_signed_err_cm1": float(err.mean()),
        "omega_max_ref_cm1": float(ref_cm1.max()),
        "omega_max_mlip_cm1": float(mlip_cm1.max()),
        "omega_min_ref_cm1": float(ref_cm1.min()),
        "omega_min_mlip_cm1": float(mlip_cm1.min()),
        "ref_unstable": ref_unstable,
        "mlip_unstable": mlip_unstable,
        "stability_agrees": ref_unstable == mlip_unstable,
    }


def _common_grid(freq_a: np.ndarray, freq_b: np.ndarray) -> np.ndarray:
    step = min(np.diff(freq_a).min(), np.diff(freq_b).min())
    lo, hi = min(freq_a.min(), freq_b.min()), max(freq_a.max(), freq_b.max())
    return np.arange(lo, hi + step, step)


def _to_unit_area(grid: np.ndarray, freq: np.ndarray, dos: np.ndarray, sigma_cm1: float) -> np.ndarray:
    g = np.interp(grid, freq, np.clip(dos, 0.0, None), left=0.0, right=0.0)
    g = np.clip(smooth_dos(grid, g, sigma_cm1=sigma_cm1), 0.0, None)
    return g / np.trapezoid(g, grid)


def harmonic_thermo(freq_cm1: np.ndarray, dos: np.ndarray, n_atoms: int, temperature_k: float) -> dict:
    """Per-atom harmonic Cv (k_B), S (k_B), F (meV) and ZPE (meV) from a DOS
    rescaled to 3*n_atoms states over its positive-frequency part."""
    mask = freq_cm1 > 1e-3
    w, g = freq_cm1[mask], np.clip(dos[mask], 0.0, None)
    g = g * (3 * n_atoms) / np.trapezoid(g, w)
    x = w * CM1_TO_K / temperature_k
    with np.errstate(over="ignore"):
        ex = np.exp(x)
        bose = 1.0 / np.expm1(x)
    cv = np.trapezoid(g * x**2 * ex / np.expm1(x) ** 2, w)
    s = np.trapezoid(g * (x * bose - np.log1p(-np.exp(-x))), w)
    zpe = np.trapezoid(g * 0.5 * w * CM1_TO_MEV, w)
    f = zpe + np.trapezoid(g * temperature_k / CM1_TO_K * CM1_TO_MEV * np.log1p(-np.exp(-x)), w)
    return {
        "cv_kB": cv / n_atoms,
        "s_kB": s / n_atoms,
        "f_meV": f / n_atoms,
        "zpe_meV": zpe / n_atoms,
    }


def dos_metrics(
    ref_freq: np.ndarray,
    ref_dos: np.ndarray,
    mlip_freq: np.ndarray,
    mlip_dos: np.ndarray,
    n_atoms: int,
    sigma_cm1: float = 2.0,
    temperature_k: float = 300.0,
) -> dict:
    """Shape + thermodynamic agreement between two vDOS (any normalization)."""
    grid = _common_grid(ref_freq, mlip_freq)
    p = _to_unit_area(grid, ref_freq, ref_dos, sigma_cm1)
    q = _to_unit_area(grid, mlip_freq, mlip_dos, sigma_cm1)

    out = {
        # Earth-mover's distance: how far (cm-1), on average, spectral weight
        # has to move to turn one DOS into the other. Directly interpretable.
        "dos_wasserstein_cm1": float(wasserstein_distance(grid, grid, p, q)),
        # Shared area of the two unit-area curves: 1 = identical, 0 = disjoint.
        "dos_overlap": float(np.trapezoid(np.minimum(p, q), grid)),
        "dos_cosine": float(p @ q / np.sqrt((p @ p) * (q @ q))),
        "dos_mean_freq_ref_cm1": float(np.trapezoid(grid * p, grid)),
        "dos_mean_freq_mlip_cm1": float(np.trapezoid(grid * q, grid)),
    }

    ref_thermo = harmonic_thermo(ref_freq, ref_dos, n_atoms, temperature_k)
    mlip_thermo = harmonic_thermo(mlip_freq, mlip_dos, n_atoms, temperature_k)
    for key in ref_thermo:
        out[f"{key}_ref"] = float(ref_thermo[key])
        out[f"{key}_mlip"] = float(mlip_thermo[key])
        out[f"{key}_err"] = float(mlip_thermo[key] - ref_thermo[key])

    # Largest phonon gap, on the 3*nat-normalized DOS so dos_tol means the
    # same thing for both (vdos_features.largest_gap).
    for tag, freq, dos in (("ref", ref_freq, ref_dos), ("mlip", mlip_freq, mlip_dos)):
        scaled = np.clip(dos, 0.0, None) * (3 * n_atoms) / np.trapezoid(np.clip(dos, 0.0, None), freq)
        features = extract_features(freq, scaled)
        out[f"gap_width_{tag}_cm1"] = features.gap_width_cm1
        out[f"imaginary_fraction_{tag}"] = features.imaginary_fraction
    return out


def load_qe_reference(
    ph_dir: Path, name: str, matdyn_x: Path | None = None,
    nk: tuple[int, int, int] = (20, 20, 20), delta_e_cm1: float = 0.5,
) -> dict | None:
    """One DFPT run's dispersion (<name>.freq on its own band path, with tick
    labels from matdyn.in), geometry/Born charges (<name>.fc) and vDOS
    (<name>.dos, generated with matdyn_x if missing and one is given; 'dos' is
    None otherwise). None if the .fc or .freq file is missing."""
    fc_path, freq_path = ph_dir / f"{name}.fc", ph_dir / f"{name}.freq"
    if not (fc_path.exists() and freq_path.exists()):
        return None
    header = parse_fc_header(fc_path)
    q_cart, freqs = parse_matdyn_freq(freq_path)
    qe = {
        "header": header,
        "q_cart": q_cart,
        "q_frac": header.q_cart_to_frac(q_cart),
        "x": path_distances(q_cart),
        "freqs": freqs,
        "ticks": [],
        "dos": None,
    }

    matdyn_in = ph_dir / "matdyn.in"
    if matdyn_in.exists():
        index = 0
        for vertex, npts in parse_matdyn_band_path(matdyn_in):
            if index >= len(q_cart):
                break
            label = label_q(vertex)
            qe["ticks"].append((qe["x"][index], r"$\Gamma$" if label == "Gamma" else label))
            index += npts

    dos_path = ensure_qe_dos(ph_dir, name, matdyn_x, tuple(nk), delta_e_cm1)
    if dos_path is not None:
        qe["dos"] = parse_matdyn_dos(dos_path)
    return qe


def load_mlip_on_reference_path(model_dir: Path, name: str, qe: dict) -> dict | None:
    """One MLIP run (phonon_pipeline output) evaluated at the q-points of
    `qe` (a load_qe_reference result), with its vDOS and meta.json. None if
    the run has no force constants."""
    run_dir = model_dir / name
    params = run_dir / "phonopy_params.yaml"
    if not params.exists():
        return None
    phonon = mlip_phonons.load(params)
    if len(phonon.unitcell) != len(qe["header"].symbols):
        raise ValueError(
            f"{run_dir}: {len(phonon.unitcell)} atoms in the MLIP cell vs "
            f"{len(qe['header'].symbols)} in QE's -- not the same structure"
        )
    dos_path = run_dir / f"{name}.dos"
    meta_path = run_dir / "meta.json"
    return {
        "freqs": mlip_phonons.frequencies_cm1(phonon, qe["q_frac"]),
        "dos": parse_matdyn_dos(dos_path) if dos_path.exists() else None,
        "meta": json.loads(meta_path.read_text()) if meta_path.exists() else {},
    }
