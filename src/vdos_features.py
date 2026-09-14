"""Extract scalar features from a Quantum ESPRESSO matdyn.x phonon DOS
(vDOS) for use as feature-based fitness targets -- a phonon gap, or a
specific low-frequency mode -- and as a synthesizability flag for
scripts/cluster_compositions.py.

matdyn.x's dos=.true. output is an unsmoothed per-bin histogram (see
phonon_dos.py's module docstring) -- for anything besides gap detection or
an exact peak-bin lookup, smooth_dos gives it a Gaussian broadening first so
a target-frequency search isn't defeated by individual empty/spiky bins.

On imaginary frequencies: the interpolated matdyn.x mesh, not just the
coarse DFPT q-grid, can show negative frequencies -- e.g. a residual
instability at a generic q-point that wasn't one of the DFPT run's sampled
high-symmetry points. This module reports that (has_imaginary,
imaginary_fraction) as *information*, not a reason to discard the structure
outright: as discussed for the tilting project, imaginary frequencies in the
cubic aristotype are expected and don't by themselves mean "not
synthesizable" (see scripts/prepare_dos_inputs.py, which is what decides
whether to score the cubic cell or a relaxed tilted structure in the first
place). A structure that's still imaginary *after* soft-mode relaxation is a
much stronger synthesizability warning than one that was imaginary only in
the unrelaxed cubic cell.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

# Matches unstable_modes.find_unstable_modes' default tolerance, so a
# numerically-zero acoustic branch near Gamma isn't flagged as imaginary.
IMAGINARY_FREQ_TOL_CM1 = -1e-3


@dataclass
class VDOSFeatures:
    has_imaginary: bool
    imaginary_fraction: float  # fraction of total DOS mass sitting below IMAGINARY_FREQ_TOL_CM1
    gap_low_cm1: float | None
    gap_high_cm1: float | None
    gap_width_cm1: float  # 0.0 if no gap found
    peak_freqs_cm1: np.ndarray
    peak_heights: np.ndarray


def smooth_dos(freq_cm1: np.ndarray, dos: np.ndarray, sigma_cm1: float = 2.0) -> np.ndarray:
    """Gaussian-broaden a raw matdyn.x DOS histogram (see module docstring).

    Assumes freq_cm1 is uniformly spaced, which matdyn.x's deltaE binning
    guarantees.
    """
    bin_width = float(freq_cm1[1] - freq_cm1[0])
    sigma_bins = sigma_cm1 / bin_width
    half_width = max(1, int(np.ceil(4 * sigma_bins)))
    x = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(dos, kernel, mode="same")


def largest_gap(
    freq_cm1: np.ndarray, dos: np.ndarray, dos_tol: float = 1e-3, search_above_cm1: float = 0.0
) -> tuple[float | None, float | None, float]:
    """Largest contiguous frequency window, above `search_above_cm1` (default:
    only the real/positive branch -- see has_imaginary), where DOS stays
    below `dos_tol`.

    Returns (low, high, width); (None, None, 0.0) if no such window exists
    (e.g. a genuinely gapless spectrum, or too coarse an `nk` mesh in
    phonon_dos.generate_dos_inputs -- widen nk or loosen dos_tol before
    trusting a "no gap" result on a borderline case).
    """
    mask = freq_cm1 >= search_above_cm1
    f, d = freq_cm1[mask], dos[mask]
    below = d < dos_tol

    best_low, best_high, best_width = None, None, 0.0
    i = 0
    while i < len(f):
        if below[i]:
            j = i
            while j < len(f) and below[j]:
                j += 1
            width = float(f[j - 1] - f[i])
            if width > best_width:
                best_low, best_high, best_width = float(f[i]), float(f[j - 1]), width
            i = j
        else:
            i += 1
    return best_low, best_high, best_width


def extract_features(
    freq_cm1: np.ndarray,
    dos: np.ndarray,
    dos_tol: float = 1e-3,
    peak_prominence: float = 1e-3,
    smoothing_sigma_cm1: float = 2.0,
) -> VDOSFeatures:
    """Build a VDOSFeatures from one structure's parsed matdyn.x DOS
    (phonon_dos.parse_matdyn_dos's output)."""
    imaginary_mask = freq_cm1 < IMAGINARY_FREQ_TOL_CM1
    has_imaginary = bool(imaginary_mask.any())
    total = float(dos.sum())
    imaginary_fraction = float(dos[imaginary_mask].sum() / total) if total > 0 else 0.0

    gap_low, gap_high, gap_width = largest_gap(freq_cm1, dos, dos_tol=dos_tol, search_above_cm1=0.0)

    smoothed = smooth_dos(freq_cm1, dos, sigma_cm1=smoothing_sigma_cm1)
    peak_idx, _ = find_peaks(smoothed, prominence=peak_prominence)

    return VDOSFeatures(
        has_imaginary=has_imaginary,
        imaginary_fraction=imaginary_fraction,
        gap_low_cm1=gap_low,
        gap_high_cm1=gap_high,
        gap_width_cm1=gap_width,
        peak_freqs_cm1=freq_cm1[peak_idx],
        peak_heights=smoothed[peak_idx],
    )


def nearest_peak(features: VDOSFeatures, target_freq_cm1: float) -> tuple[float, float, float] | None:
    """(freq, height, |freq - target|) of whichever vDOS peak is closest to
    target_freq_cm1. None if extract_features found no peaks at all (e.g.
    peak_prominence set too strict for this structure's DOS)."""
    if len(features.peak_freqs_cm1) == 0:
        return None
    idx = int(np.argmin(np.abs(features.peak_freqs_cm1 - target_freq_cm1)))
    freq = float(features.peak_freqs_cm1[idx])
    return freq, float(features.peak_heights[idx]), abs(freq - target_freq_cm1)


# ---- feature-based fitness scores (lower is a better match; inf = unscoreable) ----


def gap_fitness(features: VDOSFeatures, target_center_cm1: float, target_width_cm1: float) -> float:
    """Distance from this structure's largest gap to a target gap, as the sum
    of center-position error and width error (cm-1).

    Deliberately additive rather than a single ratio: a structure with no
    gap at all (gap_width_cm1 == 0) scores inf, unambiguously worse than one
    with a gap in roughly the right place but the wrong width -- a ratio-
    based score risks the two error terms cancelling by coincidence instead.
    """
    if features.gap_width_cm1 <= 0:
        return float("inf")
    center = (features.gap_low_cm1 + features.gap_high_cm1) / 2
    return abs(center - target_center_cm1) + abs(features.gap_width_cm1 - target_width_cm1)


def mode_fitness(features: VDOSFeatures, target_freq_cm1: float) -> float:
    """Distance (cm-1) from this structure's nearest vDOS peak to a target
    frequency -- e.g. a specific low-frequency (soft/tilt) mode you want
    reproduced. inf if extract_features found no peaks at all."""
    result = nearest_peak(features, target_freq_cm1)
    return result[2] if result is not None else float("inf")
