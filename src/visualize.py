"""Visualize phonon modes from a Quantum ESPRESSO ph.x DFPT run.

parse_qe_modes() reads eigendisplacements QE prints under each `freq (...)`
line in a ph.x dynamical-matrix file (<prefix>.dynN), at a given q-point.
plot_mode_arrows() renders a static arrow diagram of one mode;
animate_mode_gif() renders it oscillating.

Note: the .ph.out log itself does not contain eigendisplacements (ph.x only
prints frequencies there); the eigenvectors live in the .dynN files, one per
q-point, under a "Diagonalizing the dynamical matrix" section.
"""

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("my_style")
from matplotlib.animation import FuncAnimation, PillowWriter
from ase.io import read

# Adjust per structure -- these are CsPbI3-specific.
SITE_COLORS = {"Cs": "tab:purple", "Pb": "tab:gray", "I": "tab:orange"}
SITE_SIZES = {"Cs": 300, "Pb": 250, "I": 150}


def parse_qe_modes(dyn_path, q=None, tol=1e-4):
    """Return (freqs, vectors) parsed from a ph.x dynamical-matrix file (<prefix>.dynN).

    dyn_path: path to the .dynN file for the q-point of interest (ph.x writes
    one such file per irreducible q-point; the eigendisplacements are not
    printed to the .ph.out log).
    q: optional (qx, qy, qz) in the units the file uses (2*pi/alat), checked
    against the "q = ( ... )" line above the frequencies as a sanity check.
    freqs: array of length 3*n_atoms, in cm-1.
    vectors: list of (n_atoms, 3) arrays, one per mode, real part of the
    eigendisplacement (imaginary part discarded -- fine at high-symmetry
    q-points where the eigenvectors can be chosen real).
    """
    text = open(dyn_path).read()
    start = text.index("Diagonalizing the dynamical matrix")
    block = text[start:]

    if q is not None:
        qm = re.search(r"q\s*=\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)", block)
        found_q = tuple(float(x) for x in qm.groups())
        if any(abs(a - b) > tol for a, b in zip(found_q, q)):
            raise ValueError(f"{dyn_path}: q-point mismatch, file has {found_q}, expected {q}")

    freqs, vectors, current_vec = [], [], []
    for line in block.splitlines():
        m = re.match(r"\s*freq \(\s*\d+\) =\s*[-\d.]+ \[THz\] =\s*([-\d.]+) \[cm-1\]", line)
        if m:
            if current_vec:
                vectors.append(np.array(current_vec))
                current_vec = []
            freqs.append(float(m.group(1)))
            continue
        m2 = re.match(r"\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)", line)
        if m2:
            re_x, _, re_y, _, re_z, _ = map(float, m2.groups())
            current_vec.append([re_x, re_y, re_z])
    if current_vec:
        vectors.append(np.array(current_vec))
    return np.array(freqs), vectors


def plot_mode_arrows(cif_path, vectors, mode_index, freqs, amplitude=1.0, out_path="mode_arrows.png"):
    """Static 3D plot: atoms as spheres, arrows showing the phonon eigendisplacement."""
    atoms = read(cif_path)
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    disp = vectors[mode_index]
    disp = disp / np.linalg.norm(disp)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    for pos, sym in zip(positions, symbols):
        ax.scatter(*pos, color=SITE_COLORS.get(sym, "tab:blue"),
                   s=SITE_SIZES.get(sym, 150), edgecolor="black", label=sym)

    ax.quiver(
        positions[:, 0], positions[:, 1], positions[:, 2],
        disp[:, 0], disp[:, 1], disp[:, 2],
        length=amplitude, color="crimson", linewidth=2, normalize=False,
    )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    ax.set_title(f"Mode {mode_index}: {freqs[mode_index]:.1f} cm$^{{-1}}$")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def animate_mode_gif(cif_path, vectors, mode_index, freqs, amplitude=0.4, n_frames=30, out_path="mode.gif"):
    """Animated GIF of atoms oscillating along the phonon eigendisplacement."""
    atoms = read(cif_path)
    base_positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    disp = vectors[mode_index]
    disp = disp / np.linalg.norm(disp)

    point_colors = [SITE_COLORS.get(s, "tab:blue") for s in symbols]
    point_sizes = [SITE_SIZES.get(s, 150) for s in symbols]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        base_positions[:, 0], base_positions[:, 1], base_positions[:, 2],
        c=point_colors, s=point_sizes, edgecolor="black",
    )
    margin = amplitude + 1.0
    ax.set_xlim(base_positions[:, 0].min() - margin, base_positions[:, 0].max() + margin)
    ax.set_ylim(base_positions[:, 1].min() - margin, base_positions[:, 1].max() + margin)
    ax.set_zlim(base_positions[:, 2].min() - margin, base_positions[:, 2].max() + margin)
    ax.set_title(f"Mode {mode_index}: {freqs[mode_index]:.1f} cm$^{{-1}}$")

    def update(frame):
        phase = 2 * np.pi * frame / n_frames
        new_positions = base_positions + amplitude * np.cos(phase) * disp
        scatter._offsets3d = (new_positions[:, 0], new_positions[:, 1], new_positions[:, 2])
        return (scatter,)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"wrote {out_path}")


def _parse_matdyn_freq(freq_path):
    """Parse a matdyn.x plottable frequency file (flfrq output).

    Returns (qpoints, freqs): qpoints is (nks, 3) in the crystal-coordinate
    units matdyn.x was run with, freqs is (nks, nbnd) in cm-1.
    """
    with open(freq_path) as f:
        header = f.readline()
        nbnd = int(re.search(r"nbnd=\s*(\d+)", header).group(1))
        nks = int(re.search(r"nks=\s*(\d+)", header).group(1))
        tokens = [float(t) for t in f.read().split()]

    qpoints = np.empty((nks, 3))
    freqs = np.empty((nks, nbnd))
    i = 0
    for k in range(nks):
        qpoints[k] = tokens[i:i + 3]
        i += 3
        freqs[k] = tokens[i:i + nbnd]
        i += nbnd
    return qpoints, freqs


def plot_dispersion(freq_path, labels, out_path="dispersion.png"):
    """Plot a phonon dispersion from a matdyn.x .freq file.

    labels: high-symmetry point labels in path order (e.g. ["G", "X", "M",
    "G", "R", "X"]), assumed evenly spaced along the path -- i.e. matdyn.x's
    q_in_band_form input used the same number of points between each pair of
    consecutive labels.
    """
    qpoints, freqs = _parse_matdyn_freq(freq_path)
    nks = qpoints.shape[0]
    seg_dist = np.linalg.norm(np.diff(qpoints, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(seg_dist)])
    label_idx = np.linspace(0, nks - 1, len(labels)).round().astype(int)
    label_x = x[label_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    for band in range(freqs.shape[1]):
        ax.plot(x, freqs[:, band], color="tab:blue", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for lx in label_x[1:-1]:
        ax.axvline(lx, color="gray", linewidth=0.6)
    ax.set_xticks(label_x)
    ax.set_xticklabels(labels)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("Frequency (cm$^{-1}$)")
    ax.set_title("Phonon dispersion")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cif", type=str, default="./data/cubic_structures/CsCuCl3.cif", help="CIF of the base structure")
    parser.add_argument("--out", type=str, default="./output", help="Output directory")
    parser.add_argument("--dyn", type=str, default="./sherlock_outputs/ph_first/CsCuCl3/CsCuCl3.dyn1", help="QE .dynN file for the q-point of interest")
    parser.add_argument("--freq", type=str, default="./sherlock_outputs/ph_first/CsCuCl3/CsCuCl3.freq", help="matdyn.x .freq file for the dispersion plot")
    parser.add_argument("-q", "--q", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="q-point (qx, qy, qz) in the units the .dynN file uses (2*pi/alat)")

    args = parser.parse_args()
    freqs, vectors = parse_qe_modes(args.dyn, q=args.q)
    print("Gamma-point frequencies (cm-1):", freqs)
    lowest = int(np.argmin(freqs))
    plot_mode_arrows(args.cif, vectors, lowest, freqs, out_path=args.out + "/gamma_softmode_arrows.png")
    animate_mode_gif(args.cif, vectors, lowest, freqs, out_path=args.out + "/gamma_softmode.gif")

    plot_dispersion(
        args.freq,
        labels=["G", "X", "M", "G", "R", "X"],
        out_path=args.out + "/CsCuCl3_dispersion.png",
    )
