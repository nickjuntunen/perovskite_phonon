"""Figures of phonon modes and dispersions from a Quantum ESPRESSO ph.x DFPT
run; scripts/qe/visualize_modes.py is the command-line front end.

plot_mode_arrows() renders a static arrow diagram of one mode and
animate_mode_gif() renders it oscillating, from the eigendisplacements of a
<prefix>.dynN file (soft_mode_distortion.parse_dyn_file: pass
`list(dyn.eigenvectors.real)` -- the real part is all there is at
high-symmetry q-points, where the eigenvectors can be chosen real).
plot_dispersion() draws a matdyn.x .freq band structure, optionally with the
vDOS alongside.

Note: the .ph.out log itself does not contain eigendisplacements (ph.x only
prints frequencies there); the eigenvectors live in the .dynN files, one per
q-point, under a "Diagonalizing the dynamical matrix" section.
"""

import ase
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from ase.io import read

from src.qe.phonon_dos import parse_matdyn_dos
from src.qe.qe_phonon_io import parse_matdyn_freq, path_distances
from src.qe.vdos_features import smooth_dos

# Adjust per structure -- these are CsPbI3-specific.
SITE_COLORS = {"Cs": "tab:purple", "Pb": "tab:gray", "I": "tab:orange"}
SITE_SIZES = {"Cs": 300, "Pb": 250, "I": 150}


def plot_mode_arrows(cif_path, vectors, mode_index, freqs, amplitude=1, out_path="mode_arrows.png"):
    """Static 3D plot: atoms as spheres, arrows showing the phonon eigendisplacement."""
    atoms = read(cif_path)
    assert isinstance(atoms, ase.Atoms)
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
    assert isinstance(atoms, ase.Atoms)
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


def plot_dispersion(freq_path, labels, dos_path=None, dos_sigma_cm1=2.0, out_path="dispersion.png"):
    """Plot a phonon dispersion from a matdyn.x .freq file.

    labels: high-symmetry point labels in path order (e.g. ["G", "X", "M",
    "G", "R", "X"]), assumed evenly spaced along the path -- i.e. matdyn.x's
    q_in_band_form input used the same number of points between each pair of
    consecutive labels.
    dos_path: optional matdyn.x DOS file (fldos, see phonon_dos.parse_matdyn_dos)
    to plot as a vDOS panel alongside the dispersion, sharing the frequency
    axis. The raw histogram is Gaussian-broadened (dos_sigma_cm1) before
    plotting -- see vdos_features.smooth_dos.
    """
    qpoints, freqs = parse_matdyn_freq(freq_path)
    nks = qpoints.shape[0]
    x = path_distances(qpoints)
    label_idx = np.linspace(0, nks - 1, len(labels)).round().astype(int)
    label_x = x[label_idx]

    if dos_path is not None:
        fig, (ax, ax_dos) = plt.subplots(
            1, 2, figsize=(9, 5), sharey=True,
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
        )
    else:
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

    if dos_path is not None:
        dos_freq, dos = parse_matdyn_dos(dos_path)
        dos_smooth = smooth_dos(dos_freq, dos, sigma_cm1=dos_sigma_cm1)
        ax_dos.fill_betweenx(dos_freq, dos_smooth, color="tab:blue", alpha=0.3)
        ax_dos.plot(dos_smooth, dos_freq, color="tab:blue", linewidth=1.2)
        ax_dos.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_dos.set_xlim(0, None)
        ax_dos.set_xlabel("vDOS")
        ax_dos.set_title("VDOS")
        ax_dos.tick_params(labelleft=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
