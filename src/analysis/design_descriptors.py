"""Phonon and structure descriptors for one relaxed polymorph, as candidate
targets/inputs for an inverse-design model (scripts/analysis/
extract_design_descriptors.py builds the table; descriptor_selection.ipynb
decides which of them are worth modelling).

Everything is computed from what scripts/mlip/run_mlip_phonons.py already
saved -- phonopy_params.yaml (force constants) -- with no new MLIP calls.

Sublattices. Each atom is assigned to A, B or X. X is the anion (negative
formal oxidation state, or the structural check's x_site). Cations go by the
structural check's a_sites/b_sites when an element is listed on exactly one
of them, and otherwise -- an element on both sites, or no site list at all --
by its effective coordination to X (ECoN >= ECON_A_MIN: A, else B). Both
rules depend only on species and local environment, so a sublattice is always
a union of whole symmetry orbits; that's what makes the per-sublattice sums
below exact on an irreducible (symmetry-reduced) q-mesh, even though the
individual per-atom projections there are not.

Phonon descriptors, per sublattice s and for the whole cell:

    k_s        Einstein stiffness, (1/3) Tr Phi(j, j) averaged over j in s
               (eV/A^2): the on-site force constant, i.e. the restoring force
               on one atom displaced with all others held fixed. It is mass-
               independent and q-mesh-free, and it's exact: the BZ average of
               sum_nu |e_j,nu|^2 omega_nu^2 equals Tr Phi(j,j) / m_j.
    omega_E_s  sqrt(<Tr Phi(j,j) / (3 m_j)>_s), cm-1: the rms frequency of
               s's projected DOS, again exact from the force constants. For
               the whole cell this is the vDOS rms frequency.
    mean/std/log  centroid, width and exp<ln omega> of s's projected DOS (mesh)
    low/high_enrich  fraction of s's spectral weight in the lowest / highest
               10% of all modes, divided by 0.1 -- > 1 means s dominates
               that end of the spectrum (A-site rattling, B-X stretching),
               independent of the overall frequency scale
    msd_s      isotropic mean-square displacement U_iso at T (A^2): weights the
               soft end of s's spectrum (~ 1 / (m omega^2)), linked to Debye-
               Waller factors, rattling and anharmonicity

The whole-cell vDOS also gets Cv and S at T (k_B/atom), omega_max, and its
quantile function, which descriptor_selection.ipynb splits into a frequency
scale and a scale-free shape.

Mesh. A Monkhorst-Pack mesh offset by half a step, so Gamma (where three
acoustic modes have omega = 0 and the msd integrand diverges) is never sampled,
with a q-point count scaled as 1/nat so supercells aren't oversampled. Modes
with |omega| < OMEGA_MIN_CM1 are dropped from everything but the imaginary
fraction.

Structure descriptors: BX6 bond lengths and their distortion, quadratic
elongation, bond-angle variance and B off-centering; B-X-B bridge angles
(octahedral tilt); A-site effective coordination and A-X distances; bond-
valence sums per sublattice and the global instability index (GII), from
Brese-O'Keeffe universal parameters, as pymatgen's calculate_bv_sum uses them.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import ase
import numpy as np
from ase.neighborlist import neighbor_list
from phonopy import Phonopy
from pymatgen.analysis.bond_valence import BV_PARAMS
from pymatgen.core import Composition, Element
from scipy.spatial import ConvexHull

from src.analysis.composition_features import oxidation_guesses
from src.mlip.mlip_phonons import THZ_TO_CM1, load_compact

SUBLATTICES = ("A", "B", "X")

# Atoms with ECoN to X at least this are A-site, below it B-site (ideal: 12 vs 6;
# tilting brings A down to ~8, so the split sits well between the two).
ECON_A_MIN = 7.5
# Modes softer than this are acoustic modes right next to Gamma or residual
# numerical noise; dropped from ln(omega), msd and thermodynamics.
OMEGA_MIN_CM1 = 1.0
# q-points x atoms in the phonon mesh: ~16^3 for a 5-atom cubic cell.
MESH_Q_ATOMS = 20000
# Eigenvectors are diagonalized this many q-points at a time, so a 640-atom
# soft-mode supercell's (nq, 3 nat, 3 nat) array never has to exist in full.
Q_CHUNK = 16
# Brese-O'Keeffe bond-valence softness, A.
BV_B = 0.37
NEIGHBOR_CUTOFF_A = 6.0

HBAR = 1.054571817e-34  # J s
KB = 1.380649e-23  # J/K
AMU = 1.66053906660e-27  # kg
C_CM = 2.99792458e10  # cm/s
EV_PER_A2_PER_AMU_TO_CM2 = 1.602176634e-19 / 1e-20 / AMU / (2 * np.pi * C_CM) ** 2


@lru_cache(maxsize=None)
def oxidation_states(reduced_formula: str) -> dict[str, float] | None:
    """Most likely {element: formal oxidation state} for a formula: the first
    of composition_features.oxidation_guesses (pymatgen's charge-balance
    search), or None."""
    guesses = oxidation_guesses(reduced_formula)
    return dict(guesses[0]) if guesses else None


def phonopy_cell_as_atoms(phonon: Phonopy) -> ase.Atoms:
    """The phonopy unit cell as ase.Atoms, in phonopy's own atom order -- which
    the eigenvector and force-constant indices refer to."""
    cell = phonon.primitive
    return ase.Atoms(symbols=cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions,
                     masses=cell.masses, pbc=True)


# ---------------------------------------------------------------- sublattices


def _neighbors(atoms: ase.Atoms, cutoff: float = NEIGHBOR_CUTOFF_A):
    """Per atom, (neighbor indices, distances, vectors) sorted by distance."""
    i, j, d, D = neighbor_list("ijdD", atoms, cutoff)
    out = []
    for a in range(len(atoms)):
        sel = np.where(i == a)[0]
        order = sel[np.argsort(d[sel])]
        out.append((j[order], d[order], D[order]))
    return out


def econ(distances: np.ndarray, n_iter: int = 10) -> float:
    """Hoppe's effective coordination number over one atom's anion distances:
    a smooth CN that counts each bond by how close it is to the weighted mean
    bond length, so a tilted perovskite's A site reads ~8-10, not a cutoff-
    dependent integer."""
    d = np.asarray(distances)
    if d.size == 0:
        return 0.0
    d_av = d.min()
    for _ in range(n_iter):
        w = np.exp(1 - (d / d_av) ** 6)
        d_av = float((d * w).sum() / w.sum())
    return float(np.exp(1 - (d / d_av) ** 6).sum())


def assign_sublattices(
    atoms: ase.Atoms,
    oxi: dict[str, float] | None,
    a_sites: set[str] | None = None,
    b_sites: set[str] | None = None,
    x_site: str | None = None,
    neighbors=None,
) -> np.ndarray:
    """'A' / 'B' / 'X' per atom (see module docstring). '?' for a cation with
    no anion neighbors at all."""
    symbols = np.array(atoms.get_chemical_symbols())
    neighbors = neighbors or _neighbors(atoms)
    is_x = np.array([s == x_site or (oxi is not None and oxi.get(s, 0) < 0) for s in symbols])
    if not is_x.any():
        raise ValueError("no anion found (no x_site and no negative oxidation state)")
    roles = np.where(is_x, "X", "?").astype("<U1")
    a_sites, b_sites = a_sites or set(), b_sites or set()
    for k in np.where(~is_x)[0]:
        s = symbols[k]
        if s in a_sites and s not in b_sites:
            roles[k] = "A"
        elif s in b_sites and s not in a_sites:
            roles[k] = "B"
        else:
            idx, d, _ = neighbors[k]
            d_x = d[is_x[idx]]
            if d_x.size:
                roles[k] = "A" if econ(d_x[d_x < 1.5 * d_x.min()]) >= ECON_A_MIN else "B"
    return roles


# ---------------------------------------------------------------- phonons


def _mesh(phonon: Phonopy, q_atoms: int) -> list[int]:
    """Mesh with ~q_atoms / nat points, each axis proportional to its
    reciprocal-lattice-vector length."""
    nat = len(phonon.primitive)
    b = np.linalg.norm(np.linalg.inv(phonon.primitive.cell), axis=0)
    n_q = max(8.0, q_atoms / nat)
    scale = (n_q / np.prod(b)) ** (1 / 3)
    return [max(1, int(round(x * scale))) for x in b]


def _bose_terms(omega_cm1: np.ndarray, temperature_k: float):
    x = HBAR * 2 * np.pi * C_CM * omega_cm1 / (KB * temperature_k)
    with np.errstate(over="ignore"):
        cv = x**2 * np.exp(x) / np.expm1(x) ** 2
    s = x / np.expm1(x) - np.log1p(-np.exp(-x))
    coth = 1.0 / np.tanh(x / 2)
    return cv, s, coth


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, levels: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(levels, cdf, v)


def onsite_stiffness(phonon: Phonopy) -> np.ndarray:
    """(1/3) Tr Phi(j, j) per unit-cell atom, eV/A^2. Works for both full
    (n_super, n_super) and compact (n_prim, n_super) force constants."""
    fc = phonon.force_constants
    p2s = phonon.primitive.p2s_map
    rows = np.arange(len(p2s)) if fc.shape[0] == len(p2s) else p2s
    return np.array([np.trace(fc[r, s]) / 3 for r, s in zip(rows, p2s)])


def phonon_descriptors(
    phonon: Phonopy,
    roles: np.ndarray,
    temperature_k: float = 300.0,
    q_atoms: int = MESH_Q_ATOMS,
    quantile_levels: np.ndarray | None = None,
) -> tuple[dict, np.ndarray | None]:
    """Descriptor dict (see module docstring) and the whole-cell vDOS
    quantile function at quantile_levels (cm-1), if given."""
    masses = np.asarray(phonon.primitive.masses)
    nat = len(masses)
    out: dict = {}

    k = onsite_stiffness(phonon)
    omega_e2 = k / masses * EV_PER_A2_PER_AMU_TO_CM2  # cm-2, may be < 0 on an unstable cell
    out["omega_rms_fc_cm1"] = float(np.sqrt(max(omega_e2.mean(), 0.0)))
    for s in SUBLATTICES:
        m = roles == s
        out[f"k_{s}_eV_A2"] = float(k[m].mean()) if m.any() else np.nan
        out[f"omega_E_{s}_cm1"] = float(np.sqrt(max(omega_e2[m].mean(), 0.0))) if m.any() else np.nan

    mesh = _mesh(phonon, q_atoms)
    phonon.run_mesh(mesh, shift=[0.5, 0.5, 0.5], is_mesh_symmetry=True, with_eigenvectors=False)
    qpoints, q_weights = phonon.mesh.qpoints, np.asarray(phonon.mesh.weights, dtype=float)
    out["phonon_mesh"] = "x".join(map(str, mesh))

    freqs, proj, proj_m = [], [], []  # (nq, nb), (nq, nsub, nb) x2
    member = np.stack([roles == s for s in SUBLATTICES]).astype(float)  # (nsub, nat)
    for start in range(0, len(qpoints), Q_CHUNK):
        phonon.run_qpoints(qpoints[start:start + Q_CHUNK], with_eigenvectors=True)
        f = np.asarray(phonon.qpoints.frequencies) * THZ_TO_CM1
        e = np.asarray(phonon.qpoints.eigenvectors)  # (nq, 3 nat, nb)
        p_atom = (np.abs(e) ** 2).reshape(len(f), nat, 3, -1).sum(axis=2)  # (nq, nat, nb)
        freqs.append(f)
        proj.append(np.einsum("sa,qab->qsb", member, p_atom))
        proj_m.append(np.einsum("sa,qab->qsb", member / (masses * AMU), p_atom))  # sum_j |e_j|^2 / m_j
    freqs, proj, proj_m = np.concatenate(freqs), np.concatenate(proj), np.concatenate(proj_m)

    w_mode = np.broadcast_to(q_weights[:, None], freqs.shape)  # each mode counts once per q weight
    omega, w_tot = freqs.ravel(), w_mode.ravel()
    def per_sublattice(p):
        return (p * q_weights[:, None, None]).transpose(1, 0, 2).reshape(len(SUBLATTICES), -1)

    w_sub, w_sub_m = per_sublattice(proj), per_sublattice(proj_m)
    real = omega >= OMEGA_MIN_CM1

    out["imaginary_fraction"] = float(w_tot[omega < -OMEGA_MIN_CM1].sum() / w_tot.sum())
    out["omega_max_cm1"] = float(omega.max())
    o, w = omega[real], w_tot[real]
    out["omega_mean_cm1"] = float(np.average(o, weights=w))
    out["omega_log_cm1"] = float(np.exp(np.average(np.log(o), weights=w)))
    cv, entropy, coth = _bose_terms(o, temperature_k)
    # 3 nat modes per q: the mode average times 3 is per atom
    out["cv_kB_per_atom"] = float(3 * np.average(cv, weights=w))
    out["s_kB_per_atom"] = float(3 * np.average(entropy, weights=w))

    q10, q90 = _weighted_quantiles(o, w, np.array([0.1, 0.9]))
    # <u_j^2> = BZ average of sum_nu |e_j,nu|^2 hbar / (2 m_j omega) coth(hbar omega / 2kT); kg A^2
    hbar_coth_over_2omega = HBAR / (2 * 2 * np.pi * C_CM * o) * coth * 1e20
    for n, s in enumerate(SUBLATTICES):
        m = roles == s
        if not m.any():
            continue
        ws = w_sub[n][real]
        out[f"omega_mean_{s}_cm1"] = float(np.average(o, weights=ws))
        out[f"omega_std_{s}_cm1"] = float(np.sqrt(np.average((o - out[f"omega_mean_{s}_cm1"]) ** 2, weights=ws)))
        out[f"omega_log_{s}_cm1"] = float(np.exp(np.average(np.log(o), weights=ws)))
        out[f"low_enrich_{s}"] = float(ws[o <= q10].sum() / ws.sum() / 0.1)
        out[f"high_enrich_{s}"] = float(ws[o >= q90].sum() / ws.sum() / 0.1)
        # U_iso = <u^2> / 3, averaged over the atoms of s
        msd = (w_sub_m[n][real] * hbar_coth_over_2omega).sum() / q_weights.sum() / m.sum()
        out[f"msd_{s}_A2"] = float(msd / 3)

    quantiles = _weighted_quantiles(o, w, quantile_levels) if quantile_levels is not None else None
    return out, quantiles


# ---------------------------------------------------------------- structure


@lru_cache(maxsize=None)
def _bv_r0(cation: str, anion: str) -> float | None:
    """Brese-O'Keeffe R0 for a cation-anion pair (A), from pymatgen's
    per-element (r, c) table -- the same combination rule calculate_bv_sum
    uses. None if either element isn't tabulated."""
    try:
        p1, p2 = BV_PARAMS[Element(cation)], BV_PARAMS[Element(anion)]
    except KeyError:
        return None
    r1, r2, c1, c2 = p1["r"], p2["r"], p1["c"], p2["c"]
    return r1 + r2 - r1 * r2 * (np.sqrt(c1) - np.sqrt(c2)) ** 2 / (c1 * r1 + c2 * r2)


def _bvs(atoms: ase.Atoms, is_x: np.ndarray, neighbors) -> np.ndarray:
    """Bond-valence sum per atom, counting only cation-anion bonds."""
    symbols = atoms.get_chemical_symbols()
    out = np.zeros(len(atoms))
    for a, (idx, d, _) in enumerate(neighbors):
        partners = ~is_x[idx] if is_x[a] else is_x[idx]
        for b, dist in zip(idx[partners], d[partners]):
            cat, an = (symbols[b], symbols[a]) if is_x[a] else (symbols[a], symbols[b])
            r0 = _bv_r0(cat, an)
            if r0 is None:
                out[a] = np.nan
                break
            out[a] += np.exp((r0 - dist) / BV_B)
    return out


def _octahedron(vectors: np.ndarray) -> dict:
    """Distortion of one BX6 octahedron from its 6 B->X vectors."""
    d = np.linalg.norm(vectors, axis=1)
    unit = vectors / d[:, None]
    angles = np.degrees(np.arccos(np.clip(unit @ unit.T, -1, 1)))[np.triu_indices(6, 1)]
    cis = np.sort(angles)[:12]  # the 3 trans pairs are the ~180 deg ones
    volume = ConvexHull(vectors).volume
    l0 = (3 * volume / 4) ** (1 / 3)  # center-vertex distance of a regular octahedron of that volume
    return {
        "d_BX": d.mean(),
        "bond_distortion": float(np.mean(((d - d.mean()) / d.mean()) ** 2)),
        "quadratic_elongation": float(np.mean((d / l0) ** 2)),
        "angle_variance_deg2": float(np.sum((cis - 90) ** 2) / 11),
        "off_centering_A": float(np.linalg.norm(vectors.mean(axis=0))),
    }


def structure_descriptors(atoms: ase.Atoms, roles: np.ndarray, oxi: dict[str, float] | None,
                          neighbors=None) -> dict:
    neighbors = neighbors or _neighbors(atoms)
    symbols = atoms.get_chemical_symbols()
    is_x = roles == "X"
    out: dict = {"n_B": int((roles == "B").sum())}
    out["volume_per_fu_A3"] = atoms.get_volume() / max(out["n_B"], 1)
    out["a_pc_A"] = out["volume_per_fu_A3"] ** (1 / 3)

    octs = []
    for a in np.where(roles == "B")[0]:
        idx, d, D = neighbors[a]
        D = D[is_x[idx]][:6]
        if len(D) == 6:
            octs.append(_octahedron(D))
    for key in ("d_BX", "bond_distortion", "quadratic_elongation", "angle_variance_deg2", "off_centering_A"):
        out[f"oct_{key}"] = float(np.mean([o[key] for o in octs])) if octs else np.nan
    out["oct_d_BX_std_A"] = float(np.std([o["d_BX"] for o in octs])) if octs else np.nan  # B/B' size contrast

    bridge = []
    for a in np.where(is_x)[0]:
        idx, d, D = neighbors[a]
        D = D[roles[idx] == "B"][:2]
        if len(D) == 2:
            cos = D[0] @ D[1] / np.linalg.norm(D[0]) / np.linalg.norm(D[1])
            bridge.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    out["bxb_angle_mean_deg"] = float(np.mean(bridge)) if bridge else np.nan
    out["bxb_angle_min_deg"] = float(np.min(bridge)) if bridge else np.nan
    out["tilt_proxy_deg"] = (180 - out["bxb_angle_mean_deg"]) / 2

    a_econ, a_mean, a_min, a_std = [], [], [], []
    for a in np.where(roles == "A")[0]:
        idx, d, _ = neighbors[a]
        d_x = d[is_x[idx]][:12]
        if d_x.size:
            a_econ.append(econ(d_x[d_x < 1.5 * d_x.min()]))
            a_mean.append(d_x.mean())
            a_min.append(d_x.min())
            a_std.append(d_x.std() / d_x.mean())
    out["A_econ"] = float(np.mean(a_econ)) if a_econ else np.nan
    out["d_AX_mean_A"] = float(np.mean(a_mean)) if a_mean else np.nan
    out["d_AX_min_A"] = float(np.mean(a_min)) if a_min else np.nan
    out["d_AX_rel_spread"] = float(np.mean(a_std)) if a_std else np.nan
    # the tolerance factor the relaxed structure actually realizes
    out["t_geom"] = out["d_AX_mean_A"] / (np.sqrt(2) * out["oct_d_BX"])

    bvs = _bvs(atoms, is_x, neighbors)
    if oxi is not None and not np.isnan(bvs).any():
        valence = np.array([abs(oxi.get(s, np.nan)) for s in symbols])
        dev = bvs - valence
        out["gii"] = float(np.sqrt(np.nanmean(dev**2)))
        for s in SUBLATTICES:
            # < 0: underbonded (a rattler on A), > 0: compressed
            out[f"bvs_dev_{s}"] = float(np.nanmean(dev[roles == s])) if (roles == s).any() else np.nan
    return out


def site_heterogeneity(atoms: ase.Atoms, roles: np.ndarray, oxi: dict[str, float] | None) -> dict:
    """Disorder/ordering contrast within the A and B sublattices (double and
    mixed perovskites): cluster_compositions.py's descriptors average these away."""
    masses = atoms.get_masses()
    symbols = np.array(atoms.get_chemical_symbols())
    out = {}
    for s in ("A", "B"):
        m = roles == s
        out[f"n_species_{s}"] = int(len(set(symbols[m])))
        out[f"mass_cv_{s}"] = float(masses[m].std() / masses[m].mean()) if m.any() else np.nan
        if oxi is not None and m.any():
            q = np.array([oxi.get(x, np.nan) for x in symbols[m]])
            out[f"oxi_spread_{s}"] = float(np.nanmax(q) - np.nanmin(q))
    out["n_species_X"] = int(len(set(symbols[roles == "X"])))
    return out


# --- one state directory -> descriptor row; tables of them ---

QUANTILE_LEVELS = (np.arange(64) + 0.5) / 64


def describe_state_dir(job: dict) -> tuple[dict, np.ndarray | None]:
    """Every descriptor for one pipeline state directory, as (row, quantile
    function at QUANTILE_LEVELS). `job`: {"key", "dir", and optionally the
    metadata's "a_sites"/"b_sites" (sets) and "x_site"}. Errors are returned
    in the row ("error"), not raised, so it can be mapped over a pool."""
    warnings.filterwarnings("ignore")
    row = {"key": job["key"]}
    try:
        phonon = load_compact(job["dir"] / "phonopy_params.yaml")
        atoms = phonopy_cell_as_atoms(phonon)
        oxi = oxidation_states(Composition(atoms.get_chemical_formula()).reduced_formula)
        neighbors = _neighbors(atoms)
        roles = assign_sublattices(atoms, oxi, job.get("a_sites"), job.get("b_sites"), job.get("x_site"),
                                      neighbors=neighbors)
        row["sublattice_unassigned"] = int((roles == "?").sum())
        row["n_A"], row["n_X"] = int((roles == "A").sum()), int((roles == "X").sum())
        phonons, quantiles = phonon_descriptors(phonon, roles, quantile_levels=QUANTILE_LEVELS)
        row.update(phonons)
        row.update(structure_descriptors(atoms, roles, oxi, neighbors))
        row.update(site_heterogeneity(atoms, roles, oxi))
        return row, quantiles
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, None


def instability_descriptors(polymorphs, candidates):
    """Per selected polymorph (select_stable_polymorphs.py's stable_polymorphs.csv):
    energy gained relative to its own run's symmetric state and to the lowest
    cubic (Pm-3m) symmetric state of its formula, with those states' lowest
    frequency -- how unstable the aristotype it descended from was. DataFrames
    in, a DataFrame on polymorphs' index out."""
    import pandas as pd

    sym = candidates[candidates["state"] == "symmetric"]
    by_run = sym.set_index("run_name")[["energy_eV_per_atom", "omega_min_cm1"]]
    cubic = (sym[sym["spacegroup_number"] == 221].sort_values("energy_eV_per_atom")
             .groupby("formula")[["energy_eV_per_atom", "omega_min_cm1"]].first())
    out = pd.DataFrame(index=polymorphs.index)
    e = polymorphs["energy_eV_per_atom"]
    e_sym = polymorphs["run_name"].map(by_run["energy_eV_per_atom"])
    out["distortion_energy_meV_per_atom"] = 1000 * (e_sym - e)
    out["omega_min_symmetric_cm1"] = polymorphs["run_name"].map(by_run["omega_min_cm1"])
    out["cubic_distortion_energy_meV_per_atom"] = 1000 * (polymorphs["formula"].map(cubic["energy_eV_per_atom"]) - e)
    out["cubic_omega_min_cm1"] = polymorphs["formula"].map(cubic["omega_min_cm1"])
    return out


def quantiles_path(out_csv: Path) -> Path:
    """The quantile-function npz that goes with a descriptor CSV."""
    return out_csv.with_name(out_csv.stem + "_quantiles.npz")


def load_quantiles(npz_path: Path) -> dict[str, np.ndarray]:
    """key -> quantile function (cm-1) from a save_descriptor_table npz."""
    q = np.load(npz_path)
    return dict(zip(q["keys"], q["quantiles_cm1"]))


def save_descriptor_table(meta, rows: list, quantiles: dict, out_csv: Path) -> None:
    """Write the descriptor CSV (meta joined to the concatenated row frames on
    "key") and its quantiles npz (keys, levels, quantiles_cm1) together."""
    import pandas as pd

    table = meta.merge(pd.concat(rows, ignore_index=True), on="key")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".tmp.csv")
    table.to_csv(tmp, index=False)
    keys = [k for k in table["key"] if k in quantiles]
    npz = quantiles_path(out_csv)
    np.savez(npz.with_suffix(".tmp.npz"), levels=QUANTILE_LEVELS,
             quantiles_cm1=np.array([quantiles[k] for k in keys]).reshape(len(keys), len(QUANTILE_LEVELS)),
             keys=np.array(keys, dtype=str))
    # replace both only once both are written, so an interruption never leaves a mismatched pair
    tmp.replace(out_csv)
    npz.with_suffix(".tmp.npz").replace(npz)
