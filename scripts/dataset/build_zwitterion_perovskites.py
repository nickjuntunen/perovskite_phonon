"""Build organic-zwitterion lead halide perovskites from the cysteamine
prototype: in PbX2(S-R-NR3+), the thiolate S- takes one X site of each PbX3
octahedron and the ammonium / amidinium end sits in the A-site cage.

Prototype: data/organic_halide_replacement_structures/CYS-PbCl-ZW.cif,
6 x PbCl2(SCH2CH2NH3) in a hexagonal [111] setting of the cubic perovskite.
Each picked molecule (get_sh_molecules.ipynb -> pubchem_zwitterion_picks.csv)
is protonated into its zwitterion (S-, N+), embedded with RDKit (ETKDG
conformers), and put on each of the six S sites: its S on the site, its S->N+
axis along the prototype's S->N, chirality kept (proper rotations only). The
conformer (ETKDG plus every rotatable bond driven to 60/180/300 deg) and the
rotation about that axis are chosen to minimise steric
overlap with the framework and the other five molecules: best symmetric
packing (same pose on all six sites, as in the prototype), then site-by-site
refinement.

Per halide the prototype lattice is scaled by the CsPbX3 pseudocubic lattice
ratio (Cl 1, Br 5.87/5.61, I 6.29/5.61). Molecules bigger than cysteamine
don't fit the cage: when no pose is clash-free, the whole cell is expanded
isotropically in 2% steps (up to MAX_EXPANSION) until one is. The expansion is
recorded (`expansion`) as a steric-fit measure. These are starting points for
the MLIP cell relaxation, not relaxed structures.

Outputs, in --out:
    structures/zw-<CID>-Pb<X>.cif
    manifest.csv         name, cif, cid, smiles, zwitterion_smiles, iupac, halide, cation_class,
                         bonds_N_S, formula, n_atoms, expansion, clash-free flag, min contacts
    mlip_structures.txt  structures/<name>.cif per line (run_mlip_phonons.py input list)

Usage:
    uv run python3 scripts/dataset/build_zwitterion_perovskites.py
    # steric-fit screen of the whole aliphatic pool, one halide, no CIFs:
    uv run python3 scripts/dataset/build_zwitterion_perovskites.py --picks pubchem_zwitterions_aliphatic.parquet \\
        --halides Cl --max-expansion 1.10 --screen --jobs 10 --out data/organic_halide_replacement_structures/fit_screen
    uv run python3 scripts/dataset/build_zwitterion_perovskites.py --cids 6058 --halides Cl
"""

import argparse
import itertools
import multiprocessing
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_DIR

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")

ZW_DIR = DATA_DIR / "organic_halide_replacement_structures"
PROTOTYPE = ZW_DIR / "CYS-PbCl-ZW.cif"
LATTICE_SCALE = {"Cl": 1.0, "Br": 5.87 / 5.61, "I": 6.29 / 5.61}  # cubic CsPbX3 a, relative to Cl
# minimum non-bonded contacts (A); the prototype has H-H 2.44, H-X 2.20, C/N-C/N 3.20, C/N-Pb/Cl 3.17
MIN_CONTACT = {("fw", "fw"): 0.0, ("H", "H"): 1.9, ("H", "heavy"): 2.0, ("H", "fw"): 2.0, ("heavy", "heavy"): 2.9, ("fw", "heavy"): 2.9}
EXPANSION_STEP, MAX_EXPANSION = 0.02, 1.40
SOFT_MARGIN = 0.3   # A; pose search prefers contacts this much above the minimum
N_SYMMETRIC = 100   # symmetric (same pose on all six sites) starts scored on the whole cell
N_CONFORMERS, N_ANGLES = 60, 24
AXIS_TILTS_DEG, N_AZIMUTHS = (0.0, 25.0), 6  # S->N axis: the prototype's, or tilted 25 deg (6 ways)

THIOL_SITE = Chem.MolFromSmarts("[CX4][$([SX2H1]),$([SX1-])]")
AMINE = Chem.MolFromSmarts("[NX3;+0;!$(N[a]);!$(N[#6,#16]=[#7,#8,#16]);!$(N[#6]=[#6]);!$(N[#7,#8,#16]);!$(NC#N)]")
AMIDINE_IMINE = Chem.MolFromSmarts("[NX3;+0][CX3;!a]=[NX2;+0]")  # match[2] is the N that protonates
N_PLUS = Chem.MolFromSmarts("[#7+]")


def kind(sym):
    return "H" if sym == "H" else ("fw" if sym in ("Pb", "Cl", "Br", "I") else "heavy")


# --- prototype ------------------------------------------------------------------
def prototype_sites(path=PROTOTYPE):
    """Framework (Pb, X) in fractional coordinates, and per molecule site the S
    position, the S->N and S->C directions (fractional / Cartesian on the prototype lattice)."""
    s = Structure.from_file(path)
    sym = [x.specie.symbol for x in s]
    D = s.distance_matrix
    framework = [(sym[i], s[i].frac_coords) for i in range(len(s)) if sym[i] in ("Pb", "Cl")]
    sites = []
    for i in (i for i in range(len(s)) if sym[i] == "S"):
        # the molecule's C bonded to S, and its N, by walking the bond graph
        org = [j for j in range(len(s)) if sym[j] in "CNHS"]
        seen, stack = {i}, [i]
        while stack:
            a = stack.pop()
            for b in org:
                if b not in seen and D[a, b] < (1.25 if "H" in (sym[a], sym[b]) else 1.95):
                    seen.add(b)
                    stack.append(b)
        n = next(j for j in seen if sym[j] == "N")
        c = min((j for j in seen if sym[j] == "C"), key=lambda j: D[i, j])
        def vec(j):  # minimum-image Cartesian vector S -> j
            return s.lattice.get_cartesian_coords(s[j].frac_coords - s[i].frac_coords
                                                  - np.round(s[j].frac_coords - s[i].frac_coords))
        sites.append({"s_frac": s[i].frac_coords, "to_n": vec(n), "to_c": vec(c)})
    return s.lattice, framework, sites


# --- molecule -------------------------------------------------------------------
def zwitterion(smiles):
    """RDKit mol of the zwitterion (S-, N+) with Hs, and the indices of S, its C and the N+."""
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    c_idx, s_idx = mol.GetSubstructMatch(THIOL_SITE)
    s_atom = mol.GetAtomWithIdx(s_idx)
    s_atom.SetFormalCharge(-1)
    s_atom.SetNumExplicitHs(0)
    s_atom.SetNoImplicit(True)
    dm = Chem.GetDistanceMatrix(mol)
    plus = [m[0] for m in mol.GetSubstructMatches(N_PLUS)]
    if plus:
        n_idx = plus[0]
    else:
        amidine = [m[2] for m in mol.GetSubstructMatches(AMIDINE_IMINE)]
        amine = [m[0] for m in mol.GetSubstructMatches(AMINE)]
        n_idx = min(amidine or amine, key=lambda j: dm[j, s_idx])
        n_atom = mol.GetAtomWithIdx(n_idx)
        n_atom.SetFormalCharge(1)
        n_atom.SetNumExplicitHs(n_atom.GetTotalNumHs() + 1)
        n_atom.SetNoImplicit(True)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    assert Chem.GetFormalCharge(mol) == 0, smiles
    return Chem.AddHs(mol), s_idx, c_idx, n_idx


def conformers(mol, n_embed=10, seed=0, max_conformers=N_CONFORMERS):
    """ETKDG embeddings, plus every rotatable heavy-atom bond set to 60 / 180 / 300 deg
    (ETKDG alone tends to return only the folded, N-H...S bonded shape), without
    intramolecular clashes; at most max_conformers, picked at random."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_embed, params=params))
    if not ids:
        raise RuntimeError(f"embedding failed: {Chem.MolToSmiles(mol)}")
    heavy_nb = lambda atom, other: [x.GetIdx() for x in atom.GetNeighbors()
                                    if x.GetAtomicNum() > 1 and x.GetIdx() != other]
    torsions = []  # acyclic single bonds with heavy atoms on both sides: dihedral a-b-c-d
    for bond in mol.GetBonds():
        b, c = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        na, nd = heavy_nb(b, c.GetIdx()), heavy_nb(c, b.GetIdx())
        if na and nd:
            torsions.append((na[0], b.GetIdx(), c.GetIdx(), nd[0]))
    topo = Chem.GetDistanceMatrix(mol)
    far = np.triu(topo > 3, 1)  # pairs more than 3 bonds apart
    rng = np.random.default_rng(seed)
    out = []
    for cid in ids:
        states = itertools.product((60.0, 180.0, 300.0), repeat=len(torsions))
        for st in states:
            conf = Chem.Conformer(mol.GetConformer(cid))
            for t, ang in zip(torsions, st):
                rdMolTransforms.SetDihedralDeg(conf, *t, ang)
            x = conf.GetPositions()
            d = np.linalg.norm(x[:, None] - x[None], axis=-1)
            if (d[far] > 1.8).all():
                out.append(x)
    if not out:  # nothing clash-free after driving torsions: fall back to the embeddings
        out = [mol.GetConformer(i).GetPositions() for i in ids]
    # drop near duplicates (heavy-atom positions within 0.1 A)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    uniq = []
    for x in out:
        if all(np.abs(x[heavy] - y[heavy]).max() > 0.1 for y in uniq):
            uniq.append(x)
    if len(uniq) > max_conformers:
        uniq = [uniq[i] for i in sorted(rng.choice(len(uniq), max_conformers, replace=False))]
    return uniq


def frame(axis, second):
    e1 = axis / np.linalg.norm(axis)
    e2 = second - e1 * (second @ e1)
    e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def poses(xyz, s_idx, c_idx, n_idx, site, lattice):
    """Every conformer x S->N axis (the prototype's or tilted) x rotation about it,
    placed on the site: array (pose, atom, 3)."""
    proto = frame(site["to_n"], site["to_c"])
    s_cart = lattice.get_cartesian_coords(site["s_frac"])
    targets = [proto]
    for tilt in AXIS_TILTS_DEG[1:]:
        for phi in np.linspace(0, 2 * np.pi, N_AZIMUTHS, endpoint=False):
            hinge = np.cos(phi) * proto[:, 1] + np.sin(phi) * proto[:, 2]  # perpendicular to S->N
            targets.append(Rotation.from_rotvec(np.radians(tilt) * hinge).as_matrix() @ proto)
    spins = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)[:, None]
    out = []
    for target in targets:
        rots = Rotation.from_rotvec(spins * target[:, 0]).as_matrix()
        for x in xyz:
            x = x - x[s_idx]
            base = x @ frame(x[n_idx], x[c_idx]) @ target.T  # proper rotation: chirality kept
            out.append(np.einsum("aj,rij->rai", base, rots) + s_cart)
    return np.concatenate(out)


# --- sterics --------------------------------------------------------------------
KINDS = ("H", "heavy", "fw")
THRESH = np.array([[MIN_CONTACT[tuple(sorted((a, b)))] for b in KINDS] for a in KINDS])


class Environment:
    """Periodic images (3x3x3) of a set of atoms, for contact queries."""

    def __init__(self, lattice, symbols, cart):
        shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        self.kinds = np.tile([KINDS.index(kind(x)) for x in symbols], len(shifts))
        cart = np.asarray(cart).reshape(-1, 3)
        self.xyz = (cart[None] + (shifts @ lattice.matrix)[:, None]).reshape(-1, 3)
        self.tree = cKDTree(self.xyz)

    def contacts(self, kinds, cart, no_fw=None, k=24, cutoff=3.0):
        """Distances (..., atom, k) to the nearest environment atoms within 3.0 A, the
        contact threshold for each, and the environment kind; no_fw: atoms whose
        framework contacts are bonds (the thiolate S)."""
        d, j = self.tree.query(cart, k=k, distance_upper_bound=cutoff)
        hit = np.isfinite(d)
        ek = np.where(hit, self.kinds[np.minimum(j, len(self.kinds) - 1)], 0)
        thr = THRESH[np.broadcast_to(kinds[:, None], d.shape), ek]
        if no_fw is not None:
            hit &= ~(no_fw[:, None] & (ek == KINDS.index("fw")))
        return np.where(hit, d, np.inf), thr, ek

    def penalty(self, kinds, cart, no_fw=None, margin=0.0):
        """Sum of squared contact violations, per pose (cart: pose, atom, 3); margin > 0
        also penalises contacts that are legal but tight (prefers roomy poses)."""
        d, thr, _ = self.contacts(kinds, cart, no_fw, cutoff=3.0 + margin)
        return (np.clip(thr + np.where(thr > 0, margin, 0) - d, 0, None) ** 2).sum(axis=(-1, -2))

    def closest(self, kinds, cart, no_fw=None):
        d, _, ek = self.contacts(kinds, cart, no_fw)
        out = {}
        for a, b in itertools.product(range(3), range(3)):
            m = (kinds[:, None] == a) & (ek == b) & np.isfinite(d)
            if m.any():
                pair = tuple(sorted((KINDS[a], KINDS[b])))
                out[pair] = min(out.get(pair, np.inf), d[m].min())
        return out


def build(smiles, halide, proto, expansion, xyz=None):
    """Place six copies of the molecule; return (structure, clash-free, closest contacts,
    zwitterion SMILES)."""
    lattice0, framework, sites = proto
    lattice = Lattice(lattice0.matrix * LATTICE_SCALE[halide] * expansion)
    mol, s_idx, c_idx, n_idx = zwitterion(smiles)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    kinds = np.array([KINDS.index(kind(x)) for x in symbols])
    no_fw = np.arange(len(symbols)) == s_idx  # the thiolate S is bonded to its Pb neighbours
    xyz = conformers(mol) if xyz is None else xyz
    fw_sym = [halide if x == "Cl" else x for x, _ in framework]
    fw_cart = lattice.get_cartesian_coords(np.array([f for _, f in framework]))
    fw_env = Environment(lattice, fw_sym, fw_cart)
    site_poses = [poses(xyz, s_idx, c_idx, n_idx, site, lattice) for site in sites]
    fw_soft = np.array([fw_env.penalty(kinds, sp, no_fw, SOFT_MARGIN) for sp in site_poses])
    n_sites = len(sites)

    def mol_soft(k, cart, placed):
        others = np.array([placed[m] for m in range(n_sites) if m != k])
        return Environment(lattice, symbols * (n_sites - 1), others).penalty(kinds, cart, margin=SOFT_MARGIN)

    # symmetric start: the same pose index on every site reproduces the prototype's
    # packing; score the poses that sit best in the framework on the whole cell
    best, best_score = None, np.inf
    for p in np.argsort(fw_soft.sum(axis=0))[:N_SYMMETRIC]:
        placed = [site_poses[k][p] for k in range(n_sites)]
        score = fw_soft[:, p].sum() + sum(mol_soft(k, placed[k], placed) for k in range(n_sites))
        if score < best_score:
            best, best_score = placed, score
    placed = best
    for sweep in range(3):  # then re-optimise each site against the other five
        for k in range(n_sites):
            placed[k] = site_poses[k][int(np.argmin(fw_soft[k] + mol_soft(k, site_poses[k], placed)))]
    clash, closest = False, {}
    for k in range(len(sites)):
        others = np.array([placed[m] for m in range(len(sites)) if m != k])
        env = Environment(lattice, symbols * (len(sites) - 1), others)
        clash |= (fw_env.penalty(kinds, placed[k], no_fw) + env.penalty(kinds, placed[k])) > 1e-9
        for c in (fw_env.closest(kinds, placed[k], no_fw), env.closest(kinds, placed[k])):
            for pair, d in c.items():
                closest[pair] = min(closest.get(pair, np.inf), d)
    st = Structure(lattice, fw_sym + symbols * len(sites), np.vstack([fw_cart] + placed),
                   coords_are_cartesian=True)
    return st, not clash, closest, Chem.MolToSmiles(Chem.RemoveHs(mol)), xyz


def build_one(job):
    """One (molecule, halide): expand until clash-free or max_expansion. Returns the
    manifest row and the structure."""
    p, halide, proto, max_expansion = job
    expansion, xyz = 1.0, None
    while True:  # the conformers are reused across expansion steps
        st, ok, closest, zw_smiles, xyz = build(p["SMILES"], halide, proto, expansion, xyz)
        if ok or expansion >= max_expansion - 1e-9:
            break
        expansion = round(expansion + EXPANSION_STEP, 4)
    name = f"zw-{p['CID']}-Pb{halide}"
    row = {"name": name, "cif": f"structures/{name}.cif", "cid": p["CID"], "smiles": p["SMILES"],
           "zwitterion_smiles": zw_smiles, "iupac": p["IUPACName"], "halide": halide,
           "cation_class": p["cation_class"], "bonds_N_S": p["bonds_N_S"],
           "heavy_atoms": int(p["HeavyAtomCount"]), "formula": st.composition.reduced_formula,
           "n_atoms": len(st), "expansion": expansion, "clash_free": ok,
           **{f"min_{a}_{b}": round(float(d), 3) for (a, b), d in sorted(closest.items())}}
    return row, st


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--picks", type=Path, default=PROJECT_ROOT / "pubchem_zwitterion_picks.csv",
                        help="Molecules (.csv or .parquet): CID, SMILES, IUPACName, HeavyAtomCount, "
                             "cation_class, bonds_N_S")
    parser.add_argument("--prototype", type=Path, default=PROTOTYPE)
    parser.add_argument("--halides", nargs="+", default=["Cl", "Br", "I"], choices=sorted(LATTICE_SCALE))
    parser.add_argument("--cids", nargs="*", type=int, help="Only these CIDs (default: all picks)")
    parser.add_argument("--max-expansion", type=float, default=MAX_EXPANSION)
    parser.add_argument("--screen", action="store_true",
                        help="Only write <out>/screen.csv (expansion per molecule), no structures")
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes")
    parser.add_argument("--out", type=Path, default=ZW_DIR / "zwitterion_picks")
    args = parser.parse_args()

    picks = pd.read_parquet(args.picks) if args.picks.suffix == ".parquet" else pd.read_csv(args.picks)
    if args.cids:
        picks = picks[picks["CID"].isin(args.cids)]
    proto = prototype_sites(args.prototype)
    jobs = [(p, h, proto, args.max_expansion) for _, p in picks.iterrows() for h in args.halides]
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.screen:
        (args.out / "structures").mkdir(exist_ok=True)
    rows = []
    with multiprocessing.Pool(args.jobs) as pool:
        for i, (row, st) in enumerate(pool.imap(build_one, jobs, chunksize=1), 1):
            if not args.screen:
                st.to(filename=str(args.out / row["cif"]))
            rows.append(row)
            print(f"[{i}/{len(jobs)}] {row['name']:22s} {row['formula']:18s} {row['n_atoms']:4d} atoms  "
                  f"expansion {row['expansion']:.2f}{'' if row['clash_free'] else '  STILL CLASHES'}", flush=True)
    man = pd.DataFrame(rows)
    if args.screen:
        man.drop(columns=["cif"]).to_csv(args.out / "screen.csv", index=False)
        print(f"\n{len(man)} screened -> {args.out / 'screen.csv'}; clash-free at 1.00: "
              f"{(man['clash_free'] & (man['expansion'] == 1)).sum()}, "
              f"at <= {args.max_expansion}: {man['clash_free'].sum()}")
        return
    man.to_csv(args.out / "manifest.csv", index=False)
    (args.out / "mlip_structures.txt").write_text("".join(c + "\n" for c in man["cif"]))
    print(f"\n{len(man)} structures -> {args.out}; expansion > 1 for {(man['expansion'] > 1).sum()}, "
          f"still clashing {(~man['clash_free']).sum()}")


if __name__ == "__main__":
    main()
