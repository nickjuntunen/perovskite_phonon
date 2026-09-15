"""Cluster the enumerated ABX3 compositions by chemical descriptors (ionic
radii, tolerance/octahedral factors, electronegativity, mass, oxidation
state), and check whether structures that score well against a vDOS target
(scripts/score_vdos.py's output) fall into one shared chemical cluster or
several distinct ones. The latter is the "multiple synthetic routes to the
same target" redundancy this project is looking for; the former means this
target currently has one dominant chemical recipe rather than several.

Descriptors are built directly from generate_inputs.candidate_ions(), so the
composition list here always matches whatever generate_inputs.py actually
built structures for -- no separate name parsing of e.g. "CsPbI3" back into
elements, which would be ambiguous for some two-letter symbols.

Usage:
    # cluster on chemistry alone
    uv run python3 scripts/cluster_compositions.py --n-clusters 10

    # also report which cluster(s) the best vDOS-fitness structures fall into
    uv run python3 scripts/cluster_compositions.py --n-clusters 10 \\
        --scores-csv data/vdos_scores.csv --top-n 15
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import ase.data
import numpy as np
from mendeleev import element as mendeleev_element
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from generate_inputs import candidate_ions
from paths import DATA_ROOT

DESCRIPTOR_NAMES = [
    "r_A", "r_B", "r_X", "tolerance_factor", "octahedral_factor",
    "mass_A", "mass_B", "mass_X",
    "en_A", "en_B", "en_X",
    "oxidation_A", "oxidation_B", "oxidation_X",
]

_EN_CACHE: dict[str, float | None] = {}


def _electronegativity(symbol: str) -> float | None:
    """Pauling electronegativity via mendeleev, tolerant of API differences
    across mendeleev versions (attribute in some, method in others)."""
    if symbol not in _EN_CACHE:
        el = mendeleev_element(symbol)
        value = getattr(el, "en_pauling", None)
        if value is None:
            try:
                value = el.electronegativity(scale="pauling")
            except Exception:
                value = None
        _EN_CACHE[symbol] = value
    return _EN_CACHE[symbol]


def build_composition_table() -> dict[str, dict]:
    """name -> {"A": Ion, "B": Ion, "X": Ion} for every charge-balanced
    monatomic ABX3 combination -- the same enumeration
    generate_inputs.generate_all() builds structures/QE inputs for."""
    a_ions, b_ions, x_ions = candidate_ions()
    table = {}
    for a, a_ion in a_ions.items():
        for b, b_ion in b_ions.items():
            for x, x_ion in x_ions.items():
                if a_ion.oxidation_state + b_ion.oxidation_state + 3 * x_ion.oxidation_state != 0:
                    continue
                table[f"{a}{b}{x}3"] = {"A": a_ion, "B": b_ion, "X": x_ion}
    return table


def descriptor_vector(ions: dict) -> np.ndarray | None:
    """Chemical feature vector for one (A, B, X) composition, or None if any
    ion is missing an ionic radius or electronegativity -- generate_inputs.py
    would have skipped building that composition's structure anyway."""
    radius = {site: ions[site].ionic_radius_angstrom for site in "ABX"}
    if any(v is None for v in radius.values()):
        return None

    tolerance_factor = (radius["A"] + radius["X"]) / (2 ** 0.5 * (radius["B"] + radius["X"]))
    octahedral_factor = radius["B"] / radius["X"]

    mass = {site: ase.data.atomic_masses[ase.data.atomic_numbers[ions[site].symbol]] for site in "ABX"}
    en = {site: _electronegativity(ions[site].symbol) for site in "ABX"}
    if any(v is None for v in en.values()):
        return None

    return np.array([
        radius["A"], radius["B"], radius["X"], tolerance_factor, octahedral_factor,
        mass["A"], mass["B"], mass["X"],
        en["A"], en["B"], en["X"],
        ions["A"].oxidation_state, ions["B"].oxidation_state, ions["X"].oxidation_state,
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument(
        "--scores-csv", type=Path, default=None,
        help="scripts/score_vdos.py's output CSV, to cross-reference clusters against vDOS-target fitness",
    )
    parser.add_argument("--top-n", type=int, default=15, help="How many top-fitness structures to report cluster membership for")
    parser.add_argument("--out-csv", type=Path, default=DATA_ROOT / "composition_clusters.csv")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    table = build_composition_table()
    names, vectors, skipped = [], [], []
    for name, ions in sorted(table.items()):
        vec = descriptor_vector(ions)
        if vec is None:
            skipped.append(name)
            continue
        names.append(name)
        vectors.append(vec)

    if not names:
        print("No compositions had a full descriptor vector -- nothing to cluster.")
        return

    X = StandardScaler().fit_transform(np.array(vectors))
    labels = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=args.random_state).fit_predict(X)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "cluster"] + DESCRIPTOR_NAMES)
        for name, label, vec in zip(names, labels, vectors):
            writer.writerow([name, label] + list(vec))

    print(f"Clustered {len(names)} composition(s) into {args.n_clusters} groups -> {args.out_csv}")
    if skipped:
        preview = skipped[:10]
        print(f"Skipped {len(skipped)} composition(s) missing a radius/electronegativity: {preview}{'...' if len(skipped) > 10 else ''}")

    if args.scores_csv is None:
        return

    fitness_by_name = {}
    with open(args.scores_csv) as f:
        for row in csv.DictReader(f):
            base_name = row["name"].removesuffix("_tilted")
            try:
                fitness_by_name[base_name] = float(row["fitness"])
            except ValueError:
                continue  # "inf" or missing -- not scoreable, excluded below

    cluster_of = dict(zip(names, labels))
    scored = [
        (name, fitness_by_name[name], cluster_of[name])
        for name in names
        if name in fitness_by_name and fitness_by_name[name] != float("inf")
    ]
    scored.sort(key=lambda t: t[1])
    top = scored[: args.top_n]

    clusters_hit: dict[int, list[tuple[str, float]]] = {}
    for name, fitness, cluster in top:
        clusters_hit.setdefault(cluster, []).append((name, fitness))

    print(f"\nTop {len(top)} structures by vDOS-target fitness fall into {len(clusters_hit)} distinct chemical cluster(s):")
    for cluster, members in sorted(clusters_hit.items(), key=lambda kv: min(f for _, f in kv[1])):
        print(f"  cluster {cluster}:")
        for name, fitness in sorted(members, key=lambda t: t[1]):
            print(f"    {name:>15}  fitness={fitness:.2f}")

    if len(clusters_hit) > 1:
        print(
            "\n-> Best-fitting structures span multiple chemical clusters: several "
            "chemically distinct recipes reach a similar vDOS (redundant synthetic routes)."
        )
    else:
        print(
            "\n-> Best-fitting structures all fall in one chemical cluster: this target "
            "currently has one dominant chemical recipe, not several redundant ones."
        )


if __name__ == "__main__":
    main()
