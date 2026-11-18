"""Cluster the enumerated ABX3 compositions by chemical descriptors (ionic
radii, tolerance/octahedral factors, electronegativity, mass, oxidation
state), and check whether structures that score well against a vDOS target
(scripts/analysis/score_vdos.py's output) fall into one shared chemical cluster or
several distinct ones. The latter is the "multiple synthetic routes to the
same target" redundancy this project is looking for; the former means this
target currently has one dominant chemical recipe rather than several.

By default, descriptors are built directly from generate_inputs.candidate_ions(),
so the composition list matches whatever generate_inputs.py actually built
structures for -- no separate name parsing of e.g. "CsPbI3" back into
elements, which would be ambiguous for some two-letter symbols.

Pass --mp-metadata to cluster the real Materials Project perovskites instead
(data/mp_perovskites/metadata_stable.csv -- the output of
scripts/dataset/filter_stable_mp_perovskites.py: structurally-verified, near-hull
entries). Site assignments (a_sites/b_sites/x_site) and measured coordination
numbers (a_site_cn/b_site_cn) come straight from that file's structural
check; oxidation states are guessed from the bulk formula via pymatgen's
Composition.oxi_state_guesses() (fast, no per-structure bond-valence pass).
Sites with more than one species (common on the B-site, e.g. double
perovskites) get their per-element descriptor values averaged.

Usage:
    # cluster on chemistry alone
    uv run python3 scripts/analysis/cluster_compositions.py --n-clusters 10

    # also report which cluster(s) the best vDOS-fitness structures fall into
    uv run python3 scripts/analysis/cluster_compositions.py --n-clusters 10 \\
        --scores-csv data/vdos_scores.csv --top-n 15

    # cluster the surviving MP perovskites instead, with a PCA plot
    uv run python3 scripts/analysis/cluster_compositions.py --n-clusters 10 \\
        --mp-metadata data/mp_perovskites/metadata_stable.csv
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.paths import DATA_DIR
from src.utils.plotting import use_project_style
from src.analysis.composition_features import DESCRIPTOR_NAMES, descriptor_vector, mp_descriptor_table
from src.data.perovskite_builder import charge_balanced_abx3

use_project_style()

def plot_pca_clusters(X: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    """2D PCA projection of the (already standardized) descriptor matrix,
    colored by cluster assignment."""
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100

    n_clusters = int(labels.max()) + 1
    cmap = plt.colormaps["tab20" if n_clusters > 10 else "tab10"]
    colors = cmap(np.linspace(0, 1, max(n_clusters, 2)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for cluster in range(n_clusters):
        mask = labels == cluster
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=24, color=colors[cluster], edgecolors="none", alpha=0.8,
            label=f"cluster {cluster} (n={mask.sum()})",
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}\\% var.)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}\\% var.)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1 if n_clusters <= 15 else 2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument(
        "--scores-csv", type=Path, default=None,
        help="scripts/analysis/score_vdos.py's output CSV, to cross-reference clusters against vDOS-target fitness",
    )
    parser.add_argument("--top-n", type=int, default=15, help="How many top-fitness structures to report cluster membership for")
    parser.add_argument("--out-csv", type=Path, default=DATA_DIR / "composition_clusters.csv")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--mp-metadata", type=Path, default=None,
        help=(
            "Cluster real MP perovskites instead of the synthetic ABX3 enumeration -- pass "
            "data/mp_perovskites/metadata_stable.csv (or metadata_verified.csv)"
        ),
    )
    parser.add_argument(
        "--plot-path", type=Path, default=None,
        help="Where to save the 2-PC cluster scatter plot (default: alongside --out-csv, same stem + _pca.png)",
    )
    args = parser.parse_args()

    if args.mp_metadata is not None:
        names, vectors, skipped = mp_descriptor_table(args.mp_metadata)
        skip_reason = "missing a resolvable oxidation state, Shannon radius, or electronegativity"
        source_desc = f"MP perovskites ({args.mp_metadata.name})"
    else:
        table = charge_balanced_abx3()
        names, vectors, skipped = [], [], []
        for name, ions in sorted(table.items()):
            vec = descriptor_vector(ions)
            if vec is None:
                skipped.append(name)
                continue
            names.append(name)
            vectors.append(vec)
        skip_reason = "missing a radius/electronegativity"
        source_desc = "synthetic ABX3 compositions"

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

    print(f"Clustered {len(names)} {source_desc} into {args.n_clusters} groups -> {args.out_csv}")
    if skipped:
        preview = skipped[:10]
        print(f"Skipped {len(skipped)} composition(s) {skip_reason}: {preview}{'...' if len(skipped) > 10 else ''}")

    plot_path = args.plot_path or args.out_csv.with_name(args.out_csv.stem + "_pca.png")
    plot_pca_clusters(X, labels, plot_path, title=f"KMeans clusters ({source_desc}, k={args.n_clusters})")
    print(f"Wrote PCA cluster plot -> {plot_path}")

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
