"""Generate Quantum ESPRESSO SCF + DFPT phonon inputs for structures in data/structures.

Renders two Jinja2 templates per structure (templates/qe/):
- pw_scf.in.j2:  a pw.x self-consistent-field calculation, which must be run
  first to produce the wavefunctions/charge density the phonon step reads.
- ph_dfpt.in.j2: a ph.x density-functional-perturbation-theory phonon
  calculation (dispersion over a q-point grid, via `ldisp`) that reads that
  SCF run by matching `prefix`/`outdir`.

Pseudopotentials are never guessed: QEInputConfig.pseudopotentials must map
every element in the structure to a real UPF filename, since a wrong guess
would produce an input file that silently fails (or worse, quietly uses the
wrong functional) rather than erroring here.
"""
from chemistry import Ion, is_open_shell, expected_unpaired_electrons
import pseudopotentials

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import ase.data
import ase.io
import jinja2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates" / "qe"
STRUCTURES_DIR = PROJECT_ROOT / "data" / "cubic_structures"
QE_INPUTS_DIR = PROJECT_ROOT / "data" / "qe_inputs"

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


@dataclass
class QEInputConfig:
    """Tunable parameters for a Quantum ESPRESSO SCF + DFPT phonon input pair."""

    # Required: element symbol -> UPF pseudopotential filename (must exist in pseudo_dir).
    pseudopotentials: dict[str, str]

    pseudo_dir: str = "./pseudo"
    outdir: str = "./out"

    # Basis-set (plane-wave) cutoffs, in Ry.
    ecutwfc: float = 60.0
    ecutrho: float = 480.0

    occupations: str = "fixed"
    smearing: str = "gaussian"
    degauss: float = 0.01
    conv_thr: float = 1.0e-10
    mixing_beta: float = 0.7

    kpts: tuple[int, int, int] = (4, 4, 4)
    koffset: tuple[int, int, int] = (0, 0, 0)

    # Real-space replication of the structure before writing it out, e.g. (2, 2, 2)
    # for a supercell. (1, 1, 1) leaves the structure as read from data/structures.
    # Ought to stay (1, 1, 1) as the idea behind DFPT is "no supercell"
    supercell: tuple[int, int, int] = (1, 1, 1)
    qpts: tuple[int, int, int] = (4, 4, 4)
    tr2_ph: float = 1.0e-14

    # Born effective charges + dielectric tensor, for LO-TO splitting. Halide
    # perovskites are polar/ionic, so this normally should stay on.
    epsil: bool = True
    zeu: bool = True

    magnetic_moments: dict[str, float] = field(default_factory=dict)

    @property
    def nspin(self) -> int:
        return 2 if self.magnetic_moments else 1


def _species_table(atoms: ase.Atoms, config: QEInputConfig | None = None) -> list[dict]:
    seen = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in seen:
            seen.append(symbol)
    table = []
    for symbol in seen:
        entry = {"symbol": symbol, "mass": ase.data.atomic_masses[ase.data.atomic_numbers[symbol]]}
        if config is not None and symbol in config.magnetic_moments:
            entry["starting_magnetization"] = config.magnetic_moments[symbol]
        table.append(entry)
    return table


def _prepare_structure(atoms: ase.Atoms, config: QEInputConfig) -> ase.Atoms:
    if config.supercell != (1, 1, 1):
        atoms = atoms.repeat(config.supercell)
    return atoms


def render_pw_scf_input(atoms: ase.Atoms, config: QEInputConfig, prefix: str) -> str:
    """Render the pw.x SCF input for a structure."""
    atoms = _prepare_structure(atoms, config)
    species = _species_table(atoms, config)

    missing = [sp["symbol"] for sp in species if sp["symbol"] not in config.pseudopotentials]
    if missing:
        raise ValueError(f"No pseudopotential specified for: {missing}")
    for sp in species:
        sp["pseudopotential"] = config.pseudopotentials[sp["symbol"]]

    sites = [
        {"symbol": symbol, "pos": pos}
        for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions())
    ]

    template = _env.get_template("pw_scf.in.j2")
    return template.render(
        prefix=prefix,
        outdir=config.outdir,
        pseudo_dir=config.pseudo_dir,
        ecutwfc=config.ecutwfc,
        ecutrho=config.ecutrho,
        occupations=config.occupations,
        smearing=config.smearing,
        degauss=config.degauss,
        nspin=config.nspin,
        conv_thr=config.conv_thr,
        mixing_beta=config.mixing_beta,
        kpts=config.kpts,
        koffset=config.koffset,
        species=species,
        sites=sites,
        cell=atoms.cell[:],
    )


def render_ph_dfpt_input(atoms: ase.Atoms, config: QEInputConfig, prefix: str) -> str:
    """Render the ph.x DFPT phonon input for a structure (must follow a matching SCF run)."""
    atoms = _prepare_structure(atoms, config)
    species = _species_table(atoms)

    template = _env.get_template("ph_dfpt.in.j2")
    return template.render(
        prefix=prefix,
        outdir=config.outdir,
        fildyn=f"{prefix}.dyn",
        tr2_ph=config.tr2_ph,
        qpts=config.qpts,
        epsil=config.epsil,
        zeu=config.zeu,
        species=species,
    )


def pbesol_config_for(atoms: ase.Atoms, ions: dict[str, Ion] | None = None, **overrides) -> QEInputConfig:
    """Build a QEInputConfig from the local SSSP-style PBEsol set (data/pseudopotentials/).

    Cutoffs are the max of each element's individually recommended cutoff
    (data/pseudopotentials/cutoffs.json), since one plane-wave cutoff applies
    to the whole calculation. Any QEInputConfig field can be overridden via
    keyword, e.g. pbesol_config_for(atoms, kpts=(6, 6, 6)).
    """
    symbols = list(dict.fromkeys(atoms.get_chemical_symbols()))
    ecutwfc, ecutrho = pseudopotentials.suggested_cutoffs(symbols)
    fields = dict(
        pseudopotentials=pseudopotentials.pseudopotentials_for(symbols),
        pseudo_dir=str(pseudopotentials.LIBRARY_DIR),
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
    )
    if ions is not None:
        magnetic_moments = {}
        for ion in ions.values():
            if is_open_shell(ion.symbol, ion.oxidation_state):
                n_unpaired = expected_unpaired_electrons(ion.symbol, ion.oxidation_state)
                magnetic_moments[ion.symbol] = n_unpaired / pseudopotentials.valence_electrons(ion.symbol)
        if magnetic_moments:
            fields["magnetic_moments"] = magnetic_moments
            fields.setdefault("occupations", "smearing")  # redundant once you flip the global default
    fields.update(overrides)
    return QEInputConfig(**fields)


def generate_inputs_for_structure(
    structure_path: Path,
    config: QEInputConfig | Callable[[ase.Atoms], QEInputConfig],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Render and write the SCF + DFPT input pair for one structure file.

    `config` can be a QEInputConfig, or a callable (e.g. pbesol_config_for)
    that builds one from the structure's atoms.
    """
    atoms = ase.io.read(structure_path)
    if callable(config):
        config = config(atoms)
    prefix = structure_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    scf_path = out_dir / f"{prefix}.scf.in"
    ph_path = out_dir / f"{prefix}.ph.in"
    scf_path.write_text(render_pw_scf_input(atoms, config, prefix))
    ph_path.write_text(render_ph_dfpt_input(atoms, config, prefix))
    return scf_path, ph_path


def generate_inputs_for_all_structures(
    config: QEInputConfig | Callable[[ase.Atoms], QEInputConfig],
) -> list[tuple[Path, Path]]:
    """Generate SCF + DFPT inputs for every structure file in data/structures."""
    written = []
    for structure_path in sorted(STRUCTURES_DIR.iterdir()):
        if structure_path.suffix.lower() not in {".cif", ".vasp", ".xyz", ".extxyz"}:
            continue
        out_dir = QE_INPUTS_DIR / structure_path.stem
        written.append(generate_inputs_for_structure(structure_path, config, out_dir))
    return written


if __name__ == "__main__":
    for scf_path, ph_path in generate_inputs_for_all_structures(pbesol_config_for):
        print(f"wrote {scf_path} and {ph_path}")
