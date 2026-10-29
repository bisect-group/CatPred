"""Structure choice (experimental chain, AlphaFold, ESMFold2, ESM3) and the ProGRES vector cache."""

import json
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
import pytest
import torch

from emulator_bench import cache_structures
from emulator_bench.esm_cache import sequence_key
from emulator_bench.structures import ESM3_PREFIX, StructureResolver, extract_chain, resolve_sequences, sequence_candidates


def write_cif(path: Path) -> None:
    """Two chains: A has residues 1-2 (residue 2 has an insertion-code twin), B has residue 1."""
    atoms = struc.AtomArray(7)
    atoms.chain_id[:] = ["A", "A", "A", "A", "A", "B", "B"]
    atoms.res_id[:] = [1, 1, 2, 2, 2, 1, 1]
    atoms.ins_code[:] = ["", "", "", "", "A", "", ""]
    atoms.res_name[:] = ["ALA", "ALA", "GLY", "GLY", "GLY", "SER", "SER"]
    atoms.atom_name[:] = ["N", "CA", "N", "CA", "C", "N", "CA"]
    atoms.element[:] = ["N", "C", "N", "C", "C", "N", "C"]
    atoms.coord = np.arange(21, dtype=np.float32).reshape(7, 3)
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms, data_block="1ABC")
    cif.write(path)


@pytest.fixture
def dirs(tmp_path):
    made = {name: tmp_path / name for name in ("mmcif", "af", "esmfold2", "esm3")}
    for folder in made.values():
        folder.mkdir()
    return made


def make_resolver(tmp_path, dirs):
    return StructureResolver.from_dirs(dirs["mmcif"], dirs["af"], dirs["esmfold2"], dirs["esm3"], tmp_path / "chains")


def test_extract_chain_keeps_only_the_chain_and_numbers_residues(tmp_path):
    write_cif(tmp_path / "1ABC_updated.cif")
    extract_chain(tmp_path / "1ABC_updated.cif", "A", tmp_path / "1ABC_A.pdb")
    atoms = [line for line in (tmp_path / "1ABC_A.pdb").read_text().splitlines() if line.startswith("ATOM")]
    assert len(atoms) == 5
    assert {line[21] for line in atoms} == {"A"}
    assert [int(line[22:26]) for line in atoms] == [1, 1, 2, 2, 3]
    assert [line[13:16].strip() for line in atoms] == ["N", "CA", "N", "CA", "C"]
    assert [float(line[30:38]) for line in atoms][:2] == [0.0, 3.0]


def test_resolver_prefers_best_covering_experimental_chain(tmp_path, dirs):
    write_cif(dirs["mmcif"] / "1ABC_updated.cif")
    (dirs["af"] / "AF-P12345-F1-model_v6.pdb").write_text("")
    resolver = make_resolver(tmp_path, dirs)
    candidates = [
        {"pdbs": "9ZZZ:P12345", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.99},  # no mmCIF file
        {"pdbs": "1ABC:P12345", "pdb_type": "experimental", "pdb_chain": "B", "pdb_coverage": 0.6},
        {"pdbs": "1abc:P12345", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.9},
        {"pdbs": "1ABC:P12345", "pdb_type": "predicted", "pdb_chain": "A", "pdb_coverage": 1.0},  # not experimental
    ]
    result = resolver.resolve(["P12345"], candidates)
    assert (result["source"], result["structure_id"]) == ("experimental", "1ABC_A")
    assert Path(result["structure_path"]) == tmp_path / "chains" / "1ABC_A.pdb"
    resolver.extract_pending(workers=1)
    assert Path(result["structure_path"]).exists()


def test_resolver_falls_back_to_alphafold_then_esmfold2_then_esm3(tmp_path, dirs):
    (dirs["af"] / "AF-P12345-F1-model_v1.pdb").write_text("")
    (dirs["af"] / "AF-P12345-F1-model_v6.pdb").write_text("")
    for name in ("ESMFold2-P12345.pdb", "ESMFold2-Q99999|A1B.pdb", "ESM3-open-small-P12345.pdb", "ESM3-open-small-Q99999|A1B.pdb",
                 "ESM3-open-small-R11111.pdb", "ESM3-open-small-Q22222|C3D.pdb"):
        (dirs["esmfold2" if name.startswith("ESMFold2") else "esm3"] / name).write_text("")
    resolver = make_resolver(tmp_path, dirs)
    experimental_only = [{"pdbs": "1ABC:P12345", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.9}]  # no mmCIF file
    result = resolver.resolve(["P12345|X1Y"], experimental_only)
    assert result["source"] == "alphafold" and result["structure_path"].endswith("AF-P12345-F1-model_v6.pdb")
    result = resolver.resolve(["Q99999|A1B"], [])
    assert (result["source"], result["structure_id"]) == ("esmfold2", "Q99999|A1B")
    assert resolver.resolve(["R11111"], [])["source"] == "esm3"
    result = resolver.resolve(["Q22222|C3D"], [])
    assert (result["source"], result["structure_id"]) == ("esm3", "Q22222|C3D")
    assert resolver.resolve(["Z00000"], [])["source"] == "none"


def test_resolve_sequences_uses_the_candidate_rows_of_each_sequence(tmp_path, dirs):
    write_cif(dirs["mmcif"] / "1ABC_updated.cif")
    (dirs["af"] / "AF-P2-F1-model_v6.pdb").write_text("")
    row = {"value_type": "kcat", "pdbs": None, "pdb_type": None, "pdb_chain": None, "pdb_coverage": None}
    frame = pd.DataFrame([
        {**row, "sequence": "AG", "uniprot_id": "P1", "pdbs": "1ABC:P1", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.9},
        {**row, "sequence": "AG", "uniprot_id": "P1", "pdbs": "1ABC:P1", "pdb_type": "experimental", "pdb_chain": "B", "pdb_coverage": 0.5},
        {**row, "sequence": "GG", "uniprot_id": "P2"},
        {**row, "sequence": "SS", "uniprot_id": "P3"},
        {**row, "sequence": "TT", "uniprot_id": "P4"},  # not asked for
    ])
    data = tmp_path / "data"
    (data / "kcat").mkdir(parents=True)
    frame.to_parquet(data / "kcat" / "kcat_kinetic_params_3d.parquet")
    resolver = make_resolver(tmp_path, dirs)
    table = resolve_sequences(resolver, sequence_candidates(data, ["kcat"], ["AG", "GG", "SS"]), ["AG", "GG", "SS"], workers=1)
    assert table["source"].tolist() == ["experimental", "alphafold", "none"]
    assert Path(table["structure_path"][0]).exists()
    vectors = {sequence_key("AG"): torch.zeros(128), sequence_key("GG"): torch.zeros(128)}
    coverage = cache_structures.coverage_table(table, vectors, {"kcat": {"AG", "GG", "SS"}})
    assert coverage.iloc[0].to_dict() == {"task": "kcat", "sequences": 3, "experimental": 1, "alphafold": 1, "esmfold2": 0, "esm3": 0, "none": 1}


def helix(path: Path, n: int, seed: int) -> None:
    """A CA-only chain: a helix with noise, enough for progres to embed."""
    t = np.arange(n)
    xyz = np.stack([2.3 * np.cos(1.745 * t), 2.3 * np.sin(1.745 * t), 1.5 * t], axis=1) + np.random.default_rng(seed).normal(0, 0.2, (n, 3))
    lines = [f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00" for i, (x, y, z) in enumerate(xyz)]
    path.write_text("\n".join(lines) + "\nEND\n")


PINNED_PROGRES = Path("/home/da24s023/EMULaToR/outputs/baselines/CatPred/embeddings/progres")


def test_esm3_prefix_matches_the_real_file_names(tmp_path):
    (tmp_path / "ESM3-open-small-P1.pdb").write_text("")
    assert ESM3_PREFIX == "ESM3-open-small-"
    assert list(StructureResolver.from_dirs(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path).esm3) == ["P1"]


def test_progres_vectors_equal_the_package_embed_call_and_the_cache_resumes(tmp_path, work, monkeypatch):
    if PINNED_PROGRES.exists():  # the pinned copy is sha-checked below, so the wheel and weights are not downloaded again
        monkeypatch.setenv("CATPRED_PROGRES_PATH", str(PINNED_PROGRES))
    from emulator_bench.progres_shim import PROGRES_VERSION, WEIGHTS_SHA256, ensure_progres, vendor_path

    progres = ensure_progres(download=True)
    assert Path(progres.__file__).resolve().parent == vendor_path() / "progres"
    assert (vendor_path() / "progres-0.2.7.dist-info").exists() and PROGRES_VERSION == "0.2.7"
    import hashlib

    assert hashlib.sha256((vendor_path() / "data/trained_models/v_0_2_0/trained_model.pt").read_bytes()).hexdigest() == WEIGHTS_SHA256

    sequences = ["A" * 40, "A" * 55, "A" * 70, "A" * 31]
    paths = []
    for i, seq in enumerate(sequences[:3]):
        paths.append(tmp_path / f"s{i}.pdb")
        helix(paths[-1], len(seq), i)
    table = pd.DataFrame({"sequence": sequences, "source": ["esm3"] * 3 + ["none"], "structure_path": [*map(str, paths), ""]})
    cache = tmp_path / "vectors.jsonl"
    vectors = cache_structures.embed_structures(table, cache)
    assert set(vectors) == {sequence_key(s) for s in sequences[:3]}  # the sequence without a structure is left out
    for seq, path in zip(sequences, paths):
        own = progres.embed_structure(str(path), device="cpu")  # the package call with its own model load
        assert vectors[sequence_key(seq)].shape == (128,)
        assert torch.equal(vectors[sequence_key(seq)], own.float())
    assert len(cache.read_text().splitlines()) == 3
    assert cache_structures.embed_structures(table, cache).keys() == vectors.keys() and len(cache.read_text().splitlines()) == 3
    helix(paths[0], 40, 99)  # another structure for the same sequence is embedded again
    changed = cache_structures.embed_structures(table, cache)
    assert len(cache.read_text().splitlines()) == 4
    assert not torch.equal(changed[sequence_key(sequences[0])], vectors[sequence_key(sequences[0])])
    assert json.loads(cache.read_text().splitlines()[0])["fp"] != json.loads(cache.read_text().splitlines()[3])["fp"]
    with open(cache, "a") as out:
        out.write('{"key": "abc", "fp": "d')  # a write cut short by a stopped job
    assert cache_structures.embed_structures(table, cache).keys() == vectors.keys()
