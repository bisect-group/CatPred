from emulator_bench.cache_structures import dict_path
from emulator_bench.common import TASKS, discover_split_jobs


def test_ki_task_and_threshold_folders(tmp_path):
    assert "ki" in TASKS
    for sub in ("random_splits_grouped_sequence", "enzyme_sequence_splits/threshold_0.05", "conformer_cosine_splits/threshold_0.998"):
        (tmp_path / "ki" / sub).mkdir(parents=True)
        (tmp_path / "ki" / sub / "train.parquet").touch()
    jobs = discover_split_jobs(tmp_path, ["ki"])
    assert [(j.split_group, j.split_name) for j in jobs] == [
        ("random_splits_grouped_sequence", "all"), ("enzyme_sequence_splits", "threshold_0.05"), ("conformer_cosine_splits", "threshold_0.998")]


def test_dict_path_tag_keeps_the_default_name(tmp_path):
    assert dict_path(tmp_path).name == "catpred_progres_embeds_dict.pt"
    assert dict_path(tmp_path, "kcat_km_ki").name == "catpred_progres_embeds_dict_kcat_km_ki.pt"
