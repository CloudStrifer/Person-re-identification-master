"""Experiment protocols and individually resumed jobs, without model training."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import experiments
from utils.options import get_args


@pytest.fixture
def paths(tmp_path):
    manifest = tmp_path / "both.json"
    manifest.write_text('{"version":1,"records":[]}', encoding="utf-8")
    return {"root_dir": str(tmp_path / "data"), "bpe_path": str(tmp_path / "vocab.gz"),
            "clip_weights": str(tmp_path / "clip.pt"),
            "augmentation_manifests": {name: {"both": str(manifest)} for name in
                ("Synth_PEDES", "RSTPReid", "CUHK-PEDES", "ICFG-PEDES")}}


def plan_for(tmp_path, paths, groups):
    directory = tmp_path / "experiments"
    plan = experiments.build_plan(experiments.PROJECT / "configs/experiments/paper.json", paths,
        directory, groups=groups, overrides={"device": "cpu", "num_workers": 0,
                                          "num_epoch": 1, "warmup_epochs": 0})
    assert all("\\" not in job["run_dir"] and "\\" not in job.get("output", "") for job in plan["jobs"])
    output = directory / "experiments.json"
    experiments.write_plan(plan, output)
    return experiments.load_plan(output)


def job_named(plan, identity):
    return next(job for job in plan["jobs"] if job["id"] == "seed42-" + identity)


def run_directory(plan, job):
    return Path(plan["_base_dir"]) / job["run_dir"]


def mark_complete(plan, job):
    directory = run_directory(plan, job)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("last.pth", "best.pth"):
        (directory / name).write_bytes(b"explicit fake checkpoint; never loaded")
    state = {"status": "completed", "experiment_id": job["id"],
             "last_checkpoint": str(directory / "last.pth"),
             "selected_checkpoint": str(directory / "best.pth")}
    (directory / "experiment_result.json").write_text(json.dumps(state), encoding="utf-8")
    return directory


def ready_resources(args):
    return {"ready": True, "resources": {}, "missing_resources": [], "dependencies": {"torch": True}}


def test_six_module_rows_keep_paper_switches_and_common_identity_sampling(tmp_path, paths):
    plan = plan_for(tmp_path, paths, ["module_ablation"])
    rows = [job for job in plan["jobs"] if job["kind"] == "train" and job["group"] == "module_ablation"]
    assert len(rows) == 6
    expected = {"baseline": (False, False, False, False, False),
                "dual": (False, False, False, True, True),
                "dual_nac": (True, False, False, True, True),
                "dual_nac_a2dfp": (True, True, False, True, True),
                "dual_nac_cdfp": (True, False, True, True, True),
                "full": (True, True, True, True, True)}
    keys = ("use_nac", "use_a2dfp", "use_cdfp", "use_llm_augment", "use_generated")
    for job in rows:
        config = job["config"]
        suffix = job["id"].removeprefix("seed42-rstp-")
        assert tuple(config[key] for key in keys) == expected[suffix]
        assert config["sampler"] == "identity" and config["num_instance"] == 4
        assert config["batch_size"] == 64
        assert job["initialize_from"] == "seed42-synth-baseline"
        assert job["depends_on"] == [job["initialize_from"]]
        assert config["initialize_new_modules"] is True


def test_main_pretrain_and_direct_transfer_have_explicit_distinct_protocols(tmp_path, paths):
    plan = plan_for(tmp_path, paths, ["main"])
    source = job_named(plan, "synth-pretrain")
    config = source["config"]
    assert config["mode"] == "pretrain" and config["dataset_name"] == "Synth_PEDES"
    assert not config["use_nac"] and config["use_a2dfp"] and config["use_cdfp"]
    assert not config["use_llm_augment"] and not config["use_generated"]
    assert not source["depends_on"] and not source["initialize_from"]
    direct = [job for job in plan["jobs"] if job["protocol"] == "direct_transfer"]
    assert {job["dataset"] for job in direct} == {"RSTPReid", "CUHK-PEDES", "ICFG-PEDES"}
    for job in direct:
        assert job["kind"] == "evaluate" and job["checkpoint_from"] == source["id"]
        assert job["checkpoint_selection"] == "last"
        assert "finetune" not in job["config"]
    finetune = [job for job in plan["jobs"] if job["protocol"] == "finetune"]
    assert len(finetune) == 3
    for job in finetune:
        assert job["initialize_from"] == source["id"] and job["initialization_selection"] == "last"
        assert job["config"]["use_nac"] and job["config"]["initialize_new_modules"]


def test_missing_manifests_stay_pending_and_unfiltered_cannot_use_filtered_manifest(tmp_path, paths):
    missing = copy.deepcopy(paths)
    missing.pop("augmentation_manifests")
    plan = plan_for(tmp_path, missing, ["main"])
    target = job_named(plan, "finetune-rstpreid")
    assert target["config"]["use_llm_augment"] and target["config"]["use_generated"]
    assert target["requirements"] == ["augmentation_manifest"]
    report = experiments.check_job(plan, target["id"])
    assert not report["ready"] and "augmentation_manifest" in report["reason"]
    unfiltered = experiments.build_plan(experiments.PROJECT / "configs/experiments/paper.json", paths,
                                       tmp_path / "other", groups=["augmentation_ablation"])
    job = job_named(unfiltered, "rstp-dual_unfiltered")
    assert job["augmentation"] == "unfiltered" and job["requirements"] == ["augmentation_manifest"]
    for name in ("rstp-text_only", "rstp-image_only"):
        assert not job_named(unfiltered, name)["requirements"]


def test_source_last_is_not_accepted_until_its_job_is_completed(tmp_path, paths):
    plan = plan_for(tmp_path, paths, ["main"])
    source = job_named(plan, "synth-pretrain")
    direct = job_named(plan, "direct-rstpreid")
    target = job_named(plan, "finetune-rstpreid")
    directory = run_directory(plan, source)
    directory.mkdir(parents=True)
    (directory / "last.pth").write_bytes(b"interrupted source checkpoint")
    for job in (direct, target):
        with pytest.raises(FileNotFoundError, match="training job first"):
            experiments.resolved_config(plan, job)
    state = directory / "experiment_result.json"
    for invalid in ({"status": "running", "experiment_id": source["id"]},
                    {"status": "completed", "experiment_id": "different-job"}):
        state.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="incomplete"):
            experiments.resolved_config(plan, direct)
    mark_complete(plan, source)
    assert experiments.resolved_config(plan, direct)["checkpoint"] == str(directory / "last.pth")
    assert experiments.resolved_config(plan, target)["finetune"] == str(directory / "last.pth")


def test_plan_is_portable_idempotent_and_rejects_replacing_settings(tmp_path, paths, monkeypatch):
    plan = plan_for(tmp_path, paths, ["main"])
    output = Path(plan["_base_dir"]) / "experiments.json"
    assert "_base_dir" not in json.loads(output.read_text())
    before = output.read_bytes()
    experiments.write_plan(plan, output)
    assert output.read_bytes() == before
    changed = copy.deepcopy(plan)
    changed["jobs"][0]["config"]["lr"] *= 2
    with pytest.raises(ValueError, match="different settings"):
        experiments.write_plan(changed, output)
    assert output.read_bytes() == before
    for job in plan["jobs"]:
        assert not Path(job["run_dir"]).is_absolute()
        if job["kind"] == "evaluate":
            assert not Path(job["output"]).is_absolute()
    source = job_named(plan, "synth-pretrain")
    assert source["config"]["output_dir"] == source["run_dir"]
    mark_complete(plan, source)
    target = job_named(plan, "finetune-rstpreid")
    mark_complete(plan, target)
    moved = tmp_path / "moved"
    output.parent.rename(moved)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = experiments.load_plan(moved / "experiments.json")
    assert Path(loaded["_base_dir"]) == moved
    resolved = experiments.resolved_config(loaded, job_named(loaded, "synth-pretrain"))
    assert Path(resolved["output_dir"]) == moved / source["run_dir"]
    direct = experiments.resolved_config(loaded, job_named(loaded, "direct-rstpreid"))
    selected = experiments.resolved_config(loaded, job_named(loaded, "test-finetune-rstpreid"))
    assert Path(direct["checkpoint"]) == moved / source["run_dir"] / "last.pth"
    assert Path(selected["checkpoint"]) == moved / target["run_dir"] / "best.pth"
    assert Path(direct["checkpoint"]).is_file() and Path(selected["checkpoint"]).is_file()


def test_resume_does_not_require_original_clip_or_completed_source_job(tmp_path, paths):
    plan = plan_for(tmp_path, paths, ["module_ablation"])
    target = job_named(plan, "rstp-baseline")
    directory = run_directory(plan, target)
    directory.mkdir(parents=True)
    checkpoint = directory / "last.pth"
    checkpoint.write_bytes(b"explicit fake resume checkpoint; runner is mocked")
    assert not Path(paths["clip_weights"]).exists()
    assert not run_directory(plan, job_named(plan, "synth-baseline")).exists()
    resolved = experiments.resolved_config(plan, target, resume=True)
    assert resolved["resume"] == str(checkpoint)
    assert not {"finetune", "clip_weights", "initialize_new_modules"}.intersection(resolved)
    captured = []

    def inspect_resume(args):
        assert args.resume and args.resume_ckpt_file == str(checkpoint)
        assert not args.finetune and not args.clip_checkpoint
        captured.append(args)
        return ready_resources(args)

    def train(args, argv):
        assert args.resume and args.resume_ckpt_file == str(checkpoint)
        assert "--finetune" not in argv and "--clip_weights" not in argv
        return {"last_checkpoint": str(checkpoint), "selected_checkpoint": str(checkpoint)}

    with patch("tools.check_resources.inspect_resources", side_effect=inspect_resume), \
            patch("processor.baseline.run_training", side_effect=train) as training:
        assert experiments.check_job(plan, target["id"], resume=True)["ready"]
        result = experiments.run_job(plan, target["id"], resume=True)
        assert training.call_count == 1
    assert result["status"] == "completed" and len(captured) == 2
    saved = json.loads((directory / "experiment_result.json").read_text())
    assert saved["experiment_id"] == target["id"] and saved["status"] == "completed"
    with pytest.raises(FileExistsError):
        experiments.run_job(plan, target["id"], resume=True)


def test_store_true_flags_convert_to_bare_arguments():
    values = {"mode": "finetune", "finetune": "source.pth", "initialize_new_modules": True,
              "use_nac": True, "freeze_backbone": False}
    arguments = experiments.options_to_argv(values)
    position = arguments.index("--initialize_new_modules")
    assert position == len(arguments) - 1 or arguments[position + 1].startswith("--")
    args = get_args(arguments)
    assert args.initialize_new_modules and args.use_nac and not args.freeze_backbone
    assert experiments.options_to_argv({"initialize_new_modules": False}) == []


def test_sensitivity_default_points_reuse_the_same_fully_matched_run(tmp_path, paths):
    plan = plan_for(tmp_path, paths, ["module_ablation", "sensitivity"])
    reused = plan["reused_evaluations"]
    assert len(reused) == 4
    assert {row["evaluation_job"] for row in reused} == {"seed42-test-rstp-full"}
    assert {row["parameter"] for row in reused} == {
        "cdfp_margin", "a2dfp_blend_weight", "cdfp_k_negatives", "noise_threshold"}
    sensitivity_only = plan_for(tmp_path / "alone", paths, ["sensitivity"])
    assert len(sensitivity_only["reused_evaluations"]) == 3
    assert len({row["evaluation_job"] for row in sensitivity_only["reused_evaluations"]}) == 1


def test_runner_evaluation_only_calls_saved_feature_evaluation(tmp_path, paths):
    plan = plan_for(tmp_path, paths, ["main"])
    source = job_named(plan, "synth-pretrain")
    directory = mark_complete(plan, source)
    job = job_named(plan, "direct-rstpreid")
    with patch("tools.check_resources.inspect_resources", side_effect=ready_resources), \
            patch("processor.baseline.run_training", side_effect=AssertionError("evaluation cannot train")), \
            patch("processor.baseline.evaluate_checkpoint", return_value={"metrics": {"R1": 50.0}}) as evaluate:
        result = experiments.run_job(plan, job["id"])
    evaluate.assert_called_once_with(dataset_name="RSTPReid", root_dir=paths["root_dir"],
                                    bpe_path=paths["bpe_path"], device="cpu", batch_size=128,
                                    num_workers=0, checkpoint=str(directory / "last.pth"))
    assert result["status"] == "completed" and result["experiment_group"] == "main"
    assert (Path(plan["_base_dir"]) / job["output"]).is_file()
