"""Result aggregation preserves missing, duplicate and conflicting observations."""
import copy
import csv
import json
from pathlib import Path

import pytest

from tools.collect_results import collect_results, main


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def run(tmp_path):
    path = tmp_path / "logs" / "预训练"
    write(path / "configs.json", {"dataset_name": "Synth_PEDES", "mode": "pretrain", "seed": 42,
        "num_epoch": 2, "precision": "fp32", "device": "cpu", "test_batch_size": 4,
        "use_nac": True, "use_a2dfp": True, "use_cdfp": True,
        "use_llm_augment": False, "use_generated": False})
    (path / "history.jsonl").write_text('{"epoch": 1}\n{"epoch": 2}\n', encoding="utf-8")
    # Deliberately not a torch checkpoint: collecting metadata must never load it.
    (path / "last.pth").write_bytes(b"not a checkpoint")
    return path


def result(run, **changes):
    return {"checkpoint": str(run / "last.pth"), "dataset": "RSTPReid", "source_dataset": "Synth_PEDES",
        "protocol": "direct_transfer", "source_epoch": 2, "source_training_mode": "pretrain", "seed": 42,
        "source_global_step": 8, "test_annotation_sha256": "test-annotation-hash",
        "split": "test", "evaluation_scope": "provided_test_split",
        "selection": {"training_rule": "fixed_schedule_final", "checkpoint_matches_selection": True},
        "evaluation": {"device": "cpu", "precision": "fp32", "batch_size": 2, "elapsed_seconds": 1.25},
        "metrics": {"R1": 40.0, "R5": 60.0, "R10": 80.0, "mAP": 35.0, "mINP": 30.0,
            "num_queries": 20, "num_gallery": 10, "num_valid_queries": 20, "num_ignored_queries": 0}, **changes}


def test_csv_json_preserve_recorded_protocol_metrics_and_source_training_metadata(run, tmp_path):
    source = write(run / "rstp_test.json", result(run))
    json_path, csv_path = tmp_path / "summary.json", tmp_path / "summary.csv"
    summary = collect_results(run.parent, json_path, csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == summary
    assert summary["report"]["evaluated"] == 1 and summary["report"]["pending"] == 0
    row, = summary["runs"]
    assert row["run_dir"] == str(run.resolve()) and row["result_file"] == str(source.resolve())
    assert row["source_dataset"] == "Synth_PEDES" and row["dataset"] == "RSTPReid"
    assert row["protocol"] == "direct_transfer" and row["mode"] == "pretrain"
    assert row["source_epoch"] == row["completed_epochs"] == row["planned_epochs"] == 2
    assert row["source_global_step"] == 8 and row["test_annotation_sha256"] == "test-annotation-hash"
    assert row["config_origin"] == "recorded"
    assert row["training_status"] == "completed" and row["elapsed_seconds"] == 1.25
    assert row["use_nac"] and row["use_a2dfp"] and row["use_cdfp"]
    assert row["R1"] == 40 and row["mAP"] == 35 and row["resources"] is None
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        csv_row, = list(csv.DictReader(stream))
    assert csv_row["run_dir"] == str(run.resolve()) and csv_row["R1"] == "40.0"
    assert json.loads(csv_row["selection"])["training_rule"] == "fixed_schedule_final"


def test_missing_evaluation_remains_pending_with_null_metrics_and_partial_training(run):
    (run / "history.jsonl").write_text('{"epoch": 1}\n{"epoch":', encoding="utf-8")
    summary = collect_results(run.parent)
    row, = summary["runs"]
    assert row["status"] == "pending" and row["training_status"] == "partial"
    assert row["dataset"] == "Synth_PEDES" and row["R1"] is None
    assert row["source_epoch"] is None and row["selection"] is None
    assert summary["report"]["pending"] == 1
    assert len(summary["report"]["warnings"]) == 1
    assert summary["training_runs"][0]["has_checkpoint"]


def test_ignores_smoke_subset_validation_and_non_evaluation_json(run):
    write(run / "stage3_step.json", {"subset_sanity_metrics_not_benchmark": result(run)["metrics"]})
    write(run / "marked_smoke.json", result(run, benchmark_training_completed=False))
    write(run / "subset.json", result(run, evaluation_scope="tiny_validation_subset"))
    write(run / "validation.json", result(run, split="val"))
    write(run / "other.json", [1, 2, 3])
    write(run / ".recovery" / "temporary_test.json", result(run))
    summary = collect_results(run.parent)
    assert summary["report"]["evaluated"] == 0 and summary["report"]["pending"] == 1
    assert summary["report"]["rejected_files"] == []


def test_duplicate_results_are_all_retained_and_conflicts_are_explicit(run):
    low = result(run)
    high = copy.deepcopy(low)
    high["metrics"]["R1"] = 99
    write(run / "first.json", low)
    write(run / "second.json", low)
    write(run / "third.json", high)
    summary = collect_results(run.parent)
    assert summary["report"]["evaluated"] == 3
    assert summary["report"]["duplicate_groups"] == summary["report"]["conflicting_duplicate_groups"] == 1
    assert sorted(row["R1"] for row in summary["runs"]) == [40, 40, 99]
    assert len({row["duplicate_group"] for row in summary["runs"]}) == 1
    assert all(row["duplicate_count"] == 3 and row["duplicate_conflict"] for row in summary["runs"])


@pytest.mark.parametrize("bad_metric", [{"R1": float("nan")}, {"R1": True}, {"R1": 101},
    {"num_valid_queries": 21}, {"num_queries": 0}, {"num_ignored_queries": 1}])
def test_invalid_metrics_are_rejected_without_becoming_results(run, bad_metric):
    payload = result(run)
    payload["metrics"].update(bad_metric)
    write(run / "invalid.json", payload)
    summary = collect_results(run.parent)
    assert summary["report"]["evaluated"] == 0 and summary["report"]["pending"] == 1
    assert len(summary["report"]["rejected_files"]) == 1


def make_plan(tmp_path, run, jobs):
    return write(tmp_path / "experiments.json", {"version": 1, "jobs": [
        {"id": "source", "kind": "train", "run_dir": str(run)}, *jobs]})


def test_plan_lists_missing_outputs_and_excludes_unplanned_test_scores(run, tmp_path):
    known = write(tmp_path / "evaluations" / "rstp.json", result(run))
    write(run / "unplanned_higher_score.json", result(run, metrics={**result(run)["metrics"], "R1": 100}))
    plan = make_plan(tmp_path, run, [
        {"id": "rstp", "kind": "evaluate", "dataset": "RSTPReid", "protocol": "direct_transfer",
            "run_dir": str(run.relative_to(tmp_path)), "output": "evaluations/rstp.json"},
        {"id": "icfg", "kind": "evaluate", "dataset": "ICFG-PEDES", "protocol": "direct_transfer",
            "run_dir": str(run), "output": "evaluations/icfg.json"},
    ])
    summary = collect_results(tmp_path, plan_path=plan)
    assert summary["report"]["evaluated"] == 1 and summary["report"]["pending"] == 1
    first, second = summary["runs"]
    assert first["job_id"] == "rstp" and first["result_file"] == str(known.resolve()) and first["R1"] == 40
    assert second["job_id"] == "icfg" and second["status"] == "pending" and second["R1"] is None
    assert second["dataset"] == "ICFG-PEDES" and second["completed_epochs"] == 2


@pytest.mark.parametrize("change", [{"dataset": "CUHK-PEDES"}, {"protocol": "in_domain"},
                                  {"is_smoke_test": True}, {"split": "val"},
                                  {"checkpoint": "other-run/last.pth"}])
def test_plan_reports_incompatible_or_smoke_outputs_as_invalid(run, tmp_path, change):
    output = write(tmp_path / "result.json", result(run, **change))
    plan = make_plan(tmp_path, run, [{"id": "rstp", "kind": "evaluate", "dataset": "RSTPReid",
        "protocol": "direct_transfer", "run_dir": str(run), "output": str(output)}])
    summary = collect_results(tmp_path, plan_path=plan)
    row, = summary["runs"]
    assert row["status"] == "invalid" and row["R1"] is None and row["error"]
    assert summary["report"]["invalid"] == 1


def test_cli_and_recollect_do_not_import_their_own_summaries(run, tmp_path, capsys):
    write(run / "eval.json", result(run))
    json_path, csv_path = run.parent / "summary.json", run.parent / "summary.csv"
    assert main(["--root", str(run.parent), "--output_json", str(json_path), "--output_csv", str(csv_path)]) == 0
    assert "1 evaluated, 0 pending" in capsys.readouterr().out
    assert collect_results(run.parent, json_path, csv_path)["report"]["evaluated"] == 1


def test_plan_cannot_assign_one_output_to_multiple_jobs(run, tmp_path):
    job = {"id": "one", "kind": "evaluate", "dataset": "RSTPReid", "output": "same.json"}
    plan = make_plan(tmp_path, run, [job, {**job, "id": "two"}])
    with pytest.raises(ValueError, match="distinct output"):
        collect_results(tmp_path, plan_path=plan)


def test_plan_records_not_started_training_even_before_config_creation(tmp_path):
    plan = write(tmp_path / "experiments.json", {"version": 1, "jobs": [
        {"id": "future", "kind": "train", "run_dir": "future", "config": {
            "dataset_name": "RSTPReid", "num_epoch": 30, "seed": 7, "use_cdfp": True}},
        {"id": "future_test", "kind": "evaluate", "run_dir": "future", "dataset": "RSTPReid",
         "protocol": "in_domain", "output": "future/test.json"}]})
    summary = collect_results(tmp_path, plan_path=plan)
    assert summary["report"]["training_not_started"] == 1 and summary["report"]["pending"] == 1
    assert summary["training_runs"][0]["training_status"] == "not_started"
    row, = summary["runs"]
    assert row["planned_epochs"] == 30 and row["R1"] is None
    assert row["seed"] == 7 and row["use_cdfp"] and row["config_origin"] == "planned"
