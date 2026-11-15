"""Collect recorded test results without loading weights or selecting by test score.

Use --plan for a prepared experiment batch: only its evaluation outputs are
included, and missing outputs remain pending. Without a plan, inspect a run
directory such as logs, not a repository full of temporary data.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path


_METRICS = ("R1", "R5", "R10", "mAP", "mINP")
_COUNTS = ("num_queries", "num_gallery", "num_valid_queries", "num_ignored_queries")
_FLAGS = ("use_nac", "use_a2dfp", "use_cdfp", "use_llm_augment", "use_generated")
_SKIP_DIRS = {".git", ".venv", ".recovery", "__pycache__", ".pytest_cache", "tests"}
_CSV_FIELDS = ("job_id", "status", "run_dir", "protocol", "dataset", "source_dataset",
    "checkpoint", "checkpoint_sha256", "result_file", "config_origin", "seed", "mode", *_FLAGS,
    "augmentation_manifest", "selection", "source_selection_rule", "initialization", "source_epoch",
    "source_global_step", "evaluation_scope", "test_annotation_sha256", "planned_epochs", "completed_epochs",
    "training_status", "precision", "device", "batch_size", *_COUNTS, *_METRICS,
    "elapsed_seconds", "resources", "duplicate_group", "duplicate_count", "duplicate_conflict", "error")


def _read(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _key(path):
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _resolve(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _json_files(root):
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in _SKIP_DIRS)
        for name in sorted(files):
            if name.lower().endswith(".json"):
                yield Path(directory) / name


def _training_run(directory, warnings, planned_config=None):
    directory = Path(directory).resolve()
    config_path = directory / "configs.json"
    recorded = config_path.is_file()
    config = _read(config_path) if recorded else dict(planned_config or {})
    if not isinstance(config, dict):
        raise ValueError("Training config must be an object: {}".format(config_path))
    history_path = directory / "history.jsonl"
    epochs = []
    if history_path.is_file():
        for number, line in enumerate(history_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                epoch = row["epoch"]
                if type(epoch) is not int or epoch < 1:
                    raise ValueError("epoch must be a positive integer")
            except (ValueError, TypeError, KeyError) as error:
                warnings.append({"path": str(history_path), "line": number, "reason": str(error)})
                continue
            epochs.append(epoch)
    completed = max(epochs, default=0)
    planned = config.get("num_epoch")
    if type(planned) is not int or planned < 1:
        planned = None
    status = ("completed" if completed >= planned else "partial" if completed else "not_started") if planned else (
        "unknown" if completed else "not_started")
    return {"run_dir": str(directory), "config": config, "planned_epochs": planned,
            "completed_epochs": completed, "training_status": status,
            "config_origin": "recorded" if recorded else "planned" if planned_config is not None else "unavailable",
            "has_checkpoint": (directory / "last.pth").is_file()}


def _is_smoke(payload):
    return (payload.get("benchmark_training_completed") is False
            or payload.get("smoke_test") is True or payload.get("is_smoke_test") is True
            or any(word in str(payload.get("evaluation_scope", "")).lower()
                   for word in ("smoke", "subset", "fixture")))


def _validate_result(payload):
    if not isinstance(payload, dict) or payload.get("split") != "test":
        return False
    if _is_smoke(payload):
        return False
    if any(not isinstance(payload.get(key), str) or not payload[key].strip()
           for key in ("checkpoint", "dataset")):
        raise ValueError("Test result needs nonempty checkpoint and dataset strings")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Test result needs a metrics object")
    for key in _METRICS:
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("{} must be a finite percentage in [0, 100]".format(key))
    for key in _COUNTS[:3]:
        if type(metrics.get(key)) is not int or metrics[key] < 1:
            raise ValueError("{} must be a positive integer".format(key))
    if metrics["num_valid_queries"] > metrics["num_queries"]:
        raise ValueError("num_valid_queries exceeds num_queries")
    if "num_ignored_queries" in metrics:
        ignored = metrics["num_ignored_queries"]
        if type(ignored) is not int or ignored < 0 or ignored + metrics["num_valid_queries"] != metrics["num_queries"]:
            raise ValueError("Ignored/valid query counts do not add up")
    return True


def _row(path, payload=None, run=None, job=None):
    payload, run, job = payload or {}, run or {}, job or {}
    config = run.get("config", {})
    metrics = payload.get("metrics", {})
    options = payload.get("model_options", {})
    evaluation = payload.get("evaluation", {})
    if not isinstance(options, dict) or not isinstance(evaluation, dict):
        raise ValueError("model_options and evaluation must be objects when present")
    source = payload.get("source_dataset", config.get("dataset_name"))
    dataset = payload.get("dataset", job.get("dataset", config.get("dataset_name")))
    protocol = payload.get("protocol", job.get("protocol"))
    if protocol is None and source is not None and dataset is not None:
        protocol = "in_domain" if source == dataset else "direct_transfer"
    row = {key: None for key in _CSV_FIELDS}
    row.update(job_id=job.get("id"), status="evaluated" if payload else "pending",
        run_dir=run.get("run_dir", job.get("run_dir")), protocol=protocol, dataset=dataset,
        source_dataset=source, checkpoint=payload.get("checkpoint", job.get("checkpoint")),
        checkpoint_sha256=payload.get("checkpoint_sha256"), result_file=str(Path(path).resolve()),
        config_origin=run.get("config_origin"), seed=payload.get("seed", config.get("seed", job.get("seed"))),
        mode=payload.get("source_training_mode", config.get("mode")),
        augmentation_manifest=config.get("augmentation_manifest"), selection=payload.get("selection"),
        source_selection_rule=payload.get("source_selection_rule"), initialization=payload.get("initialization"),
        source_epoch=payload.get("source_epoch"), source_global_step=payload.get("source_global_step"),
        evaluation_scope=payload.get("evaluation_scope"), test_annotation_sha256=payload.get("test_annotation_sha256"),
        planned_epochs=run.get("planned_epochs"),
        completed_epochs=run.get("completed_epochs"), training_status=run.get("training_status"),
        precision=evaluation.get("precision", config.get("precision")),
        device=evaluation.get("device", config.get("resolved_device", config.get("device"))),
        batch_size=evaluation.get("batch_size", config.get("test_batch_size")),
        elapsed_seconds=payload.get("elapsed_seconds", evaluation.get("elapsed_seconds")),
        resources=payload.get("resources", evaluation.get("resources")), duplicate_count=1,
        duplicate_conflict=False)
    row.update({flag: options.get(flag, config.get(flag)) for flag in _FLAGS})
    row.update({key: metrics.get(key) for key in (*_METRICS, *_COUNTS)})
    return row


def _duplicates(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "evaluated":
            identity = (row["checkpoint_sha256"] or row["checkpoint"], row["dataset"], row["protocol"])
            groups[identity].append(row)
    duplicates = conflicts = 0
    for identity, members in groups.items():
        if len(members) < 2:
            continue
        duplicates += 1
        annotation_hashes = {row["test_annotation_sha256"] for row in members if row["test_annotation_sha256"]}
        conflict = (len({tuple(row[key] for key in (*_METRICS, *_COUNTS)) for row in members}) > 1
                    or len(annotation_hashes) > 1)
        conflicts += int(conflict)
        group_id = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:12]
        for row in members:
            row.update(duplicate_group=group_id, duplicate_count=len(members), duplicate_conflict=conflict)
    return duplicates, conflicts


def collect_results(root, output_json=None, output_csv=None, *, plan_path=None):
    """Return all recorded evaluations and explicit pending rows, never a best score.

    No checkpoints are opened. ``plan_path`` accepts version-1 experiments.json
    with jobs whose kind is 'train' or 'evaluate'; an evaluate job uses output,
    dataset, protocol and optionally run_dir/checkpoint. Relative plan paths are
    resolved against the plan directory. Other JSON is ignored.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Experiment root does not exist: {}".format(root))
    outputs = {_key(value) for value in (output_json, output_csv) if value is not None}
    paths = [path for path in _json_files(root) if _key(path) not in outputs]
    warnings, rejected, rows = [], [], []
    runs = {}
    for path in paths:
        if path.name == "configs.json":
            runs[_key(path.parent)] = _training_run(path.parent, warnings)

    def find_run(path, payload, job=None):
        if job and job.get("run_dir"):
            directory = Path(job["run_dir"])
        elif payload.get("checkpoint"):
            directory = _resolve(payload["checkpoint"], path.parent).parent
        else:
            return None
        key = _key(directory)
        if key not in runs and (directory / "configs.json").is_file():
            runs[key] = _training_run(directory, warnings)
        return runs.get(key)

    if plan_path is not None:
        plan_path = Path(plan_path).expanduser().resolve()
        plan = _read(plan_path)
        if (not isinstance(plan, dict) or type(plan.get("version")) is not int or plan["version"] != 1
                or not isinstance(plan.get("jobs"), list)):
            raise ValueError("Experiment plan requires version: 1 and a jobs list")
        ids, result_paths = set(), set()
        for raw_job in plan["jobs"]:
            if not isinstance(raw_job, dict) or not isinstance(raw_job.get("id"), str) or raw_job["id"] in ids:
                raise ValueError("Experiment jobs need unique string IDs")
            ids.add(raw_job["id"])
            job = dict(raw_job)
            if job.get("kind") not in {"train", "evaluate"}:
                raise ValueError("Experiment job kind must be train or evaluate")
            if job.get("run_dir"):
                job["run_dir"] = str(_resolve(job["run_dir"], plan_path.parent))
                if (Path(job["run_dir"]) / "configs.json").is_file():
                    find_run(plan_path, {}, job)
            if job["kind"] == "train":
                if job.get("run_dir") and _key(job["run_dir"]) not in runs:
                    if not isinstance(job.get("config", {}), dict):
                        raise ValueError("Planned training config must be an object")
                    runs[_key(job["run_dir"])] = _training_run(job["run_dir"], warnings, job.get("config", {}))
                continue
            if not isinstance(job.get("output"), str) or not job["output"]:
                raise ValueError("Evaluation jobs require an output path")
            path = _resolve(job["output"], plan_path.parent)
            if _key(path) in result_paths:
                raise ValueError("Evaluation jobs must use distinct output paths")
            result_paths.add(_key(path))
            payload = {}
            error = None
            if path.is_file():
                try:
                    payload = _read(path)
                    if not _validate_result(payload):
                        raise ValueError("Expected a test evaluation result, not a smoke/subset/other report")
                    for key in ("dataset", "protocol"):
                        if job.get(key) is not None and payload.get(key) is not None and job[key] != payload[key]:
                            raise ValueError("Result {} differs from the experiment job".format(key))
                    if job.get("run_dir") and _key(_resolve(payload["checkpoint"], plan_path.parent).parent) != _key(job["run_dir"]):
                        raise ValueError("Result checkpoint does not belong to the experiment's source run_dir")
                except (ValueError, TypeError) as failure:
                    error = str(failure)
                    rejected.append({"path": str(path), "reason": error})
                    payload = {}
            run = find_run(path, payload, job)
            row = _row(path, payload, run, job)
            if error:
                row.update(status="invalid", error=error)
            rows.append(row)
    else:
        for path in paths:
            try:
                payload = _read(path)
                if not _validate_result(payload):
                    continue
                rows.append(_row(path, payload, find_run(path, payload)))
            except (ValueError, TypeError) as error:
                rejected.append({"path": str(path), "reason": str(error)})
        evaluated_runs = {row["run_dir"] for row in rows if row["run_dir"]}
        for run in sorted(runs.values(), key=lambda value: value["run_dir"]):
            if run["run_dir"] not in evaluated_runs:
                rows.append(_row(Path(run["run_dir"]) / "test_metrics.json", run=run))

    duplicate_groups, conflicting_groups = _duplicates(rows)
    result = {"version": 1, "root": str(root), "plan": str(Path(plan_path).resolve()) if plan_path else None,
        "runs": rows, "training_runs": [runs[key] for key in sorted(runs)],
        "report": {"evaluated": sum(row["status"] == "evaluated" for row in rows),
            "pending": sum(row["status"] == "pending" for row in rows),
            "invalid": sum(row["status"] == "invalid" for row in rows),
            "training_completed": sum(run["training_status"] == "completed" for run in runs.values()),
            "training_partial": sum(run["training_status"] == "partial" for run in runs.values()),
            "training_not_started": sum(run["training_status"] == "not_started" for run in runs.values()),
            "duplicate_groups": duplicate_groups, "conflicting_duplicate_groups": conflicting_groups,
            "rejected_files": rejected, "warnings": warnings,
            "selection": "All accepted records retained; no test-score selection or averaging."}}
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()} for row in rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="logs")
    parser.add_argument("--plan", dest="plan_path")
    parser.add_argument("--output_json", default="results/summary.json")
    parser.add_argument("--output_csv", default="results/summary.csv")
    args = parser.parse_args(argv)
    result = collect_results(**vars(args))
    print("{} evaluated, {} pending, {} invalid; {} duplicate groups".format(
        result["report"]["evaluated"], result["report"]["pending"], result["report"]["invalid"], result["report"]["duplicate_groups"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
