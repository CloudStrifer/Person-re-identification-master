"""Generate and run explicit, individually selected NCHNet experiments."""
import argparse
import copy
import json
from pathlib import Path
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
GROUPS = ("main", "module_ablation", "augmentation_ablation", "sensitivity")


def _read(path):
    if isinstance(path, dict):
        return copy.deepcopy(path)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    from augmentation.workflow import atomic_write_json
    atomic_write_json(path, value)


def _slug(value):
    return value.lower().replace("_", "-")


def options_to_argv(options):
    from utils.options import build_parser
    actions = build_parser()._option_string_actions
    result = []
    for key, value in options.items():
        if value is None:
            continue
        option = "--" + key
        if option not in actions:
            raise ValueError("Unknown training option: " + option)
        if actions[option].nargs == 0:
            if type(value) is not bool:
                raise ValueError("Flag requires a boolean: " + option)
            if value:
                result.append(option)
            continue
        result.append(option)
        values = value if isinstance(value, (list, tuple)) else [value]
        result.extend(str(item).lower() if isinstance(item, bool) else str(item) for item in values)
    return result


def build_plan(spec, paths, output_dir, *, groups=None, targets=None, seeds=None, overrides=None):
    """Pure expansion; no dataset reads, model imports, downloads or training."""
    spec, paths = _read(spec), _read(paths)
    if spec.get("version") != 1:
        raise ValueError("Experiment specification requires version 1")
    groups = list(GROUPS if groups is None else groups)
    if set(groups) - set(GROUPS):
        raise ValueError("Unknown experiment group")
    targets = list(spec["targets"] if targets is None else targets)
    if not targets or set(targets) - set(spec["targets"]):
        raise ValueError("Targets must be nonempty supported target datasets")
    seeds = list(seeds if seeds is not None else [spec["common"]["seed"]])
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers")
    common = {**spec["common"], **(overrides or {})}
    reserved = {"dataset_name", "root_dir", "finetune", "resume", "output_dir", "mode", "augmentation_manifest"}
    if reserved.intersection(overrides or {}):
        raise ValueError("Protocol/path fields cannot be training overrides")
    for required in ("root_dir", "bpe_path", "clip_weights"):
        if not paths.get(required):
            raise ValueError("Paths require " + required)
    output_dir = Path(output_dir)
    jobs = []
    reused_evaluations = []

    def root_for(dataset):
        return paths.get("dataset_roots", {}).get(dataset, paths["root_dir"])

    def train(identity, dataset, modules, augmentation, seed, group, source=None, note=None, changes=None):
        identity = f"seed{seed}-{identity}"
        config = {**common, **spec["module_profiles"][modules], **(changes or {}),
            "seed": seed, "dataset_name": dataset, "root_dir": root_for(dataset),
            "bpe_path": paths["bpe_path"], "clip_weights": paths["clip_weights"],
            "mode": "pretrain" if dataset == "Synth_PEDES" else "finetune",
            "use_llm_augment": augmentation in {"text", "both", "unfiltered"},
            "use_generated": augmentation in {"image", "both", "unfiltered"},
            "output_dir": (Path("runs") / identity).as_posix(), "name": identity}
        manifest = None
        if augmentation != "none":
            available = paths.get("augmentation_manifests", {}).get(dataset, {})
            if isinstance(available, str):
                available = {"both": available}
            # Text-only/image-only can use the respective flags on a dual manifest.
            manifest = available.get(augmentation) or (available.get("both") if augmentation in {"text", "image"} else None)
            if augmentation == "unfiltered" and manifest and manifest == available.get("both"):
                raise ValueError("Unfiltered and filtered experiments require separate manifests")
            config["augmentation_manifest"] = manifest or ""
        source_id = f"seed{seed}-{source}" if source else None
        if source_id:
            config["initialize_new_modules"] = True
        job = {"id": identity, "kind": "train", "group": group, "dataset": dataset,
            "protocol": "pretrain" if dataset == "Synth_PEDES" else "finetune",
            "variant": modules, "augmentation": augmentation, "seed": seed,
            "depends_on": [source_id] if source_id else [], "initialize_from": source_id,
            "initialization_selection": "last", "config": config,
            "run_dir": config["output_dir"], "paper_table": note,
            "requirements": ["augmentation_manifest"] if augmentation != "none" and not manifest else []}
        jobs.append(job)
        return job

    def evaluation(identity, dataset, seed, group, source, selection="selected", protocol="in_domain"):
        identity = f"seed{seed}-{identity}"
        jobs.append({"id": identity, "kind": "evaluate", "group": group,
            "dataset": dataset, "protocol": protocol, "seed": seed,
            "depends_on": [source["id"]], "checkpoint_from": source["id"], "checkpoint_selection": selection,
            "run_dir": source["run_dir"], "output": (Path("results") / (identity + ".json")).as_posix(),
            "config": {"dataset_name": dataset, "root_dir": root_for(dataset), "bpe_path": paths["bpe_path"],
                "device": common["device"], "test_batch_size": common["test_batch_size"], "num_workers": common["num_workers"]}})
        return jobs[-1]

    for seed in dict.fromkeys(seeds):
        if "main" in groups:
            source = train("synth-pretrain", "Synth_PEDES", "pretrain", spec.get("source_augmentation", "none"), seed, "main")
            for dataset in targets:
                slug = _slug(dataset)
                evaluation("direct-" + slug, dataset, seed, "main", source, "last", "direct_transfer")
                target = train("finetune-" + slug, dataset, "full", "both", seed, "main", "synth-pretrain")
                evaluation("test-finetune-" + slug, dataset, seed, "main", target)
        if any(group in groups for group in GROUPS[1:]):
            train("synth-baseline", "Synth_PEDES", "baseline", "none", seed, "ablation_source")
            default_evaluation = None
            for group in ("module_ablation", "augmentation_ablation"):
                if group not in groups:
                    continue
                for row in spec[group]:
                    target = train("rstp-" + row["id"], "RSTPReid", row["modules"], row["augmentation"], seed,
                        group, "synth-baseline", row.get("paper_table"))
                    result_job = evaluation("test-rstp-" + row["id"], "RSTPReid", seed, group, target)
                    if row["id"] == "full" and row["modules"] == "full" and row["augmentation"] == "both":
                        default_evaluation = result_job["id"]
            if "sensitivity" in groups:
                for key, values in spec["sensitivity"].items():
                    for value in values:
                        if value == common[key] and default_evaluation:
                            reused_evaluations.append({"seed": seed, "parameter": key, "value": value,
                                                       "evaluation_job": default_evaluation})
                            continue
                        name = "sensitivity-{}-{}".format(key.replace("_", "-"), str(value).replace(".", "p"))
                        target = train(name, "RSTPReid", "full", "both", seed, "sensitivity",
                            "synth-baseline", "Figures 6/7", {key: value})
                        result_job = evaluation("test-" + name, "RSTPReid", seed, "sensitivity", target)
                        if value == common[key]:
                            default_evaluation = result_job["id"]
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment IDs")
    return {"version": 1, "groups": groups, "jobs": jobs, "_base_dir": str(output_dir.resolve()),
        "implementation_conventions": spec.get("implementation_conventions", []),
        "unsupported_experiments": spec.get("unsupported_experiments", []),
        "reused_evaluations": reused_evaluations,
        "training_overrides": overrides or {}, "results_are_pending": True}


def write_plan(plan, output):
    output = Path(output)
    saved = {key: value for key, value in plan.items() if not key.startswith("_")}
    if output.exists() and _read(output) != saved:
        raise ValueError("Plan already exists with different settings; use a new directory")
    _write(output, saved)
    return plan


def load_plan(path):
    plan = _read(path)
    if plan.get("version") != 1 or not isinstance(plan.get("jobs"), list):
        raise ValueError("Expected a version 1 experiment plan with jobs")
    plan["_base_dir"] = str(Path(path).resolve().parent)
    return plan


def _path(plan, value):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else Path(plan["_base_dir"]) / path


def _job(plan, identity):
    matches = [job for job in plan["jobs"] if job["id"] == identity]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate job: " + identity)
    return matches[0]


def _source_checkpoint(plan, source_id, selection):
    source = _job(plan, source_id)
    directory = _path(plan, source["run_dir"])
    result = directory / "experiment_result.json"
    if not result.is_file():
        raise FileNotFoundError("Run the training job first: " + source_id)
    state = _read(result)
    if state.get("status") != "completed" or state.get("experiment_id") != source_id:
        raise ValueError("Training dependency is incomplete: " + source_id)
    if selection == "last":
        return directory / "last.pth"
    # Keep a completed run portable when its whole directory is moved.
    name = Path(state["selected_checkpoint"].replace("\\", "/")).name
    if name not in {"best.pth", "last.pth"}:
        raise ValueError("Unexpected selected checkpoint name")
    return directory / name


def resolved_config(plan, job, *, resume=False):
    config = copy.deepcopy(job["config"])
    if job.get("requirements"):
        raise ValueError("Missing paths for {}: {}".format(job["id"], ", ".join(job["requirements"])))
    if resume:
        if job["kind"] != "train":
            raise ValueError("Evaluation has no optimizer state to resume")
        config = {key: config[key] for key in ("device", "root_dir", "bpe_path", "num_workers")}
        if job["config"].get("augmentation_manifest"):
            config["augmentation_manifest"] = job["config"]["augmentation_manifest"]
        config["resume"] = str(_path(plan, job["run_dir"]) / "last.pth")
        return config
    if job["kind"] == "train":
        config["output_dir"] = str(_path(plan, job["run_dir"]))
    if job["kind"] == "train" and job.get("initialize_from"):
        config["finetune"] = str(_source_checkpoint(plan, job["initialize_from"], job["initialization_selection"]))
    elif job["kind"] == "evaluate":
        config["checkpoint"] = str(_source_checkpoint(plan, job["checkpoint_from"], job["checkpoint_selection"]))
    return config


def check_job(plan, identity, *, resume=False):
    """Presence/environment preflight; full data validation happens on load."""
    job = _job(plan, identity)
    try:
        config = resolved_config(plan, job, resume=resume)
        from tools.check_resources import inspect_resources
        from utils.options import get_args
        if job["kind"] == "evaluate":
            args = get_args(options_to_argv({**config, "clip_weights": config["checkpoint"]}))
        else:
            args = get_args(options_to_argv(config))
        report = inspect_resources(args)
        manifest = config.get("augmentation_manifest")
        if manifest:
            present = Path(manifest).is_file()
            report["resources"]["augmentation_manifest"] = {"path": manifest, "exists": present}
            if not present:
                report["missing_resources"].append("augmentation_manifest")
                report["ready"] = False
        if config["device"] == "cuda" and report["dependencies"]["torch"]:
            import torch
            report["torch_cuda_available"] = torch.cuda.is_available()
            report["ready"] = report["ready"] and report["torch_cuda_available"]
        report["job"] = identity
        return report
    except (ValueError, FileNotFoundError) as error:
        return {"ready": False, "job": identity, "reason": str(error)}


def run_job(plan, identity, *, resume=False):
    """Run one named job; never recursively launch its dependencies."""
    job = _job(plan, identity)
    destination = (_path(plan, job["run_dir"]) / "experiment_result.json"
                   if job["kind"] == "train" else _path(plan, job["output"]))
    if destination.exists():
        raise FileExistsError("Job output already exists: " + str(destination))
    report = check_job(plan, identity, resume=resume)
    if not report["ready"]:
        raise ValueError("Preflight failed: " + json.dumps(report, ensure_ascii=False))
    config = resolved_config(plan, job, resume=resume)
    import torch
    from processor.baseline import resolve_device
    device = resolve_device(config["device"])
    resources = {"python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "device": str(device),
        "torch_cuda_version": torch.version.cuda, "cpu_threads": torch.get_num_threads()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        resources.update(gpu_name=torch.cuda.get_device_name(device),
            gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory)
    started = time.monotonic()
    if job["kind"] == "evaluate":
        if resume:
            raise ValueError("Evaluation has no optimizer state to resume")
        from processor.baseline import evaluate_checkpoint
        options = dict(config)
        options["batch_size"] = options.pop("test_batch_size")
        result = evaluate_checkpoint(**options)
        result.update(experiment_id=identity, experiment_group=job["group"], evaluation_scope="provided_test_split")
    else:
        from processor.baseline import run_training
        from utils.options import get_args
        arguments = options_to_argv(config)
        result = run_training(get_args(arguments), argv=arguments)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        resources.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    result.update(status="completed", experiment_id=identity,
                  elapsed_seconds=round(time.monotonic() - started, 3), resources=resources)
    _write(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("plan")
    generate.add_argument("--spec", default=str(PROJECT / "configs/experiments/paper.json"))
    generate.add_argument("--paths", required=True)
    generate.add_argument("--output_dir", required=True)
    generate.add_argument("--group", action="append", choices=GROUPS, dest="groups")
    generate.add_argument("--target", action="append", dest="targets")
    generate.add_argument("--seed", type=int, action="append", dest="seeds")
    generate.add_argument("--overrides", help="Optional JSON training overrides, recorded as protocol changes")
    for command in ("show", "check", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("--plan", required=True)
        sub.add_argument("--job", required=command != "show")
        if command in {"check", "run"}:
            sub.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_plan(args.spec, args.paths, args.output_dir, groups=args.groups, targets=args.targets,
                          seeds=args.seeds, overrides=_read(args.overrides) if args.overrides else None)
        write_plan(plan, Path(args.output_dir) / "experiments.json")
        print("Prepared {} jobs; no training executed.".format(len(plan["jobs"])))
        return 0
    plan = load_plan(args.plan)
    if args.command == "show":
        if args.job:
            print(json.dumps(_job(plan, args.job), ensure_ascii=False, indent=2))
        else:
            for job in plan["jobs"]:
                print("{}  {}  {}  depends={}".format(
                    job["id"], job["kind"], job["dataset"], ",".join(job["depends_on"]) or "none"))
    elif args.command == "check":
        report = check_job(plan, args.job, resume=args.resume)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 2
    else:
        print(json.dumps(run_job(plan, args.job, resume=args.resume), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
