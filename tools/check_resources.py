"""Read-only resource check using only the standard library."""
import importlib.util
import filecmp
import json
from pathlib import Path
import subprocess
import sys


def resolve_annotation_layout(root, folder, annotation_file, annotation_aliases=()):
    """Resolve only declared annotation names, rejecting conflicting copies.

    Shared with the actual data loader without importing its torch dependencies.
    Missing files retain the primary path so preflight can report them normally.
    """
    root = Path(root).expanduser()
    names = (annotation_file, *annotation_aliases)
    directory = (root if root.name.casefold() == folder.casefold()
                 or any((root / name).is_file() for name in names) else root / folder)
    existing = [directory / name for name in names if (directory / name).is_file()]
    selected = existing[0] if existing else directory / annotation_file
    if any(not filecmp.cmp(selected, other, shallow=False) for other in existing[1:]):
        raise ValueError("Ambiguous {} annotation files with different contents: {}. "
                         "Keep one authoritative annotation file.".format(folder, ", ".join(map(str, existing))))
    return directory, selected


def inspect_resources(args):
    names = {"RSTPReid": ("RSTPReid", "data_captions.json"), "CUHK-PEDES": ("CUHK-PEDES", "reid_raw.json"), "ICFG-PEDES": ("ICFG-PEDES", "ICFG-PEDES.json"), "Synth_PEDES": ("Synth-Pedes", "synthpedes-dataset.json")}
    dataset = args.pretrain if getattr(args, "pretrain", "") and args.mode == "pretrain" else args.dataset_name
    if dataset not in names:
        raise ValueError(f"Unknown dataset: {dataset}")
    folder, annotation = names[dataset]
    root = Path(args.root_dir).resolve()
    aliases = ("ICFG_PEDES.json",) if dataset == "ICFG-PEDES" else ()
    location, annotation_path = resolve_annotation_layout(root, folder, annotation, aliases)
    resources = {"annotations": annotation_path, "bpe": Path(args.bpe_path).resolve()}
    if dataset != "Synth_PEDES":
        resources["images_directory"] = location / "imgs"
    checkpoint = getattr(args, "resume_ckpt_file", "") or getattr(args, "finetune", "") or getattr(args, "clip_checkpoint", "")
    if checkpoint:
        resources["checkpoint"] = Path(checkpoint).resolve()
    else:
        cache = Path(getattr(args, "download_root", None) or Path.home() / ".cache" / "clip")
        resources["clip_weights"] = cache / (args.pretrain_choice.replace("/", "-") + ".pt")
    dependencies = {name: importlib.util.find_spec(module) is not None for name, module in {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "Pillow": "PIL", "PyYAML": "yaml", "easydict": "easydict", "prettytable": "prettytable", "ftfy": "ftfy", "regex": "regex", "tqdm": "tqdm"}.items()}
    files = {name: {"path": str(path), "exists": path.exists()} for name, path in resources.items()}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        gpu_info = gpu.stdout.strip() if gpu.returncode == 0 else "not detected"
    except (OSError, subprocess.TimeoutExpired):
        gpu_info = "not detected"
    missing = [name for name, value in files.items() if not value["exists"]]
    required_missing = [name for name in missing if not (name == "clip_weights" and getattr(args, "allow_download", False))]
    return {"ready": not required_missing and all(dependencies.values()), "python": sys.executable, "python_version": sys.version.split()[0], "dataset": dataset, "resources": files, "missing_resources": missing, "dependencies": dependencies, "nvidia_smi": gpu_info, "note": "Presence check only; contents, package imports, CUDA execution and memory require actual validation. No downloads."}


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.options import get_args
    report = inspect_resources(get_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
