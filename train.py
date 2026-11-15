"""Single-device entry point for the recovered NCHNet baseline."""
import sys
from utils.options import get_args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = get_args(argv)
    if args.check_resources:
        import json
        from tools.check_resources import inspect_resources
        report = inspect_resources(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 2
    if not args.training:
        import json
        from processor.baseline import evaluate_checkpoint, _explicit_destinations
        if not args.checkpoint:
            raise ValueError("--test requires --checkpoint; alternatively use python test.py --checkpoint PATH.")
        supplied = _explicit_destinations(argv)
        overrides = {key: getattr(args, key) if key in supplied else None for key in ("dataset_name", "root_dir", "bpe_path")}
        result = evaluate_checkpoint(args.checkpoint, **overrides, device=args.device,
            batch_size=args.test_batch_size, num_workers=args.num_workers)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.checkpoint:
        raise ValueError("--checkpoint is for evaluation; use --test or test.py, --resume, or --finetune as appropriate.")
    from processor.baseline import run_training
    run_training(args, argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
