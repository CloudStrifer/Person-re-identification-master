"""Evaluate saved retrieval features; a target dataset override performs direct transfer."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description="NCHNet independent test evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_name", default=None,
                        help="Target dataset; overriding the source evaluates without target training or checkpoint selection")
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--bpe_path", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON metrics output")
    args = parser.parse_args(argv)
    if args.test_batch_size is not None and args.test_batch_size <= 0:
        parser.error("test_batch_size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("num_workers must be nonnegative")
    from processor.baseline import evaluate_checkpoint
    result = evaluate_checkpoint(args.checkpoint, dataset_name=args.dataset_name,
        root_dir=args.root_dir, bpe_path=args.bpe_path, device=args.device,
        batch_size=args.test_batch_size, num_workers=args.num_workers, output_file=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
