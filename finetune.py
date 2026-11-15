"""Compatibility entry point using the same implementation as train.py."""
import sys
from train import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--mode" not in arguments and not any(a.startswith("--mode=") for a in arguments):
        arguments = ["--mode", "finetune", *arguments]
    raise SystemExit(main(arguments))

