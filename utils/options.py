"""Command-line configuration for the recovered, single-device baseline."""
import argparse
import math
from pathlib import Path


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes", "1", "on"}:
        return True
    if value.lower() in {"false", "no", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false, yes/no, 1/0 or on/off")


def _boolean(parser, name, default=False, **kwargs):
    parser.add_argument("--" + name, type=str_to_bool, nargs="?", const=True, default=default, **kwargs)
    parser.add_argument("--no-" + name, dest=name, action="store_false", help=argparse.SUPPRESS)


def build_parser():
    parser = argparse.ArgumentParser(description="NCHNet recovered baseline")
    parser.add_argument("--mode", choices=("baseline", "pretrain", "finetune"), default="baseline")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--dataset_name", choices=("RSTPReid", "CUHK-PEDES", "ICFG-PEDES", "Synth_PEDES"), default="RSTPReid")
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--log_period", type=int, default=100)
    parser.add_argument("--eval_period", type=int, default=1)
    parser.add_argument("--val_dataset", choices=("val",), default="val", help="Only validation data may select checkpoints")
    parser.add_argument("--resume", nargs="?", const=True, default=False, metavar="CHECKPOINT")
    parser.add_argument("--resume_ckpt_file", default="")
    parser.add_argument("--finetune", default="", help="Initialize from a retrieval checkpoint; not an exact resume")
    parser.add_argument("--initialize_new_modules", action="store_true", help="With --finetune, explicitly initialize wholly absent A2DFP or NAC branches")
    parser.add_argument("--pretrain", default="", help="Legacy source dataset alias for pretrain mode")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to evaluate")
    parser.add_argument("--check_resources", action="store_true", help="Inspect requirements without training")
    parser.add_argument("--test", dest="training", default=True, action="store_false")
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--clip_weights", "--clip_checkpoint", dest="clip_checkpoint", default="", help="Local CLIP checkpoint")
    parser.add_argument("--download_root", default=None)
    _boolean(parser, "allow_download", help="Explicitly permit downloading CLIP weights")
    parser.add_argument("--bpe_path", default=str(Path(__file__).resolve().parents[1] / "data" / "bpe_simple_vocab_16e6.txt.gz"))
    parser.add_argument("--augmentation_manifest", default="", help="Offline generated text/image manifest; paths are relative to this file")
    parser.add_argument("--img_size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument("--temperature", type=float, default=0.02)
    _boolean(parser, "freeze_backbone")
    _boolean(parser, "img_aug", default=True)
    parser.add_argument("--loss_names", default="sdm+id")
    parser.add_argument("--id_loss_weight", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sampler", choices=("auto", "random", "identity"), default="auto", help="auto uses identity sampling for CDFP, random otherwise")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--optimizer", choices=("Adam", "AdamW", "SGD"), default="Adam")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--lr_factor", type=float, default=10.0)
    parser.add_argument("--bias_lr_factor", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_method", choices=("linear", "constant"), default="linear")
    parser.add_argument("--lrscheduler", choices=("cosine", "step", "linear", "poly", "exp"), default="cosine")
    parser.add_argument("--milestones", nargs="+", type=int, default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--power", type=float, default=0.9)
    # A2DFP/CDFP are optional; other feature flags remain for later recovery stages.
    for name in ("MLM", "nam", "use_dfp", "use_a2dfp", "use_cdfp", "use_nac", "use_llm_augment", "use_generated"):
        _boolean(parser, name)
    parser.add_argument("--include_nac", "-include_nac", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--noise_threshold", "-noise_threshold", "--nac_threshold", type=float, default=0.7)
    parser.add_argument("--generator_depth", "-generator_depth", "--nac_generator_depth", type=int, default=6)
    parser.add_argument("--gen_loss_weight", "-gen_loss_weight", type=float, default=0.5)
    parser.add_argument("--cmt_depth", "--nac_cmt_depth", type=int, default=4)
    parser.add_argument("--nac_corruption_rate", type=float, default=0.15, help="Controlled attribute-span corruption rate for aligned human training captions")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0)
    parser.add_argument("--masked_token_rate", type=float, default=0.8)
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1)
    parser.add_argument("--llm_num_aug", type=int, default=2)
    parser.add_argument("--a2dfp_embed_dim", type=int, default=2048)
    parser.add_argument("--a2dfp_blend_weight", type=float, default=0.5)
    parser.add_argument("--a2dfp_loss_weight", type=float, default=None, help="Deprecated: no separate A2DFP auxiliary objective is defined")
    parser.add_argument("--cdfp_loss_weight", type=float, default=0.1)
    parser.add_argument("--cdfp_margin", type=float, default=0.1)
    parser.add_argument("--cdfp_temperature", type=float, default=None)
    parser.add_argument("--cdfp_k_negatives", type=int, default=2)
    return parser


def get_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.img_size = tuple(args.img_size)
    if isinstance(args.resume, str):
        if args.resume_ckpt_file and args.resume_ckpt_file != args.resume:
            parser.error("--resume and --resume_ckpt_file specify different checkpoints")
        args.resume_ckpt_file = args.resume
        args.resume = True
    if args.resume_ckpt_file:
        args.resume = True
    if args.resume and not args.resume_ckpt_file:
        parser.error("--resume requires a checkpoint path")
    if args.resume and args.finetune:
        parser.error("choose an exact --resume or --finetune initialization, not both")
    if args.training and not args.resume and (args.use_llm_augment or args.use_generated) and not args.augmentation_manifest:
        parser.error("Offline augmentation requires --augmentation_manifest PATH")
    if args.initialize_new_modules and (args.resume or not args.finetune):
        parser.error("--initialize_new_modules requires --finetune and cannot be used with --resume")
    if args.local_rank != 0:
        parser.error("stage 1 supports a single process; distributed training is not enabled")
    for name in ("batch_size", "test_batch_size", "num_epoch", "text_length", "stride_size", "num_instance", "log_period", "eval_period"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0 or min(args.img_size) <= 0:
        parser.error("num_workers must be nonnegative and img_size positive")
    if not 0 <= args.warmup_epochs < args.num_epoch:
        parser.error("warmup_epochs must satisfy 0 <= warmup_epochs < num_epoch")
    if not 0 < args.warmup_factor <= 1:
        parser.error("warmup_factor must be in (0, 1]")
    if args.lr <= 0 or args.lr_factor <= 0 or (args.head_lr is not None and args.head_lr <= 0):
        parser.error("learning rates and lr_factor must be positive")
    if args.temperature <= 0 or not 0 <= args.label_smoothing < 1:
        parser.error("temperature must be positive and label_smoothing must be in [0, 1)")
    if args.weight_decay < 0 or args.target_lr < 0:
        parser.error("weight decay and target learning rate must be nonnegative")
    if args.head_lr is None:
        args.head_lr = args.lr * args.lr_factor
    args.use_nac = args.use_nac or args.include_nac
    args.include_nac = args.use_nac
    if not math.isfinite(args.noise_threshold) or not 0 <= args.noise_threshold <= 2:
        parser.error("noise_threshold must be finite and in [0, 2] for cosine noise scores")
    if args.generator_depth < 1 or args.cmt_depth < 1:
        parser.error("generator_depth and cmt_depth must be positive")
    if not math.isfinite(args.gen_loss_weight) or args.gen_loss_weight < 0:
        parser.error("gen_loss_weight must be finite and nonnegative")
    if not math.isfinite(args.nac_corruption_rate) or not 0 <= args.nac_corruption_rate <= 1:
        parser.error("nac_corruption_rate must be finite and in [0, 1]")
    if not math.isfinite(args.a2dfp_blend_weight) or not 0 <= args.a2dfp_blend_weight <= 1:
        parser.error("a2dfp_blend_weight must be finite and in [0, 1]")
    if args.a2dfp_embed_dim < 1:
        parser.error("a2dfp_embed_dim must be positive (the hidden projection dimension)")
    if args.a2dfp_loss_weight is not None:
        parser.error("A2DFP has no separate auxiliary loss; configure its residual blend and the retrieval objectives instead")
    if not math.isfinite(args.cdfp_margin) or args.cdfp_margin < 0 or not math.isfinite(args.cdfp_loss_weight) or args.cdfp_loss_weight < 0:
        parser.error("CDFP margin and loss weight must be finite and nonnegative")
    if args.cdfp_k_negatives < 1:
        parser.error("cdfp_k_negatives must be positive")
    if args.use_cdfp and args.cdfp_temperature is not None:
        parser.error("CDFP uses the paper's raw-cosine hinge formulation; temperature scaling is not defined for this mode")
    args._sampler_requested = args.sampler
    if args.sampler == "auto":
        args.sampler = "identity" if args.use_cdfp else "random"
    if args.use_cdfp and args.sampler != "identity":
        parser.error("CDFP requires --sampler identity (or auto) to supply positive and negative identities")
    if args.use_cdfp and (args.num_instance < 2 or args.batch_size % args.num_instance or args.batch_size // args.num_instance < 2):
        parser.error("CDFP requires K>=2, batch_size divisible by K, and at least two identities per batch")
    args.distributed = False
    return args
