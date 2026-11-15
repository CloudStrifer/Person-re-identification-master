"""Prepare stage-four offline augmentation; imports never generate or download."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command, *, catalog=True, scope=True):
        result = commands.add_parser(command)
        if catalog:
            result.add_argument("--catalog", required=True)
        result.add_argument("--output", required=True)
        if scope:
            result.add_argument("--limit", type=int)
            result.add_argument("--source_id", "--source-id", action="append", dest="source_ids")
        return result

    catalog = common("catalog", catalog=False)
    catalog.add_argument("--dataset", "--dataset_name", default="RSTPReid", dest="dataset_name")
    catalog.add_argument("--root_dir", default="data")

    def caption_options(command):
        result = common(command)
        result.add_argument("--provider", choices=("attribute-cache", "qwen"), default="attribute-cache")
        result.add_argument("--attributes", help="JSON mapping exact image IDs/retained image_path to extracted attributes")
        result.add_argument("--seed", type=int, default=0)
        result.add_argument("--model", default="qwen-vl-max-latest")
        result.add_argument("--base_url", "--base-url")
        result.add_argument("--api_key_env", default="DASHSCOPE_API_KEY")
        result.add_argument("--timeout", type=float, default=60)
        return result

    caption_options("text")
    imported = common("import-captions")
    imported.add_argument("--captions", required=True)
    imported.add_argument("--strict", action="store_true", help="Fail rather than report and skip unusable legacy source pairs")
    images = common("images")
    images.add_argument("--local_model_dir", "--base_model", required=True, dest="local_model_dir")
    images.add_argument("--checkpoint_dir", required=True)
    images.add_argument("--subject_map")
    images.add_argument("--output_dir")
    images.add_argument("--num_candidates", type=int, default=20)
    images.add_argument("--seed", type=int, default=0)
    images.add_argument("--num_inference_steps", type=int, default=30)
    images.add_argument("--guidance_scale", type=float, default=7.5)
    images.add_argument("--height", type=int, default=512)
    images.add_argument("--width", type=int, default=256)
    images.add_argument("--negative_prompt")
    images.add_argument("--device", default="cpu")
    images.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    images.add_argument("--network_alpha", type=float)
    filtering = common("filter")
    filtering.add_argument("--candidates", required=True)
    filtering.add_argument("--clip_weights", required=True)
    filtering.add_argument("--device", default="cpu")
    filtering.add_argument("--keep", type=int, default=10)
    filtering.add_argument("--batch_size", type=int, default=8)
    generated_captions = caption_options("caption-images")
    generated_captions.add_argument("--selected", required=True)
    merged = common("merge", catalog=False, scope=False)
    merged.add_argument("--catalog")
    merged.add_argument("--inputs", nargs="+", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    from augmentation import workflow
    try:
        if command in {"text", "caption-images"}:
            provider_name = options.pop("provider")
            provider_options = {key: options.pop(key) for key in ("model", "base_url", "api_key_env", "timeout")}
            if provider_name == "qwen":
                if options.get("attributes"):
                    raise ValueError("--attributes is for the attribute-cache backend; choose one backend")
                from augmentation.text import QwenVLAttributeProvider
                options["provider"] = QwenVLAttributeProvider(**provider_options)
            elif not options.get("attributes"):
                raise ValueError("The default attribute-cache backend requires --attributes")
        names = {"catalog": "export_catalog", "text": "augment_text", "import-captions": "import_captions",
                 "images": "generate_images", "filter": "filter_images", "caption-images": "caption_images",
                 "merge": "merge_manifests"}
        renames = {"catalog": "catalog_path", "captions": "captions_path", "candidates": "candidates_path",
                   "selected": "selected_path"}
        for key, renamed in renames.items():
            if key in options:
                options[renamed] = options.pop(key)
        result = getattr(workflow, names[command])(**options)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        parser.exit(2, "{}\n".format(error))
    rows = result.get("sources", result.get("records", result.get("candidates", [])))
    print("{}: {} records -> {}".format(command, len(rows), options["output"]))
    if "report" in result and "skipped_invalid_sources" in result["report"]:
        print("Skipped {} invalid training sources; details are in the output report.".format(len(result["report"]["skipped_invalid_sources"])))
    return result


if __name__ == "__main__":
    main()
