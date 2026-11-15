"""Original pedestrian annotations, without online generation or token mutation."""

import json
import logging
from pathlib import Path

from tools.check_resources import resolve_annotation_layout


def empty_retrieval_split():
    return {"image_pids": [], "img_paths": [], "caption_pids": [],
            "captions": [], "gt_img_paths": []}


class PedestrianAnnotations:
    """Read one dataset and map *training* identities to contiguous labels.

    ``train_records`` is the canonical metadata-preserving representation.
    ``train`` retains five-field rows for existing samplers/reference scripts.
    Evaluation identity labels remain the original dataset labels.
    """

    logger = logging.getLogger("NCHNet.dataset")
    folder = ""
    annotation_file = ""
    annotation_aliases = ()
    image_folder = "imgs"
    path_key = "file_path"
    default_split = None
    identity_disjoint_splits = True

    def __init__(self, root="", verbose=True, text_length=77, **options):
        self.augmentation_signature = None
        self.text_length = text_length
        dataset_root, annotation_path = resolve_annotation_layout(
            root, self.folder, self.annotation_file, self.annotation_aliases)
        self.dataset_dir = str(dataset_root.resolve())
        self.img_dir = str(dataset_root / self.image_folder) if self.image_folder else str(dataset_root)
        self.anno_path = str(annotation_path)
        if not Path(self.anno_path).is_file():
            raise FileNotFoundError(
                f"Missing {self.folder} original annotations: {self.anno_path}. "
                "Set --root_dir to the data parent (or dataset folder); "
                "generated-caption caches are not original annotations."
            )
        if not Path(self.img_dir).is_dir():
            raise FileNotFoundError(f"Missing {self.folder} image directory: {self.img_dir}")
        with open(self.anno_path, encoding="utf-8-sig") as stream:
            annotations = json.load(stream)
        if not isinstance(annotations, list):
            raise ValueError(f"Expected a list of original annotation records: {self.anno_path}")

        grouped = {"train": [], "val": [], "test": []}
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(f"Annotation #{index} must be an object: {self.anno_path}")
            split = annotation.get("split", self.default_split)
            if split in ("validation", "valid"):
                split = "val"
            if split not in grouped:
                raise ValueError(f"Unknown or missing split {split!r} at record #{index}: {self.anno_path}")
            try:
                source_pid = int(annotation["id"])
                relative_path = annotation[self.path_key]
                captions = annotation["captions"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid original annotation #{index}: {self.anno_path}") from error
            if isinstance(captions, str):
                captions = [captions]
            if (not isinstance(relative_path, str) or not relative_path
                    or not isinstance(captions, list) or not captions
                    or any(not isinstance(c, str) or not c.strip() for c in captions)):
                raise ValueError(f"Image path and nonempty captions required at record #{index}: {self.anno_path}")
            path = Path(relative_path.replace("\\", "/"))
            if not path.is_absolute():
                path = Path(self.img_dir) / path
            grouped[split].append({
                "source_pid": source_pid, "source_image_id": relative_path.replace("\\", "/"),
                "img_path": str(path), "captions": captions, "split": split,
            })

        seen_paths = {}
        for split, records in grouped.items():
            for record in records:
                path = str(Path(record["img_path"]).resolve())
                previous = seen_paths.setdefault(path, (split, record["source_pid"]))
                if previous != (split, record["source_pid"]):
                    raise ValueError(f"Source image has conflicting split/identity: {path}")

        if self.identity_disjoint_splits:
            identity_split = {}
            for split, records in grouped.items():
                for record in records:
                    source_pid = record["source_pid"]
                    previous = identity_split.setdefault(source_pid, split)
                    if previous != split:
                        raise ValueError(
                            f"{self.folder} requires identity-disjoint splits: "
                            f"identity {source_pid} appears in both {previous} and {split}. "
                            "Use the original benchmark split rather than splitting individual images."
                        )

        self.train_annos, self.val_annos, self.test_annos = (
            grouped["train"], grouped["val"], grouped["test"])
        self.label_map = {pid: label for label, pid in enumerate(
            sorted({record["source_pid"] for record in self.train_annos}))}
        self.train_id_container = set(self.label_map.values())
        self.train_records = []
        image_ids = {}
        for record in self.train_annos:
            image_id = image_ids.setdefault(record["img_path"], len(image_ids))
            for caption in record["captions"]:
                self.train_records.append({
                    "pid": self.label_map[record["source_pid"]], "image_id": image_id,
                    "img_path": record["img_path"], "caption": caption,
                    "source_pid": record["source_pid"], "source_image_id": record["source_image_id"],
                    "split": "train", "caption_source": "human",
                    "sample_id": f"train:{image_id}:{len(self.train_records)}",
                })
        self.train = [(r["pid"], r["image_id"], r["img_path"], r["caption"], None)
                      for r in self.train_records]
        self.val = self._retrieval(self.val_annos)
        self.test = self._retrieval(self.test_annos)
        self.val_id_container = set(self.val["image_pids"])
        self.test_id_container = set(self.test["image_pids"])
        if options.get("use_llm_augment", False) or options.get("use_generated", False):
            from .augmentation import apply_augmentation
            apply_augmentation(self, options.get("augmentation_manifest"),
                               use_llm_augment=options.get("use_llm_augment", False),
                               use_generated=options.get("use_generated", False))
        if verbose:
            self.show_dataset_info()

    @staticmethod
    def _retrieval(records):
        result = empty_retrieval_split()
        gallery_seen = set()
        for record in records:
            pid, path = record["source_pid"], record["img_path"]
            if path not in gallery_seen:
                result["image_pids"].append(pid)
                result["img_paths"].append(path)
                gallery_seen.add(path)
            for caption in record["captions"]:
                result["caption_pids"].append(pid)
                result["captions"].append(caption)
                result["gt_img_paths"].append(path)
        return result

    def show_dataset_info(self):
        self.logger.info("%s: train=%d identities/%d images/%d captions; val=%d images; test=%d images",
                         self.folder, len(self.label_map), len({r[1] for r in self.train}),
                         len(self.train), len(self.val["img_paths"]), len(self.test["img_paths"]))
