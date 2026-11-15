"""ICFG-PEDES original annotations (no fabricated validation split)."""
from .annotations import PedestrianAnnotations


class ICFGPEDES(PedestrianAnnotations):
    folder = "ICFG-PEDES"
    annotation_file = "ICFG-PEDES.json"
    annotation_aliases = ("ICFG_PEDES.json",)
