"""CUHK-PEDES original annotations."""
from .annotations import PedestrianAnnotations


class CUHKPEDES(PedestrianAnnotations):
    folder = "CUHK-PEDES"
    annotation_file = "reid_raw.json"
