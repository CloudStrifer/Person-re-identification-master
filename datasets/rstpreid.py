"""RSTPReid original annotations; offline augmentation is a later stage."""
from .annotations import PedestrianAnnotations


class RSTPReid(PedestrianAnnotations):
    folder = "RSTPReid"
    annotation_file = "data_captions.json"
    path_key = "img_path"
