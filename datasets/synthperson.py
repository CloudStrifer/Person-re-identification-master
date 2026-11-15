"""Full SYNTH-PEDES annotations with contiguous training identities."""
from .annotations import PedestrianAnnotations


class Synth_PEDES(PedestrianAnnotations):
    folder = "Synth-Pedes"
    annotation_file = "synthpedes-dataset.json"
    image_folder = ""
    default_split = "train"
    # SYNTH is a pretraining source; honor any explicit external manifest split.
    # Benchmark identity-disjoint protocol is enforced by the target adapters.
    identity_disjoint_splits = False

    def __init__(self, root="./data", train_split=1.0, **kwargs):
        if train_split != 1.0:
            raise ValueError(
                "The paper pretrains on full SYNTH-PEDES. Use train_split=1.0; "
                "an alternative holdout must be an explicitly defined experiment."
            )
        super().__init__(root=root, **kwargs)
        # Automatically generated source captions are not clean human targets.
        for record in self.train_records:
            if record["caption_source"] == "human":
                record["caption_source"] = "synthetic"
