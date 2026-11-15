"""Deterministic single-process identity sampler for four/five-field rows."""

import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances, seed=42):
        if num_instances < 2 or batch_size % num_instances or batch_size // num_instances < 2:
            raise ValueError("Identity sampling needs K>=2, batch_size divisible by K, and at least two identities/batch")
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.seed, self.epoch = seed, 0
        self.index_dic = defaultdict(list)
        self.image_ids = {}
        for index, item in enumerate(data_source):
            pid, image_id = (item["pid"], item["image_id"]) if isinstance(item, dict) else item[:2]
            self.index_dic[pid].append(index)
            self.image_ids[index] = image_id
        self.pids = sorted(self.index_dic)
        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError(f"Need at least {self.num_pids_per_batch} training identities, found {len(self.pids)}")
        self._indices = self._build_indices()

    def _build_indices(self):
        rng = random.Random(self.seed + self.epoch)
        groups = {}
        for pid, indices in self.index_dic.items():
            by_image = defaultdict(list)
            for index in indices:
                by_image[self.image_ids[index]].append(index)
            for bucket in by_image.values():
                rng.shuffle(bucket)
            ordered = []
            while by_image:
                image_ids = list(by_image)
                rng.shuffle(image_ids)
                for image_id in image_ids:
                    ordered.append(by_image[image_id].pop())
                    if not by_image[image_id]:
                        del by_image[image_id]
            count = math.ceil(len(ordered) / self.num_instances) * self.num_instances
            while len(ordered) < count:
                ordered.append(rng.choice(indices))
            groups[pid] = [ordered[i:i + self.num_instances]
                           for i in range(0, count, self.num_instances)]
        available, result = list(self.pids), []
        while len(available) >= self.num_pids_per_batch:
            for pid in rng.sample(available, self.num_pids_per_batch):
                result.extend(groups[pid].pop())
                if not groups[pid]:
                    available.remove(pid)
        return result

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._indices = self._build_indices()

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)
