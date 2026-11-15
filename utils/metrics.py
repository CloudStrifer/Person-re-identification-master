"""Text/image retrieval metrics, with bounded query-chunk memory use."""
import logging

import torch
import torch.nn.functional as F


def _rank_statistics(similarity, q_pids, g_pids, max_rank):
    similarity = torch.as_tensor(similarity).detach().float().cpu()
    q_pids = torch.as_tensor(q_pids).detach().view(-1).cpu()
    g_pids = torch.as_tensor(g_pids).detach().view(-1).cpu()
    if similarity.ndim != 2 or similarity.shape != (len(q_pids), len(g_pids)):
        raise ValueError("Similarity shape must match the query/gallery identity arrays")
    if max_rank <= 0 or len(q_pids) == 0 or len(g_pids) == 0:
        raise ValueError("Retrieval requires nonempty query/gallery and a positive max_rank")
    if not torch.isfinite(similarity).all():
        raise ValueError("Retrieval similarities must be finite")
    indices = torch.argsort(similarity, dim=1, descending=True, stable=True)
    matches = g_pids[indices].eq(q_pids[:, None])
    valid = matches.any(dim=1)
    count = int(valid.sum())
    if not count:
        return torch.zeros(max_rank), torch.tensor(0.), torch.tensor(0.), 0, indices
    matches = matches[valid]
    cumulative = matches.cumsum(dim=1)
    cmc = cumulative.clamp(max=1).float()
    ranks = torch.arange(max_rank).clamp(max=len(g_pids) - 1)
    cmc_sum = cmc[:, ranks].sum(dim=0)
    positions = torch.arange(1, len(g_pids) + 1, dtype=torch.float32)
    ap_sum = ((cumulative / positions) * matches).sum(dim=1).div(matches.sum(dim=1)).sum()
    last_relevant = (matches * positions).amax(dim=1)
    inp_sum = (matches.sum(dim=1) / last_relevant).sum()
    return cmc_sum, ap_sum, inp_sum, count, indices


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    """CMC/mAP/mINP in percent; queries with no gallery match are excluded.

    CMC is padded to max_rank for small galleries: Rank-k where k exceeds the
    gallery size equals the final available rank.
    """
    cmc, ap, inp, count, indices = _rank_statistics(similarity, q_pids, g_pids, max_rank)
    if not count:
        raise ValueError("No query identity has a match in the gallery")
    cmc = cmc * (100.0 / count)
    if not get_mAP:
        return cmc, indices[:, :min(max_rank, indices.shape[1])]
    return cmc, ap * (100.0 / count), inp * (100.0 / count), indices


def retrieval_metrics(query_features, gallery_features, query_ids, gallery_ids,
                      chunk_size=256):
    """Compute metrics without materializing the full query-by-gallery matrix."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    query_features = torch.as_tensor(query_features).detach().float().cpu()
    gallery_features = torch.as_tensor(gallery_features).detach().float().cpu()
    if query_features.ndim != 2 or gallery_features.ndim != 2:
        raise ValueError("Retrieval features must be matrices")
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("Query and gallery feature dimensions differ")
    if len(query_features) == 0 or len(gallery_features) == 0:
        raise ValueError("Cannot evaluate an empty query/gallery")
    if not torch.isfinite(query_features).all() or not torch.isfinite(gallery_features).all():
        raise ValueError("Retrieval features must be finite")
    query_ids = torch.as_tensor(query_ids).view(-1).cpu()
    gallery_ids = torch.as_tensor(gallery_ids).view(-1).cpu()
    if len(query_features) != len(query_ids) or len(gallery_features) != len(gallery_ids):
        raise ValueError("Feature and identity counts differ")
    query_features = F.normalize(query_features, dim=1)
    gallery_features = F.normalize(gallery_features, dim=1)
    cmc_sum, ap_sum, inp_sum, valid_count = torch.zeros(10), 0.0, 0.0, 0
    for start in range(0, len(query_ids), chunk_size):
        stop = start + chunk_size
        stats = _rank_statistics(query_features[start:stop] @ gallery_features.T,
                                 query_ids[start:stop], gallery_ids, 10)
        cmc_sum += stats[0]
        ap_sum += float(stats[1])
        inp_sum += float(stats[2])
        valid_count += stats[3]
    if not valid_count:
        raise ValueError("No query identity has a match in the gallery")
    cmc = cmc_sum * (100.0 / valid_count)
    return {"R1": float(cmc[0]), "R5": float(cmc[4]), "R10": float(cmc[9]),
            "mAP": ap_sum * 100.0 / valid_count,
            "mINP": inp_sum * 100.0 / valid_count,
            "num_queries": len(query_ids), "num_gallery": len(gallery_ids),
            "num_valid_queries": valid_count, "num_ignored_queries": len(query_ids) - valid_count}


class Evaluator:
    def __init__(self, img_loader, txt_loader, chunk_size=256):
        self.img_loader, self.txt_loader = img_loader, txt_loader
        self.chunk_size = chunk_size
        self.logger = logging.getLogger("NCHNet.eval")

    def _compute_embedding(self, model):
        model = getattr(model, "module", model)
        was_training = model.training
        model.eval()
        try:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            qids, gids, qfeats, gfeats = [], [], [], []
            with torch.inference_mode():
                for batch in self.txt_loader:
                    if isinstance(batch, dict):
                        pid, captions = batch["pids"], batch["caption_ids"]
                        mask = batch.get("attention_mask")
                    else:
                        pid, captions, *rest = batch
                        mask = rest[0] if rest else None
                    captions = captions.to(device)
                    mask = mask.to(device) if mask is not None else None
                    feat = model.encode_text(captions, attention_mask=mask) if mask is not None else model.encode_text(captions)
                    qids.append(torch.as_tensor(pid).view(-1).cpu())
                    qfeats.append(feat.float().cpu())
                for batch in self.img_loader:
                    if isinstance(batch, dict):
                        pid, images = batch["pids"], batch["images"]
                    else:
                        pid, images = batch
                    feat = model.encode_image(images.to(device))
                    gids.append(torch.as_tensor(pid).view(-1).cpu())
                    gfeats.append(feat.float().cpu())
            if not qfeats or not gfeats:
                raise ValueError("Cannot evaluate empty text/image loaders")
            return torch.cat(qfeats), torch.cat(gfeats), torch.cat(qids), torch.cat(gids)
        finally:
            model.train(was_training)

    def evaluate(self, model, i2t_metric=False):
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        metrics = retrieval_metrics(qfeats, gfeats, qids, gids, self.chunk_size)
        self.logger.info("t2i R1 %.3f R5 %.3f R10 %.3f mAP %.3f mINP %.3f (%d ignored queries)",
                         metrics["R1"], metrics["R5"], metrics["R10"], metrics["mAP"],
                         metrics["mINP"], metrics["num_ignored_queries"])
        if i2t_metric:
            metrics["i2t"] = retrieval_metrics(gfeats, qfeats, gids, qids, self.chunk_size)
        return metrics

    def eval(self, model, i2t_metric=False):
        return self.evaluate(model, i2t_metric)["R1"]
