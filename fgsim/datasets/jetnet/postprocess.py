from jetnet.utils import jet_features

from fgsim.config import conf
from fgsim.datasets.jetnet.utils import to_efp, to_stacked_mask


def postprocess(batch, sim_or_gen: str):
    metrics = conf.metrics.val if conf.command == "train" else conf.metrics.test
    if len({"kpd", "fpd"} & set(metrics)) and "efps" not in batch.keys:
        batch["efps"] = to_efp(batch)
    if "hlv" not in batch:
        batch["hlv"] = {}
    jn_dict = jet_features(to_stacked_mask(batch).cpu().numpy()[..., :3])
    for k, v in jn_dict.items():
        batch["hlv"][k] = v
    return batch
