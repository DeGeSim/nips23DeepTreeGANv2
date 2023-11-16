from fgsim.config import conf

from .auc import auc
from .dcd import cd, dcd
from .ft_w1 import ft_w1
from .w1disc import w1disc

if conf.dataset_name == "jetnet":
    from .jetnet import cov_mmd, fpd, fpnd, kpd, w1efp, w1m, w1p
elif conf.dataset_name == "calochallange":
    from .calo import (
        cyratio,
        fpc,
        marginal,
        marginalEw,
        nhits,
        response,
        showershape,
        sphereratio,
    )
