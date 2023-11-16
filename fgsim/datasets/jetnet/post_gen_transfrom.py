import torch
from torch_scatter import scatter_add

from fgsim.utils import check_tensor

from .scaler import scaler


def norm_pt_sum(pts, batchidx):
    pt_scaler = scaler.transfs_x[2]

    assert pt_scaler.method == "box-cox"
    assert pt_scaler.standardize
    # get parameters for the backward tranformation
    lmbd = pt_scaler.lambdas_[0]
    mean = pt_scaler._scaler.mean_[0]
    scale = pt_scaler._scaler.scale_[0]

    # Backwards transform
    pts = pts.clone().double() * scale + mean
    check_tensor(pts)
    if lmbd == 0:
        pts = torch.exp(pts.clone())
    else:
        pts = torch.pow(pts.clone() * lmbd + 1, 1 / lmbd)
    check_tensor(pts)

    # Norm
    ptsum_per_batch = scatter_add(pts, batchidx, dim=-1)
    pts = pts / ptsum_per_batch[batchidx]
    check_tensor(pts)

    # Forward transform
    if lmbd == 0:
        pts = torch.log(pts.clone())
    else:
        pts = (torch.pow(pts.clone(), lmbd) - 1) / lmbd

    pts = (pts.clone() - mean) / scale
    check_tensor(pts)
    return pts.float()
