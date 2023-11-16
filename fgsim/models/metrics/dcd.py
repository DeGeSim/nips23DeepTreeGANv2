import torch


# https://github.com/wutong16/Density_aware_Chamfer_Distance/blob/main/utils_v2/model_utils.py
# https://arxiv.org/abs/2111.12702v1
def dcd(x, gt, alpha=1, lpnorm=2.0, pow=1.0, n_lambda=1, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    dist1, dist2, idx1, idx2 = distChamfer(x, gt, lpnorm=lpnorm, pow=pow)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    assert torch.all(dist1 >= 0)
    assert torch.all(dist2 >= 0)
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)
    if torch.any(loss1 < 0):
        raise Exception

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)
    if torch.any(loss2 < 0):
        raise Exception
    return (loss1 + loss2) / 2


def cd(output, gt, lpnorm=2.0, pow: float = 1.0):
    dist1, dist2, _, _ = distChamfer(gt, output, lpnorm=lpnorm, pow=pow)
    cd_p = (dist1.mean(1) + dist2.mean(1)) / 2
    return cd_p
    # cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    # cd_t = dist1.mean(1) + dist2.mean(1)

    # # , calc_f1=False, separate=False
    # if separate:
    #     res = [
    #         torch.cat(
    #             [
    #                 torch.sqrt(dist1).mean(1).unsqueeze(0),
    #                 torch.sqrt(dist2).mean(1).unsqueeze(0),
    #             ]
    #         ),
    #         torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)]),
    #     ]
    # else:
    #     res = [cd_p, cd_t]
    # if calc_f1:
    #     f1, _, _ = fscore(dist1, dist2)
    #     res.append(f1)
    # return res


def distChamfer(a, b, lpnorm=2.0, pow=1.0):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    # original implementation equivalent to
    P = torch.pow(torch.cdist(a, b, p=lpnorm), pow)
    # assert torch.all(P >= 0)
    return (
        torch.min(P, 2)[0],
        torch.min(P, 1)[0],
        torch.min(P, 2)[1].int(),
        torch.min(P, 1)[1].int(),
    )


def fscore(dist1, dist2, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud
    # euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2
