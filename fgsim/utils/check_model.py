import torch

from fgsim.monitoring import logger


def is_anormal_tensor(inp: torch.Tensor) -> bool:
    return bool(torch.any(torch.isinf(inp)) or torch.any(torch.isinf(inp)))


def contains_nans(inp, string=""):
    if isinstance(inp, torch.Tensor):
        res = is_anormal_tensor(inp)
        return (res, string)
    elif hasattr(inp, "state_dict"):
        return contains_nans(inp.state_dict())
    elif hasattr(inp, "to_dict"):
        return contains_nans(inp.to_dict())
    elif hasattr(inp, "items"):
        for k, elem in inp.items():
            res, string = contains_nans(elem, str(k) + " " + string)
            if res:
                return (res, string)
        return (res, string)
    elif hasattr(inp, "__iter__"):
        for k, elem in enumerate(inp):
            res, string = contains_nans(elem, str(k) + " " + string)
            if res:
                return (res, string)
        return (res, string)
    elif isinstance(inp, (int, float)):
        return (False, string)
    elif inp is None:
        return (False, string)
    else:
        raise Exception


def check_chain_for_nans(chain):
    nan_detected = False
    # go backwards to the chain, if the model is fine, there
    # is no need to check anything else
    oldstr = ""
    for i, e in list(
        zip(
            range(len(chain)),
            chain,
        )
    )[::-1]:
        problem, element_name = contains_nans(e)
        if problem:
            nan_detected = True
            oldstr = element_name
            if i == 0:
                logger.error(
                    f"Nan in elem number {0} in the chain of type {type(chain[0])}"
                    f" {'concerning' if oldstr else ''} {oldstr}."
                )
                raise ValueError
        if not problem and nan_detected:
            logger.error(
                f"Nan in elem number {i+1} in the chain of type {type(chain[i+1])}"
                f" {'concerning' if oldstr else ''} {oldstr}."
            )
            raise ValueError


# import numpy as np

# check_chain_for_nans(
#     (
#         torch.tensor([2, np.nan]),
#         torch.tensor([2, 2]),
#         torch.tensor([2, 2]),
#     )
# )
#  should raise ERROR - Nan in elem number 0 in the
#  chain of type <class 'torch.Tensor'>  .
