import numpy as np
from sklearn.linear_model import LinearRegression

from fgsim.config import conf
from fgsim.monitoring import logger


def early_stopping(holder) -> bool:
    history = holder.history
    """
    If in the last `valsteps` the stopping metric have shown no
    improvement(=decay), return True, else False.

    Args:
      state (DictConfig): DictConfig

    Returns:
      A boolean value.
    """
    if conf.debug:
        return False
    # Make sure some metrics have been recorded
    if len(history["val"].keys()) == 0:
        return False
    # collect at least two values before evaluating the criteria
    # if len(history["stop_crit"]) < 2:
    #     return False
    # loss_arrs = []
    # for k in conf.models.keys():  # iterate the models
    #     model_loss_dict = history["losses"][k]
    #     if len(model_loss_dict) == 0:
    #         break
    #     # first we need to find out how many times losses
    #     # have been recorded for this model
    #     for lname, lossd in model_loss_dict.items():
    #         if isinstance(lossd, dict):
    #             for _, sublossd in lossd.items():
    #                 assert isinstance(sublossd, list)
    #                 n_recorded_losses = len(sublossd)
    #                 break
    #         else:
    #             assert isinstance(lossd, list)
    #             n_recorded_losses = len(lossd)
    #             break
    #         break
    #     # elementwise add the loss history for each of the losses for a model
    #     lsum = np.zeros(n_recorded_losses)
    #     # sum over the losses for the model
    #     for lname in model_loss_dict:
    #         lsum += histdict_to_np(model_loss_dict[lname])
    #     # apped the summed loss array for the current model to the list
    #     loss_arrs.append(lsum)

    return all(
        [
            is_minimized(np.array(history["val"][metric]))
            for metric in conf.metrics.stopping
        ]
    )


def histdict_to_np(histd):
    # this is to take care of composite losses like CEDiscLoss
    if isinstance(histd, dict):
        return np.array(histd["sum"])
    elif isinstance(histd, list):
        return np.array(histd)
    else:
        raise RuntimeError


def is_not_dropping(arr: np.ndarray):
    valsteps = conf.training.early_stopping.validation_steps
    subarr = arr[-valsteps:]
    # subm = np.mean(subarr)
    # subarr = np.(subarr - subm) / subm
    reg = LinearRegression()
    reg.fit(
        X=np.arange(len(subarr)).reshape(-1, 1),
        y=subarr.reshape(-1),
    )
    return reg.coef_[-1] > conf.training.early_stopping.improvement


def is_minimized(arr: np.ndarray):
    valsteps = conf.training.early_stopping.validation_steps
    stop_metric = np.array(arr)

    if len(stop_metric) < valsteps + 1:
        return False
    growth = 1 - (min(stop_metric[-valsteps:]) / min(stop_metric[:-valsteps]))

    logger.info(
        f"Relative Improvement in the last {valsteps} validation steps: {growth}%"
    )
    if growth < conf.training.early_stopping.improvement:
        return True
    else:
        return False
