from collections import OrderedDict, deque

import numpy as np


class MetricAggregator:
    def __init__(self) -> None:
        self.__metric_collector = OrderedDict()

    def aggregate(self):
        aggr_dict = {k: np.mean(v) for k, v in self.__metric_collector.items()}
        self.__metric_collector = OrderedDict()
        return aggr_dict

    def append_dict(self, upd):
        # Make sure the fields in the state are available
        for metric_name in upd:
            if metric_name not in self.__metric_collector:
                self.__metric_collector[metric_name] = deque()
            # Write values to the state
            self.__metric_collector[metric_name].append(upd[metric_name])


class GradHistAggregator:
    def __init__(self) -> None:
        self.__grad_collector = OrderedDict()
        self.__weigth_collector = OrderedDict()
        self.grad_history = OrderedDict()
        self.weigth_history = OrderedDict()
        self.steps = deque()
        # self.max_memory = 10

    def aggregate(self, step):
        aggr_dict = {k: np.mean(v) for k, v in self.__grad_collector.items()}
        self.append_dict_(self.grad_history, aggr_dict)

        aggr_dict = {k: np.mean(v) for k, v in self.__weigth_collector.items()}
        self.append_dict_(self.weigth_history, aggr_dict)

        self.steps.append(step)

        self.__grad_collector = OrderedDict()
        return self.weigth_history, self.grad_history

    def append_dict(self, upd):
        self.append_dict_(self.__weigth_collector, upd["weights"])
        self.append_dict_(self.__grad_collector, upd["grads"])

    def append_dict_(self, target, upd):
        # Make sure the fields in the state are available
        for metric_name in upd:
            if metric_name not in target:
                target[metric_name] = deque()
            # Write values to the state
            target[metric_name].append(upd[metric_name])

    # def compress_history(self):
    #     for k, v in self.history.items():
    #         self.history[k] = deque(
    #             list(np.array(v).reshape(2, -1)[0, -1].reshape(-1)),
    #  self.max_memory
    #         )
