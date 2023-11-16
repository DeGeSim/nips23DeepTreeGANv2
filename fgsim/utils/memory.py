import gc
import pprint
from typing import Optional

import torch

from fgsim.config import device
from fgsim.monitoring import logger


class GpuMemMonitor:
    """
    Context manager methods:
    ```
    manager = GpuMemMonitor()
    with manager("batch"):
        batch.to(device)
    manager.print_recorded()
    ```
    `print_current_state`: print reserved/allocated/total gpu memory
    `print_delta`: print reserved/allocated gpu memory delta
    """

    def __init__(self):
        self.gpu_mem_res = torch.cuda.memory_reserved(device)
        self.gpu_mem_alloc = torch.cuda.memory_allocated(device)
        self.sizes = {}
        self.lastvar: Optional[str] = None
        self.overwrite: bool = True

    def _update_mem(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.gpu_mem_res = torch.cuda.memory_reserved(device)
        self.gpu_mem_alloc = torch.cuda.memory_allocated(device)

    def __call__(self, varname: str, overwrite: bool = True):
        if self.lastvar is not None:
            raise RuntimeError("Already recording memory footprint.")
        self.lastvar = varname
        self.overwrite = overwrite
        self._update_mem()
        return self

    def __enter__(self):
        assert self.lastvar is not None, (
            "Call this with the name of the variable the memory change should be"
            " assigned to."
        )
        return self

    def __exit__(self, *args, **kwargs):
        if self.overwrite or self.lastvar not in self.sizes:
            cur_alloc = torch.cuda.memory_allocated(device)
            self.sizes[self.lastvar] = cur_alloc - self.gpu_mem_alloc
        self.lastvar = None

    def print_recorded(self):
        self.sizes = dict(sorted(self.sizes.items(), key=lambda x: x[1]))

        logger.warning(
            pprint.pformat({key: f"{val:.2E}" for key, val in self.sizes.items()})
        )
        sizes_sum = sum(self.sizes.values())
        self._update_mem()
        frac_recorded = sizes_sum / self.gpu_mem_alloc
        logger.warning(
            f"Fraction recorded {frac_recorded*100:.0f}%, missing"
            f" {self.gpu_mem_alloc-frac_recorded:.2E}"
        )

    def print_current_state(self, msg: Optional[str] = None):
        self._update_mem()
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = self.gpu_mem_res
        allocated = self.gpu_mem_alloc
        if msg is None:
            msg = ""
        else:
            msg = msg + " "
        logger.info(
            f"{msg[:25]:>25} GPU Memory: reserved {reserved:.2E} allocated"
            f" {allocated:.2E} avail {reserved-allocated:.2E} total {total:.2E}"
        )

    def print_delta(self, msg: Optional[str] = None):
        gc.collect()
        torch.cuda.empty_cache()
        cur_reserved = torch.cuda.memory_reserved(device)
        cur_alloc = torch.cuda.memory_allocated(device)

        if msg is None:
            msg = ""
        else:
            msg = msg + " "

        logger.info(
            f"{msg[:15]:>15} GPU Memory Δ: reserved"
            f" {cur_reserved-self.gpu_mem_res:+.2E} allocated"
            f" {cur_alloc-self.gpu_mem_alloc:+.2E}"
        )
        self.gpu_mem_res = cur_reserved
        self.gpu_mem_alloc = cur_alloc


gpu_mem_monitor = GpuMemMonitor()
