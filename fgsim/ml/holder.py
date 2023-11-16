"""This modules manages all objects that need to be available for the training:
Subnetworks, losses and optimizers. The Subnetworks and losses are dynamically
imported, depending on the config. Contains the code for checkpointing of model
and optimzer status."""


from collections import defaultdict
from contextlib import contextmanager
from glob import glob

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.ml.eval_metrics import EvaluationMetrics
from fgsim.ml.loss import LossesCol
from fgsim.ml.network import SubNetworkCollector
from fgsim.ml.optim import OptimAndSchedulerCol
from fgsim.monitoring import TrainLog, logger
from fgsim.utils import check_tensor

from .checkpoint import CheckPointManager


class Holder:
    """ "This class holds the models, the loss functions and the optimizers.
    It manages the checkpointing and holds a member 'state' that contains
    information about the current state of the training"""

    # Nameing convention snw = snw
    def __init__(self, device=torch.device("cpu")) -> None:
        self.device = device
        # Human readable, few values
        self.state: DictConfig = OmegaConf.create(
            {
                "epoch": 0,
                "processed_events": 0,
                "grad_step": 0,
                "complete": False,
            }
        )
        self.history = {
            "losses": {snwname: defaultdict(list) for snwname in conf.models},
            "val": defaultdict(list),
        }
        self.train_log = TrainLog(self.state)
        self.checkpoint_manager = CheckPointManager(self)

        self.models: SubNetworkCollector = SubNetworkCollector(conf.models)
        # For SWA the models need to be on the right device before initializing
        self.models = self.models.float().to(device)
        # if not conf.debug:
        #     self.models = torch.compile(self.models)
        # self.train_log.log_model_graph(self.models)
        self.swa_models = {
            k: AveragedModel(v)
            for k, v in self.models.parts.items()
            if conf.models[k].scheduler.name == "SWA"
        }

        if conf.command in ["train", "implant_checkpoint"]:
            self.optims: OptimAndSchedulerCol = OptimAndSchedulerCol(
                conf.models, self.models, self.swa_models, self.train_log
            )

        # try to load a check point
        self.checkpoint_loaded = False

        if (
            conf.command != "train" or not conf.debug
        ) and conf.command != "implant_checkpoint":
            if conf.ray:
                self.checkpoint_manager.load_ray_checkpoint(
                    sorted(glob(f"{conf.path.run_path}/checkpoint_*"))[-1]
                )
            else:
                self.checkpoint_manager.load_checkpoint()

        # # Hack to move the optim parameters to the correct device
        # # https://github.com/pytorch/pytorch/issues/8741
        # with gpu_mem_monitor("optims"):
        #     self.optims.load_state_dict(self.optims.state_dict())

        logger.warning(f"Starting with state {str(OmegaConf.to_yaml(self.state))}")

        # Keep the generated samples ready, to be accessed by the losses
        self.gen_points: Batch = None
        self.gen_points_w_grad: Batch = None

        # import torcheck
        # for partname, model in self.models.parts.items():
        #     torcheck.register(self.optims[partname])
        #     torcheck.add_module_changing_check(model, module_name=partname)
        #     # torcheck.add_module_inf_check(model, module_name=partname)
        #     # torcheck.add_module_nan_check(model, module_name=partname)

        self.losses: LossesCol = LossesCol(self.train_log)
        self.eval_metrics = EvaluationMetrics(self.train_log, self.history)

        self.to(self.device)

    def to(self, device):
        self.device = device
        self.models = self.models
        self.swa_models = {
            k: AveragedModel(v).to(device)
            for k, v in self.models.parts.items()
            if conf.models[k].scheduler.name == "SWA"
        }
        if conf.command == "train":
            self.optims.to(device)

        return self

    def generate(self, cond: torch.Tensor, n_pointsv: torch.Tensor):
        check_tensor(cond, n_pointsv)

        self.models.eval()

        gen = self.models.gen
        if self.state["epoch"] > 10:
            if conf["models"]["gen"]["scheduler"]["name"] == "SWA":
                gen = self.swa_models["gen"]

        # generate the random vector
        z = torch.randn(
            *self.models.gen.z_shape,
            requires_grad=True,
            dtype=torch.float,
            device=self.device,
        )

        with torch.no_grad():
            z.requires_grad = False
            gen_batch = gen(z, cond, n_pointsv)
        return gen_batch

    def _check_gen_batch(self, gen_batch, sim_batch):
        check_tensor(gen_batch.x)
        assert not torch.isnan(gen_batch.x).any()
        assert sim_batch.x.shape[-1] == gen_batch.x.shape[-1]
        if conf.models.gen.params.sample_until_full:
            assert (gen_batch.ptr == sim_batch.ptr).all()
        else:
            assert (gen_batch.ptr >= sim_batch.ptr).all()

    def _check_sim_batch(self, sim_batch):
        assert not torch.isnan(sim_batch.x).any()
        assert sim_batch.y.shape[-1] == len(conf.loader.y_features)
        assert sim_batch.x.shape[-1] == len(conf.loader.x_features)
        check_tensor(sim_batch.x, sim_batch.y)
        assert torch.allclose(
            sim_batch.n_pointsv.long(), (sim_batch.ptr[1:] - sim_batch.ptr[:-1])
        )

    def pass_batch_through_model(
        self,
        sim_batch,
        train_gen: bool = False,
        train_disc: bool = False,
        eval=False,
    ):
        assert not (train_gen and train_disc)
        assert not (eval and (train_gen or train_disc))
        self._check_sim_batch(sim_batch)

        batch_size = conf.loader.batch_size
        cond_gen_features = conf.loader.cond_gen_features
        cond_critic_features = conf.loader.cond_critic_features
        if eval:
            self.models.eval()
        else:
            self.models.train()

        gen = self.models.gen
        disc = self.models.disc
        if self.state["epoch"] > 10 and eval:
            if conf["models"]["gen"]["scheduler"]["name"] == "SWA":
                gen = self.swa_models["gen"]

            if conf["models"]["disc"]["scheduler"]["name"] == "SWA":
                disc = self.swa_models["disc"]

        # generate the random vector
        z = torch.randn(
            *self.models.gen.z_shape,
            requires_grad=True,
            dtype=torch.float,
            device=self.device,
        )

        cond_gen = torch.empty((batch_size, 0)).float().to(self.device)
        cond_critic = torch.empty((batch_size, 0)).float().to(self.device)
        if sum(cond_gen_features) > 0:
            cond_gen = sim_batch.y[..., cond_gen_features]
        if sum(cond_critic_features) > 0:
            cond_critic = sim_batch.y[..., cond_critic_features]

        with with_grad(train_gen):
            z.requires_grad = train_gen
            gen_batch = gen(z, cond_gen, sim_batch.n_pointsv)

        self._check_gen_batch(gen_batch, sim_batch)

        # if train_gen or train_disc:
        gen_batch = self.postprocess(gen_batch)
        # sim_batch = self.postprocess(sim_batch)
        res = {"sim_batch": sim_batch, "gen_batch": gen_batch}

        # In both cases the gradient needs to pass though gen_crit
        with with_grad(train_gen or train_disc):
            res |= prepend_to_key(
                disc(self.disc_preprocess(gen_batch), cond_critic), "gen_"
            )

        assert not torch.isnan(res["gen_crit"]).any()

        # we dont need to compute sim_crit if only the generator is trained
        # but we need it for the validation
        # and for the feature matching loss
        if (
            train_disc
            or (train_disc == train_gen)
            or ("feature_matching" in conf.models.gen.losses and train_gen)
        ):
            with with_grad(train_disc):
                sim_batch_disc_input = self.disc_preprocess(sim_batch)
                res |= prepend_to_key(
                    disc(sim_batch_disc_input, cond_critic), "sim_"
                )
        return res

    def disc_preprocess(self, batch: Batch) -> Batch:
        # rotate alpha without changing the distribution for the evaulation
        batch = batch.clone()
        # if conf.dataset_name == "calochallange":
        #     from fgsim.datasets.calochallange.alpharot import rotate_alpha

        #     alphapos = conf.loader.x_features.index("alpha")
        #     batch.x[..., alphapos] = rotate_alpha(
        #         batch.x[..., alphapos].clone(), batch.batch, center=True
        #     )
        return batch

    def postprocess(self, batch: Batch) -> Batch:
        if conf.dataset_name == "jetnet" and conf.loader.n_points == 150:
            from fgsim.datasets.jetnet.post_gen_transfrom import norm_pt_sum

            pt_pos = conf.loader.x_ftx_energy_pos
            pts = batch.x[..., pt_pos].clone()
            batch.x[..., pt_pos] = norm_pt_sum(pts, batch.batch).clone()

        return batch


def prepend_to_key(d: dict, s: str) -> dict:
    return {s + k: v for k, v in d.items()}


@contextmanager
def with_grad(condition):
    if not condition:
        with torch.no_grad():
            yield
    else:
        yield
