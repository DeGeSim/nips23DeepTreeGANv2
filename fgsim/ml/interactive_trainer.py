from IPython import display
from tqdm import tqdm

from fgsim.commands.training import Trainer
from fgsim.config import conf
from fgsim.ml.holder import Holder
from fgsim.plot.xyscatter import xy_hist


class InteractiveTrainer(Trainer):
    def __init__(self, holder: Holder) -> None:
        super().__init__(holder)

    def train_epoch(self):
        self.pre_epoch()
        istep_start = (
            self.holder.state.processed_events
            // conf.loader.batch_size
            % self.loader.n_grad_steps_per_epoch
        )
        for batch in tqdm(
            self.loader.qfseq,
            initial=istep_start,
            total=self.loader.n_grad_steps_per_epoch,
            miniters=20,
            desc=f"Epoch {self.holder.state.epoch}",
        ):
            batch = self.pre_training_step(batch)
            res = self.training_step(batch)
            self.post_training_step()
            self.eval_step(res)

    def eval_step(self, res):
        v1 = 0
        v2 = 1
        sim = res["sim_batch"].x[:, [v1, v2]].cpu().detach().numpy()
        gen = res["gen_batch"].x[:, [v1, v2]].cpu().detach().numpy()
        fig = xy_hist(
            sim=sim,
            gen=gen,
            title="Single Training Batch",
            step=self.holder.state.grad_step,
            v1name=conf.loader.x_features[v1],
            v2name=conf.loader.x_features[v2],
        )

        display.clear_output(wait=True)
        display.display(fig)
