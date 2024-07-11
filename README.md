# DeepTreeGANv2
Repository for https://arxiv.org/abs/2312.00042 


Predecessor Model:
[arxiv](https://arxiv.org/abs/2311.12616) [Repository](https://github.com/DeGeSim/chep23DeepTreeGAN)


## Usage


Setup the enviroment:

~~~bash
$ bash setup_venv.sh
$ source venv/bin/activate
~~~

Run the testing:
~~~bash
$ bash setup_venv.sh
$ python -m fgsim --hash af5e229 train # top quarks
~~~

~~~
$ python -m fgsim --hash af5e229 test
23-11-13 19:30 - WARNING - Replaced path /home/mscham/fgsim with /home/mscham/fgsim/wd/scan_cdr5_fullcopy/af5e229.
23-11-13 19:30 - INFO - tag: scan_cdr5_fullcopy hash: af5e229 loader_hash: fb94d80
23-11-13 19:30 - INFO - Running command test
23-11-13 19:30 - WARNING - Loaded model from checkpoint at epoch 3857 grad_step 2217838.
23-11-13 19:30 - WARNING - Starting with state epoch: 3857
processed_events: 443570000
grad_step: 2217838
complete: false
best_step: 1632000
best_epoch: 2838
time_train_step_start: 1689954732.8672411
time_io_end: 1689954732.8707848
time_train_step_end: 1689954733.5023878

23-11-13 19:30 - INFO - Loading test dataset from wd/scan_cdr5_fullcopy/af5e229/test_best/testdata.pt
23-11-13 19:30 - INFO - Evalutating best dataset
23-11-13 19:31 - INFO - Metric w1p took 18.4957 sec
23-11-13 19:33 - INFO - Metric w1efp took 157.862564 sec
23-11-13 19:34 - INFO - Metric fpnd took 22.33024 sec
23-11-13 19:34 - INFO - {'w1m': (1.447208831563592, 0.1304301255259286), 'w1p': (0.1854726515637384, 0.02251938614739039), 'w1efp': (4.575847330328108, 0.20128979462176394), 'fpnd': (0.10054866184640332, nan), 'kpd': (-0.0057607056005437585, 0.014783962010685103), 'fpd': (0.33844204006553613, 0.06506682857144748)}
~~~


Retrain the models:
~~~bash
$ python -m fgsim --tag t_retrain setup
> Experiment setup with hash 8dea68a.
$ python -m fgsim --hash 8dea68a train
>
23-09-20 15:39 INFO   tag: t_retrain hash: 8dea68a loader_hash: 0d09873
           INFO   Running command train
           WARNING  Proceeding without loading checkpoint.
           WARNING  Starting with state epoch: 0
                processed_events: 0
                grad_step: 0
                complete: false
           INFO   Using the first 50 batches for validation and the next 250 batches for testing.
           INFO   Device: Tesla V100-SXM2-32GB
           INFO   Validating
Generating eval batches: 100%|████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  8.04it/s]
           INFO   Postprocessing
           INFO   Postprocessing done
           INFO   w1m 118.82    w1p 42.76     fpnd 198.90   auc 0.03      w1disc 1.24
           WARNING  New best model at step 0
WARNING:fgsim:New best model at step 0
Epoch 0:   4%|████▏                                                                   | 589/14725 [00:51<20:30, 11.49it/s]
Epoch 1:   8%|████████▎                                                               | 1178/14725 [00:46<17:50, 12.65it/s]
Epoch 2:  12%|████████████▍                                                           | 1767/14725 [00:44<16:08, 13.37it/s]
Epoch 3:  14%|██████████████
~~~

## Authors

Moritz Scham is funded by Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI (grant number: ZT-I-PF-5-3).
