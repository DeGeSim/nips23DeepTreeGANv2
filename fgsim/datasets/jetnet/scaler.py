from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler,
)

from fgsim.config import conf
from fgsim.io import ScalerBase
from fgsim.io.dequantscaler import dequant_stdscale

from .graph_transform import events_to_batch_unscaled
from .readin import file_manager, read_chunks


def Identity(x):
    return x


def DummyTransformer():
    return FunctionTransformer(Identity, Identity)


scaler = ScalerBase(
    files=file_manager.files,
    len_dict=file_manager.file_len_dict,
    transfs_x=[
        StandardScaler(),
        StandardScaler(),
        PowerTransformer(method="box-cox", standardize=True),
    ],
    transfs_y=[
        MinMaxScaler((-1, 1)),  # type
        StandardScaler(),  # pt
        StandardScaler(),  # eta
        StandardScaler(),  # mass
        make_pipeline(
            *dequant_stdscale((0, conf.loader.n_points + 1))
        ),  # num_particles
    ],
    read_chunk=read_chunks,
    events_to_batch=events_to_batch_unscaled,
)
