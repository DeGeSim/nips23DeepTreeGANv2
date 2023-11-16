from ..config import conf

if conf.dataset_name == "jetnet":
    from .jetnet import Dataset, postprocess, scaler
elif conf.dataset_name == "calochallange":
    from .calochallange import Dataset, postprocess, scaler
