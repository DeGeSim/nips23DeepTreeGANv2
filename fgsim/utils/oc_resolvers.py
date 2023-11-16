from omegaconf import DictConfig, ListConfig, OmegaConf


# Add a custum resolver to OmegaConf allowing for divisions
# Give int back if you can:
def divide(numerator, denominator):
    if numerator // denominator == numerator / denominator:
        return numerator // denominator
    else:
        return numerator / denominator


def prod(p1, p2):
    return p1 * p2


def optionlist(options, ol):
    return DictConfig({item: options[item] for item in ol})


def merge(*configs):
    return OmegaConf.merge(*configs)


def oc_sum(*configs):
    return sum(*configs)


def mergedefault(config: DictConfig, sel: str, key: str) -> DictConfig:
    assert "default" in config
    assert key in config["default"]
    if key in config[sel]:
        return OmegaConf.merge(config["default"][key], config[sel][key])
    else:
        return OmegaConf.merge(config["default"][key])


def listadd(*configs):
    out = ListConfig([])
    for e in configs:
        out = out + e
    return out


def register_resolvers():
    OmegaConf.register_new_resolver("div", divide, replace=True)
    OmegaConf.register_new_resolver("prod", prod, replace=True)
    OmegaConf.register_new_resolver("sum", oc_sum, replace=True)
    OmegaConf.register_new_resolver("optionlist", optionlist, replace=True)
    OmegaConf.register_new_resolver("mergedefault", mergedefault, replace=True)
    OmegaConf.register_new_resolver("merge", merge, replace=True)
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("listadd", listadd, replace=True)
