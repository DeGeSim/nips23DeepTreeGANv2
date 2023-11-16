import hashlib
from typing import List

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError


# Exclude the keys that do not affect the training
def removekeys(omconf: DictConfig, excluded_keys: List[str]) -> DictConfig:
    filtered_omconf = OmegaConf.masked_copy(
        omconf,
        [k for k in omconf.keys() if k not in excluded_keys],
    )
    return filtered_omconf


def dict_to_kv(o, keystr=""):
    """Converts a nested dict {"a":"foo", "b": {"foo":"bar"}} to \
    [("a","foo"),("b.foo","bar")]."""
    if hasattr(o, "keys"):
        outL = []
        for k in o.keys():
            elemres = dict_to_kv(o[k], keystr + str(k) + ".")
            if (
                len(elemres) == 2
                and type(elemres[0]) == str
                and type(elemres[1]) == str
            ):
                outL.append(elemres)
            else:
                for e in elemres:
                    outL.append(e)
        return outL
    elif hasattr(o, "__str__"):
        return (keystr.strip("."), str(o))
    else:
        raise ValueError


def dict_to_keylist(o, keystr=""):
    """Converts a nested dict {"a":"foo", "b": {"foo":"bar"}} to \
    ["a","b","b/foo"]."""
    if hasattr(o, "keys"):
        outL = []
        for k in o.keys():
            subkeystr = f"{keystr}/{k}"
            outL.append(subkeystr)
            try:
                elemres = dict_to_keylist(o[k], subkeystr)
            except InterpolationKeyError:
                elemres = []
            for e in elemres:
                outL.append(e)
        return outL
    return []


# convert the config to  key-value pairs
# sort them, hash the results
def gethash(omconf: DictConfig) -> str:
    OmegaConf.resolve(omconf)
    kv_list = [f"{e[0]}: {e[1]}" for e in dict_to_kv(omconf)]
    kv_str = "\n".join(sorted(kv_list))
    omhash = str(hashlib.sha1(kv_str.encode()).hexdigest()[:7])
    return omhash


# recurrsive function that calculates the needed update for base
def compute_update_config(base, updated):
    keys_base = set(base.keys())
    keys_updated = set(updated.keys())
    keys_removed = keys_base - keys_updated
    assert len(keys_removed) == 0
    keys_both = keys_base & keys_updated
    keys_added = keys_updated - keys_base
    if keys_added != set():
        print(keys_added)
    out_dict = {k: updated[k] for k in keys_added}
    for k in keys_both:
        if isinstance(base[k], DictConfig):
            if OmegaConf.is_interpolation(base, k):
                continue
            assert isinstance(updated[k], DictConfig)
            subdiff = compute_update_config(base[k], updated[k])
            # dont add empty dicts
            if len(subdiff.keys()):
                out_dict[k] = subdiff
        else:
            if base[k] != updated[k]:
                out_dict[k] = updated[k]
    return out_dict
