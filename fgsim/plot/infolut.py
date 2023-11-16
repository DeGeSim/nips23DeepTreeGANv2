from typing import Dict, Optional, Union

import numpy as np

from fgsim.config import conf

labels_dict: Dict[str, str] = {
    "etarel": "$\\eta_\\mathrm{rel}$",
    "phirel": "$\\phi_\\mathrm{rel}$",
    "ptrel": "$p^\\mathrm{T}_\\mathrm{rel}$",
    "mass": "$m_{rel}$",
    "phi": "$\\sum  ϕ_{rel}$",
    "pt": "$\\sum p_{T}^{rel}$",
    "eta": "$\\sum η_{rel}$",
    "response": "Response ($\\textstyle\\sum\\nolimits_i \\mathrm{E_i}/E$)",
    "showershape_peak_layer": "Peak Layer",
    "showershape_psr": (
        "$\\textstyle \\frac{|\\mathrm{Layer}^\\mathrm{Peak}-"
        "\\mathrm{Layer}^\\mathrm{Turnoff}|+1}{|\\mathrm{Layer}^\\mathrm{Peak}"
        "-\\mathrm{Layer}^\\mathrm{Turnon}|+1}$"
    ),
    "showershape_turnon_layer": "Turnon Layer",
    "sphereratio_ratio": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.3 σ)\\mathrm{E_i} /"
        " \\sum\\nolimits_i^{\\mathrm{Sphere}(0.8 σ)}\\mathrm{E_i}$"
    ),
    "sphereratio_small": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.3 σ)} \\mathrm{E_i}$"
    ),
    "sphereratio_large": (
        "$\\sum\\nolimits_i^{\\mathrm{Sphere}(0.8 σ)} \\mathrm{E_i}$"
    ),
    "cyratio_ratio": (
        "$\\frac{\\sum\\nolimits_i^{\\mathrm{Cylinder}(0.2 σ)}"
        " \\mathrm{E_i}}{\\sum\\nolimits_i^{\\mathrm{Cylinder}(0.6 σ)}"
        "\\mathrm{E_i}}$"
    ),
    "cyratio_small": (
        "$\\sum\\nolimits_i^{\\mathrm{Cylinder}(0.2 σ)} \\mathrm{E_i}$"
    ),
    "cyratio_large": (
        "$\\sum\\nolimits_i^{\\mathrm{Cylinder}(0.6 σ)} \\mathrm{E_i}$"
    ),
    "fpc_x": "First PCA vector x",
    "fpc_y": "First PCA vector y",
    "fpc_z": "First PCA vector z",
    "fpc_eval": "First PCA Eigenvalue",
    "nhits_n": "Number of Hits",
    "nhits_n_by_E": "Number of Hits / Shower Energy",
}


def var_to_label(v: Union[str, int]) -> str:
    if isinstance(v, int):
        vname = conf.loader.x_features[v]
    else:
        vname = v
    if vname in labels_dict:
        return labels_dict[vname]
    else:
        return vname


def var_to_bins(v: Union[str, int]) -> Optional[np.ndarray]:
    if isinstance(v, int):
        vname = conf.loader.x_features[v]
    else:
        vname = v

    if (
        conf.dataset_name == "calochallange"
        and "calochallange2" in conf.loader.dataset_path
    ):
        from caloutils import calorimeter

        return {
            "E": np.linspace(0, 6000, 100 + 1) - 0.5,
            "z": np.linspace(0, calorimeter.num_z, calorimeter.num_z + 1) - 0.5,
            "alpha": (
                np.linspace(0, calorimeter.num_alpha, calorimeter.num_alpha + 1)
                - 0.5
            ),
            "r": np.linspace(0, calorimeter.num_r, calorimeter.num_r + 1) - 0.5,
        }[vname]
    return None
