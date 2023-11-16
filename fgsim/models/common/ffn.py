import math
from typing import List, Optional

from torch import nn

from fgsim.config import conf
from fgsim.models.common.benno import WeightNormalizedLinear


class FFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_init_method: Optional[str] = None,
        activation: Optional[str] = None,
        norm: Optional[str] = None,
        dropout: Optional[float] = None,
        n_layers: Optional[int] = None,
        final_linear: bool = False,
        bias: bool = False,
        hidden_layer_size: Optional[int] = None,
        equallr: Optional[bool] = None,
        **kwargs
    ) -> None:
        if norm is None:
            norm = conf.ffn.norm
        if dropout is None:
            dropout = conf.ffn.dropout
        if dropout == 0:
            dropout = None
        if n_layers is None:
            n_layers = conf.ffn.n_layers

        if hidden_layer_size is None:
            hidden_layer_size = max(
                conf.ffn.hidden_layer_size, input_dim, output_dim
            )
        if activation is None:
            activation = conf.ffn.activation
        if weight_init_method is None:
            weight_init_method = conf.ffn.weight_init_method

        if equallr is None:
            equallr = conf.ffn.equallr

        if equallr and norm in ["spectral", "weight", "bwn"]:
            raise RuntimeError()

        def activation_function():
            return getattr(nn, activation)(
                **conf.ffn.activation_params[conf.ffn.activation]
            )

        super().__init__()
        # +2 for input and output
        features: List[int] = (
            [input_dim]
            + [hidden_layer_size] * (n_layers - 1)
            + [
                output_dim,
            ]
        )
        # to keep the std of 1, the last layer should not see a reduction
        # in dimensionality, because otherwise it

        self.seq = nn.Sequential()
        for ilayer in range(n_layers):
            if norm == "bwn":
                m = WeightNormalizedLinear(
                    features[ilayer],
                    features[ilayer + 1],
                    bias=bias,
                    scale=True,
                    init_scale=0.5,
                    init_factor=0.5,
                )
            else:
                if equallr:
                    m = EqualLinear(
                        features[ilayer], features[ilayer + 1], bias=bias
                    )
                else:
                    m = nn.Linear(features[ilayer], features[ilayer + 1], bias=bias)
                if norm == "spectral":
                    m = nn.utils.parametrizations.spectral_norm(m)
                elif norm == "weight":
                    m = nn.utils.weight_norm(m)

            self.seq.append(m)
            if ilayer == n_layers - 1 and final_linear:
                continue
            else:
                if dropout:
                    self.seq.append(nn.Dropout(dropout))
                if norm == "batchnorm":
                    self.seq.append(
                        nn.BatchNorm1d(
                            features[ilayer + 1],
                            affine=False,
                            track_running_stats=False,
                        )
                    )
                elif norm == "layernorm":
                    self.seq.append(nn.LayerNorm(features[ilayer + 1]))
                elif norm in ("none", "spectral", "weight", "bwn"):
                    pass
                else:
                    raise Exception
                self.seq.append(activation_function())

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init_method = weight_init_method
        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.bias = bias
        if not equallr:
            self.reset_parameters()

    def forward(self, x):
        oldshape = x.shape
        if len(oldshape) == 3:
            x = x.reshape(oldshape[0] * oldshape[1], oldshape[2])
        x = self.seq(x.clone())
        if len(oldshape) == 3:
            x = x.reshape(oldshape[0], oldshape[1], self.output_dim)
        return x

    def __repr__(self):
        return (
            f"FFN({self.input_dim}->{self.output_dim},n_layers={self.n_layers},"
            f"hidden_nodes={self.hidden_layer_size},activation={self.activation})"
        )

    def reset_parameters(self):
        self.seq.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init_method == "default":
                m.reset_parameters()
            elif self.weight_init_method == "kaiming_uniform_":
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=conf.ffn.activation_params["LeakyReLU"]["negative_slope"],
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                return
            elif self.weight_init_method == "xavier_uniform_":
                nonlinearity = {
                    "SELU": "selu",
                    "Sigmoid": "sigmoid",
                    "ReLU": "relu",
                    "LeakyReLU": "leaky_relu",
                    "Tanh": "tanh",
                    "GELU": "relu",
                }[self.activation]
                if nonlinearity == "leaky_relu":
                    getattr(nn.init, self.weight_init_method)(
                        m.weight,
                        gain=nn.init.calculate_gain(
                            nonlinearity,
                            conf.ffn.activation_params["LeakyReLU"][
                                "negative_slope"
                            ],
                        ),
                    )
                else:
                    getattr(nn.init, self.weight_init_method)(
                        m.weight, gain=nn.init.calculate_gain(nonlinearity)
                    )
            else:
                pass


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)  # 1
        del module._parameters[name]
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)  # 2

        return fn

    def __call__(self, module, input):  # 3
        weight = self.compute_weight(module)  # 4
        setattr(module, self.name, weight)  # 5


def equal_lr(module, name="weight"):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim, bias=bias)
        linear.weight.data.normal_()
        if bias:
            linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)
