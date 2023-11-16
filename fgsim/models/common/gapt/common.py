# import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class LinearNet(nn.Module):
    """
    Module for fully connected networks with leaky relu activations

    Args:
        layers (list): list with layers of the fully connected network,
          optionally containing the input and output sizes inside
          e.g. ``[input_size, ... hidden layers ..., output_size]``
        input_size (list): size of input, if 0 or unspecified, first element of `layers` will be
          treated as the input size
        output_size (list): size of output, if 0 or unspecified, last element of `layers` will be
          treated as the output size
        final_linear (bool): keep the final layer operation linear i.e. no normalization,
          no nonlinear activation.Defaults to False.
        leaky_relu_alpha (float): negative slope of leaky relu. Defaults to 0.2.
        dropout_p (float): dropout fraction after each layer. Defaults to 0.
        batch_norm (bool): use batch norm or not. Defaults to False.
        spectral_norm (bool): use spectral norm or not. Defaults to False.
    """

    def __init__(
        self,
        layers: list,
        input_size: int = 0,
        output_size: int = 0,
        final_linear: bool = False,
        leaky_relu_alpha: float = 0.2,
        dropout_p: float = 0,
        batch_norm: bool = False,
        spectral_norm: bool = False,
    ):
        super(LinearNet, self).__init__()

        self.final_linear = final_linear
        self.leaky_relu_alpha = leaky_relu_alpha
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(p=dropout_p)

        layers = layers.copy()

        if input_size:
            layers.insert(0, input_size)
        if output_size:
            layers.append(output_size)

        self.net = nn.ModuleList()
        if batch_norm:
            self.bn = nn.ModuleList()

        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self.net.append(linear)
            if batch_norm:
                self.bn.append(nn.BatchNorm1d(layers[i + 1]))

        if spectral_norm:
            for i in range(len(self.net)):
                if i != len(self.net) - 1 or not final_linear:
                    self.net[i] = SpectralNorm(self.net[i])

    def forward(self, x: Tensor):
        """
        Runs input `x` through linear layers and returns output

        Args:
            x (Tensor): input tensor of shape ``[batch size, # input features]``
        """
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i != len(self.net) - 1 or not self.final_linear:
                x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
                if self.batch_norm:
                    x = self.bn[i](x)
            x = self.dropout(x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(net = {self.net})"


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class MAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_layers: list = [],
        layer_norm: bool = False,
        dropout_p: float = 0.0,
        final_linear: bool = True,
        linear_args={},
    ):
        super(MAB, self).__init__()

        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.ff = LinearNet(
            ff_layers,
            input_size=embed_dim,
            output_size=embed_dim,
            final_linear=final_linear,
            **linear_args,
        )

        self.layer_norm = layer_norm

        if self.layer_norm:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor, y: Tensor, y_mask: Tensor = None):
        if y_mask is not None:
            # torch.nn.MultiheadAttention needs a mask of shape [batch_size * num_heads, N, N]
            y_mask = torch.repeat_interleave(y_mask, self.num_heads, dim=0)

        x = x + self.attention(x, y, y, attn_mask=y_mask, need_weights=False)[0]
        if self.layer_norm:
            x = self.norm1(x)
        x = self.dropout(x)

        x = x + self.ff(x)
        if self.layer_norm:
            x = self.norm2(x)
        x = self.dropout(x)

        return x


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class SAB(nn.Module):
    def __init__(self, **mab_args):
        super(SAB, self).__init__()
        self.mab = MAB(**mab_args)

    def forward(self, x: Tensor, mask: Tensor = None):
        if mask is not None:
            # torch.nn.MultiheadAttention needs a mask vector for each target node
            # i.e. reshaping from [B, N, 1] -> [B, N, N]
            mask = mask.transpose(-2, -1).repeat((1, mask.shape[-2], 1))

        return self.mab(x, x, mask)


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class ISAB(nn.Module):
    def __init__(self, num_inds, embed_dim, **mab_args):
        super(ISAB, self).__init__()
        self.Identity = nn.Parameter(torch.Tensor(1, num_inds, embed_dim))
        self.num_inds = num_inds
        nn.init.xavier_uniform_(self.Identity)
        self.mab0 = MAB(embed_dim=embed_dim, **mab_args)
        self.mab1 = MAB(embed_dim=embed_dim, **mab_args)

    def forward(self, X, mask: Tensor = None):
        if mask is not None:
            mask = mask.transpose(-2, -1).repeat((1, self.num_inds, 1))
        H = self.mab0(self.Identity.repeat(X.size(0), 1, 1), X, mask)
        return self.mab1(X, H)


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class PMA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_seeds: int,
        **mab_args,
    ):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, embed_dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(embed_dim, **mab_args)

    def forward(self, x: Tensor, mask: Tensor = None):
        if mask is not None:
            mask = mask.transpose(-2, -1)

        return self.mab(self.S.repeat(x.size(0), 1, 1), x, mask)


def _attn_mask(mask: Tensor) -> Optional[Tensor]:
    """
    Convert JetNet mask scheme (1 - real, 0 -padded) to nn.MultiHeadAttention mask scheme
    (True - ignore, False - attend)
    """
    if mask is None:
        return None
    else:
        return (1 - mask).bool()


# from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / (sigma.expand_as(w) + 1e-12))

    def _made_params(self):
        try:
            # u = getattr(self.module, self.name + "_u")
            # v = getattr(self.module, self.name + "_v")
            # w = getattr(self.module, self.name + "_bar")
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
