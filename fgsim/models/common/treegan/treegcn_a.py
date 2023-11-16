import math

import torch
import torch.nn as nn

from fgsim.models.common import FFN


class TreeGCN(nn.Module):
    def __init__(
        self,
        batch: int,
        depth: int,
        features,
        degrees,
        support=10,
        n_parents=1,  # number of points in the next layer
        activation=True,
    ):
        self.batch = batch
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth + 1]
        self.n_parents = n_parents  # The number of the node in the tree
        self.degree = degrees[depth]  # The Branching ratio
        self.activation = activation
        super(TreeGCN, self).__init__()

        # Is the transformation matrix, that brings the root node
        # to the correct feature size
        self.W_root = nn.ModuleList(
            [
                nn.Linear(features[inx], self.out_feature, bias=False)
                for inx in range(self.depth + 1)
            ]
        )

        # This is the matrix for the branching
        # U^l_j
        # self.W_branch = nn.Parameter(
        #     torch.FloatTensor(
        #         self.n_parents, self.in_feature, self.degree * self.in_feature
        #     )
        # )
        self.branch_nn = FFN(self.in_feature, self.degree * self.in_feature)

        # Loop term, F^l_K in the paper
        self.W_loop = nn.Sequential(
            nn.Linear(self.in_feature, self.in_feature * support, bias=False),
            nn.Linear(self.in_feature * support, self.out_feature, bias=False),
        )

        # Bias term for the convolution
        self.bias = nn.Parameter(
            torch.FloatTensor(1, self.degree, self.out_feature)
        )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        stdv = 1.0 / math.sqrt(self.out_feature)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, tree):
        # the tree is a the feature matrix
        # this tree passed forward between the the TreeGCN layers
        # Make sure that the layer fits
        assert tree[-1].size(1) == self.n_parents

        # ## First we generate the new neighbors
        # tree[-1] :  batch_size x parents x features
        # unsqueze  -> batch_size x parents x 1 x in_feature
        # self.W_branch: parents x in_feature x self.in_feature*branches
        # @ multiplies the last two dimension and broadcasts the rest
        # branch = tree[-1].unsqueeze(2) @ self.W_branch
        branch = self.branch_nn(tree[-1])
        # => branch shape: batch_size x parents x 1 x in_feature*branches
        branch = self.leaky_relu(branch)
        branch = branch.view(
            self.batch, self.n_parents * self.degree, self.in_feature
        )

        # ## Now the Convolution ###
        branch = self.W_loop(branch)

        # here the ancestors are aggregated
        root = 0
        for inx in range(self.depth + 1):
            # Project the parents to the correct features
            root_node = self.W_root[inx](tree[inx])
            # shape: batch_size x points x features

            # this is the number of points in the last layer /
            # number of parents
            i_num_parents = tree[inx].size(1)
            # number of points that need to be generated / number of parents
            repeat_num = int(self.n_parents / i_num_parents)
            # repeat the transformed parents, to that we get one parent per point
            # that we have one root per points that we want
            # We repeat in the last dimenstion and then reshape to the proper from,
            # to have the same root nodes next to each other
            # Add the node to the previously collected ancestors
            root = root + root_node.repeat(1, 1, repeat_num).view(
                self.batch, self.n_parents, self.out_feature
            )
            pass

        # Add branch and root term
        branch = (
            root.repeat(1, 1, self.degree).view(self.batch, -1, self.out_feature)
            + branch
        )

        if self.activation:
            branch = self.leaky_relu(
                branch + self.bias.repeat(1, self.n_parents, 1)
            )

        tree.append(branch)

        return tree
