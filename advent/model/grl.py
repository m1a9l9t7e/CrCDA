# Implementation of Gradient Reversal Layer
# usage: x = GradientReversalFunction.apply(input, grl_lambda)

import torch


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        return dx, None
