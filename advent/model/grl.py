# Implementation of Gradient Reversal Layer
# usage: x = GradientReversalFunction.apply(input, grl_lambda)

import torch


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_wrapper):
        ctx.lambda_wrapper = lambda_wrapper
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_wrapper.get_lambda()
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        return dx, None


class LambdaWrapper:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_

    def set_lambda(self, value):
        self.lambda_ = value

    def get_lambda(self):
        return self.lambda_
