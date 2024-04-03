import torch
from scipy import integrate



def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.minimum(
        sigma_to,
        eta
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat


def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )
    return betas.numpy()


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def mixed_checkpoint(func, inputs: dict, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass. This differs from the original checkpoint function
    borrowed from https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py in that
    it also works with non-tensor inputs
    :param func: the function to evaluate.
    :param inputs: the argument dictionary to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        tensor_keys = [key for key in inputs if isinstance(inputs[key], torch.Tensor)]
        tensor_inputs = [
            inputs[key] for key in inputs if isinstance(inputs[key], torch.Tensor)
        ]
        non_tensor_keys = [
            key for key in inputs if not isinstance(inputs[key], torch.Tensor)
        ]
        non_tensor_inputs = [
            inputs[key] for key in inputs if not isinstance(inputs[key], torch.Tensor)
        ]
        args = tuple(tensor_inputs) + tuple(non_tensor_inputs) + tuple(params)
        return MixedCheckpointFunction.apply(
            func,
            len(tensor_inputs),
            len(non_tensor_inputs),
            tensor_keys,
            non_tensor_keys,
            *args,
        )
    else:
        return func(**inputs)


class MixedCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        run_function,
        length_tensors,
        length_non_tensors,
        tensor_keys,
        non_tensor_keys,
        *args,
    ):
        ctx.end_tensors = length_tensors
        ctx.end_non_tensors = length_tensors + length_non_tensors
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        assert (
            len(tensor_keys) == length_tensors
            and len(non_tensor_keys) == length_non_tensors
        )

        ctx.input_tensors = {
            key: val for (key, val) in zip(tensor_keys, list(args[: ctx.end_tensors]))
        }
        ctx.input_non_tensors = {
            key: val
            for (key, val) in zip(
                non_tensor_keys, list(args[ctx.end_tensors : ctx.end_non_tensors])
            )
        }
        ctx.run_function = run_function
        ctx.input_params = list(args[ctx.end_non_tensors :])

        with torch.no_grad():
            output_tensors = ctx.run_function(
                **ctx.input_tensors, **ctx.input_non_tensors
            )
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # additional_args = {key: ctx.input_tensors[key] for key in ctx.input_tensors if not isinstance(ctx.input_tensors[key],torch.Tensor)}
        ctx.input_tensors = {
            key: ctx.input_tensors[key].detach().requires_grad_(True)
            for key in ctx.input_tensors
        }

        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = {
                key: ctx.input_tensors[key].view_as(ctx.input_tensors[key])
                for key in ctx.input_tensors
            }
            # shallow_copies.update(additional_args)
            output_tensors = ctx.run_function(**shallow_copies, **ctx.input_non_tensors)
        input_grads = torch.autograd.grad(
            output_tensors,
            list(ctx.input_tensors.values()) + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (
            (None, None, None, None, None)
            + input_grads[: ctx.end_tensors]
            + (None,) * (ctx.end_non_tensors - ctx.end_tensors)
            + input_grads[ctx.end_tensors :]
        )


ckpt = torch.utils.checkpoint.checkpoint
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        #args = tuple(inputs) + tuple(params)
        #return CheckpointFunction.apply(func, len(inputs), *args)
        return ckpt(func, *inputs)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            assert image_only_indicator is not None, "need image_only_indicator ..."
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator)
        x = (
            alpha.to(x_spatial.dtype) * x_spatial
            + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        return x

    import math
    from typing import Optional

    import torch
    import torch.nn as nn
    from einops import rearrange, repeat

    def make_beta_schedule(
            schedule,
            n_timestep,
            linear_start=1e-4,
            linear_end=2e-2,
    ):
        if schedule == "linear":
            betas = (
                    torch.linspace(
                        linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
                    )
                    ** 2
            )
        return betas.numpy()

    def extract_into_tensor(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def mixed_checkpoint(func, inputs: dict, params, flag):
        """
        Evaluate a function without caching intermediate activations, allowing for
        reduced memory at the expense of extra compute in the backward pass. This differs from the original checkpoint function
        borrowed from https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py in that
        it also works with non-tensor inputs
        :param func: the function to evaluate.
        :param inputs: the argument dictionary to pass to `func`.
        :param params: a sequence of parameters `func` depends on but does not
                       explicitly take as arguments.
        :param flag: if False, disable gradient checkpointing.
        """
        if flag:
            tensor_keys = [key for key in inputs if isinstance(inputs[key], torch.Tensor)]
            tensor_inputs = [
                inputs[key] for key in inputs if isinstance(inputs[key], torch.Tensor)
            ]
            non_tensor_keys = [
                key for key in inputs if not isinstance(inputs[key], torch.Tensor)
            ]
            non_tensor_inputs = [
                inputs[key] for key in inputs if not isinstance(inputs[key], torch.Tensor)
            ]
            args = tuple(tensor_inputs) + tuple(non_tensor_inputs) + tuple(params)
            return MixedCheckpointFunction.apply(
                func,
                len(tensor_inputs),
                len(non_tensor_inputs),
                tensor_keys,
                non_tensor_keys,
                *args,
            )
        else:
            return func(**inputs)

    class MixedCheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(
                ctx,
                run_function,
                length_tensors,
                length_non_tensors,
                tensor_keys,
                non_tensor_keys,
                *args,
        ):
            ctx.end_tensors = length_tensors
            ctx.end_non_tensors = length_tensors + length_non_tensors
            ctx.gpu_autocast_kwargs = {
                "enabled": torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            assert (
                    len(tensor_keys) == length_tensors
                    and len(non_tensor_keys) == length_non_tensors
            )

            ctx.input_tensors = {
                key: val for (key, val) in zip(tensor_keys, list(args[: ctx.end_tensors]))
            }
            ctx.input_non_tensors = {
                key: val
                for (key, val) in zip(
                    non_tensor_keys, list(args[ctx.end_tensors: ctx.end_non_tensors])
                )
            }
            ctx.run_function = run_function
            ctx.input_params = list(args[ctx.end_non_tensors:])

            with torch.no_grad():
                output_tensors = ctx.run_function(
                    **ctx.input_tensors, **ctx.input_non_tensors
                )
            return output_tensors

        @staticmethod
        def backward(ctx, *output_grads):
            # additional_args = {key: ctx.input_tensors[key] for key in ctx.input_tensors if not isinstance(ctx.input_tensors[key],torch.Tensor)}
            ctx.input_tensors = {
                key: ctx.input_tensors[key].detach().requires_grad_(True)
                for key in ctx.input_tensors
            }

            with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
                # Fixes a bug where the first op in run_function modifies the
                # Tensor storage in place, which is not allowed for detach()'d
                # Tensors.
                shallow_copies = {
                    key: ctx.input_tensors[key].view_as(ctx.input_tensors[key])
                    for key in ctx.input_tensors
                }
                # shallow_copies.update(additional_args)
                output_tensors = ctx.run_function(**shallow_copies, **ctx.input_non_tensors)
            input_grads = torch.autograd.grad(
                output_tensors,
                list(ctx.input_tensors.values()) + ctx.input_params,
                output_grads,
                allow_unused=True,
            )
            del ctx.input_tensors
            del ctx.input_params
            del output_tensors
            return (
                    (None, None, None, None, None)
                    + input_grads[: ctx.end_tensors]
                    + (None,) * (ctx.end_non_tensors - ctx.end_tensors)
                    + input_grads[ctx.end_tensors:]
            )

    ckpt = torch.utils.checkpoint.checkpoint

    def checkpoint(func, inputs, params, flag):
        """
        Evaluate a function without caching intermediate activations, allowing for
        reduced memory at the expense of extra compute in the backward pass.
        :param func: the function to evaluate.
        :param inputs: the argument sequence to pass to `func`.
        :param params: a sequence of parameters `func` depends on but does not
                       explicitly take as arguments.
        :param flag: if False, disable gradient checkpointing.
        """
        if flag:
            # args = tuple(inputs) + tuple(params)
            # return CheckpointFunction.apply(func, len(inputs), *args)
            return ckpt(func, *inputs)
        else:
            return func(*inputs)

    class CheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, run_function, length, *args):
            ctx.run_function = run_function
            ctx.input_tensors = list(args[:length])
            ctx.input_params = list(args[length:])
            ctx.gpu_autocast_kwargs = {
                "enabled": torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad():
                output_tensors = ctx.run_function(*ctx.input_tensors)
            return output_tensors

        @staticmethod
        def backward(ctx, *output_grads):
            ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
            with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
                # Fixes a bug where the first op in run_function modifies the
                # Tensor storage in place, which is not allowed for detach()'d
                # Tensors.
                shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
                output_tensors = ctx.run_function(*shallow_copies)
            input_grads = torch.autograd.grad(
                output_tensors,
                ctx.input_tensors + ctx.input_params,
                output_grads,
                allow_unused=True,
            )
            del ctx.input_tensors
            del ctx.input_params
            del output_tensors
            return (None, None) + input_grads

    def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        if not repeat_only:
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half
            ).to(device=timesteps.device)
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat(
                    [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
                )
        else:
            embedding = repeat(timesteps, "b -> b d", d=dim)
        return embedding

    def zero_module(module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()
        return module

    def scale_module(module, scale):
        """
        Scale the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().mul_(scale)
        return module

    def mean_flat(tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def normalization(channels):
        """
        Make a standard normalization layer.
        :param channels: number of input channels.
        :return: an nn.Module for normalization.
        """
        return GroupNorm32(32, channels)

    # PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    class GroupNorm32(nn.GroupNorm):
        def forward(self, x):
            return super().forward(x.float()).type(x.dtype)

    def conv_nd(dims, *args, **kwargs):
        """
        Create a 1D, 2D, or 3D convolution module.
        """
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    def linear(*args, **kwargs):
        """
        Create a linear module.
        """
        return nn.Linear(*args, **kwargs)

    def avg_pool_nd(dims, *args, **kwargs):
        """
        Create a 1D, 2D, or 3D average pooling module.
        """
        if dims == 1:
            return nn.AvgPool1d(*args, **kwargs)
        elif dims == 2:
            return nn.AvgPool2d(*args, **kwargs)
        elif dims == 3:
            return nn.AvgPool3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    class AlphaBlender(nn.Module):
        strategies = ["learned", "fixed", "learned_with_images"]

        def __init__(
                self,
                alpha: float,
                merge_strategy: str = "learned_with_images",
                rearrange_pattern: str = "b t -> (b t) 1 1",
        ):
            super().__init__()
            self.merge_strategy = merge_strategy
            self.rearrange_pattern = rearrange_pattern

            assert (
                    merge_strategy in self.strategies
            ), f"merge_strategy needs to be in {self.strategies}"

            if self.merge_strategy == "fixed":
                self.register_buffer("mix_factor", torch.Tensor([alpha]))
            elif (
                    self.merge_strategy == "learned"
                    or self.merge_strategy == "learned_with_images"
            ):
                self.register_parameter(
                    "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
                )
            else:
                raise ValueError(f"unknown merge strategy {self.merge_strategy}")

        def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
            if self.merge_strategy == "fixed":
                alpha = self.mix_factor
            elif self.merge_strategy == "learned":
                alpha = torch.sigmoid(self.mix_factor)
            elif self.merge_strategy == "learned_with_images":
                assert image_only_indicator is not None, "need image_only_indicator ..."
                alpha = torch.where(
                    image_only_indicator.bool(),
                    torch.ones(1, 1, device=image_only_indicator.device),
                    rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
                )
                alpha = rearrange(alpha, self.rearrange_pattern)
            else:
                raise NotImplementedError
            return alpha

        def forward(
                self,
                x_spatial: torch.Tensor,
                x_temporal: torch.Tensor,
                image_only_indicator: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            alpha = self.get_alpha(image_only_indicator)
            x = (
                    alpha.to(x_spatial.dtype) * x_spatial
                    + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
            )
            return x

import functools
import importlib
import os
from functools import partial
from inspect import isfunction

import fsspec
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file as load_safetensors


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def get_string_from_tuple(s):
    try:
        # Check if the string starts and ends with parentheses
        if s[0] == "(" and s[-1] == ")":
            # Convert the string to a tuple
            t = eval(s)
            # Check if the type of t is tuple
            if type(t) == tuple:
                return t[0]
            else:
                pass
    except:
        pass
    return s


def is_power_of_two(n):
    """
    chat.openai.com/chat
    Return True if n is a power of 2, otherwise return False.

    The function is_power_of_two takes an integer n as input and returns True if n is a power of 2, otherwise it returns False.
    The function works by first checking if n is less than or equal to 0. If n is less than or equal to 0, it can't be a power of 2, so the function returns False.
    If n is greater than 0, the function checks whether n is a power of 2 by using a bitwise AND operation between n and n-1. If n is a power of 2, then it will have only one bit set to 1 in its binary representation. When we subtract 1 from a power of 2, all the bits to the right of that bit become 1, and the bit itself becomes 0. So, when we perform a bitwise AND between n and n-1, we get 0 if n is a power of 2, and a non-zero value otherwise.
    Thus, if the result of the bitwise AND operation is 0, then n is a power of 2 and the function returns True. Otherwise, the function returns False.

    """
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        if isinstance(xc[bi], list):
            text_seq = xc[bi][0]
        else:
            text_seq = xc[bi]
        lines = "\n".join(
            text_seq[start : start + nc] for start in range(0, len(text_seq), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    return path


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def isheatmap(x):
    if not isinstance(x, torch.Tensor):
        return False

    return x.ndim == 2


def isneighbors(x):
    if not isinstance(x, torch.Tensor):
        return False
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


def exists(x):
    return x is not None


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def load_model_from_config(config, ckpt, verbose=True, freeze=True):
    print(f"Loading model from {ckpt}")
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError

    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.eval()
    return model


def get_configs_path() -> str:

    this_dir = os.path.dirname(__file__)
    candidates = (
        os.path.join(this_dir, "configs"),
        os.path.join(this_dir, "..", "configs"),
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find SGM configs in {candidates}")


def get_nested_attribute(obj, attribute_path, depth=None, return_key=False):
    """
    Will return the result of a recursive get attribute call.
    E.g.:
        a.b.c
        = getattr(getattr(a, "b"), "c")
        = get_nested_attribute(a, "b.c")
    If any part of the attribute call is an integer x with current obj a, will
    try to call a[x] instead of a.x first.
    """
    attributes = attribute_path.split(".")
    if depth is not None and depth > 0:
        attributes = attributes[:depth]
    assert len(attributes) > 0, "At least one attribute should be selected"
    current_attribute = obj
    current_key = None
    for level, attribute in enumerate(attributes):
        current_key = ".".join(attributes[: level + 1])
        try:
            id_ = int(attribute)
            current_attribute = current_attribute[id_]
        except ValueError:
            current_attribute = getattr(current_attribute, attribute)

    return (current_attribute, current_key) if return_key else current_attribute