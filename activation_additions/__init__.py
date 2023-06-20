# %%

from contextlib import contextmanager
from typing import Tuple, List, Callable, Optional
import torch as t
import torch.nn as nn


# types
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = List[Hook]


@contextmanager
def pre_hooks(hooks: Hooks):
    """
    Context manager to register pre-forward hooks on a list of modules. The hooks are removed when the context exits.
    """
    try:
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def get_blocks(model: nn.Module) -> nn.ModuleList:
    """
    Get the ModuleList containing the transformer blocks from a model.
    """
    def numel_(mod):
        return sum(p.numel() for p in mod.parameters())
    model_numel = numel_(model)
    canidates = [
        mod
        for mod in model.modules()
        if isinstance(mod, nn.ModuleList) and numel_(mod) > .5*model_numel
    ]
    assert len(canidates) == 1, f'Found {len(canidates)} ModuleLists with >50% of model params.'
    return canidates[0]


@contextmanager
def residual_stream(model: nn.Module, layers: Optional[List[int]] = None) -> List[t.Tensor]:
    """
    Context manager to store residual stream activations in the model at the specified layers.
    Alternatively "model(..., output_hidden_states=True)" can be used, this is more flexible though and works with model.generate().
    """

    stream = [None] * len(get_blocks(model))
    layers = layers or range(len(stream))
    def _make_hook(i):
        def _hook(_, inputs):
            # concat along the sequence dimension
            stream[i] = inputs[0] if stream[i] is None else t.cat([stream[i], inputs[0]], dim=1)
        return _hook

    hooks = [(layer, _make_hook(i)) for i, layer in enumerate(get_blocks(model)) if i in layers]
    with pre_hooks(hooks):
        yield stream


def _device(model):
    "Get the device of the first parameter of the model. Assumes all parameters are on the same device."
    return next(model.parameters()).device


def get_vectors(model: nn.Module, tokenizer, prompts: List[str], layer: int):
    """
    Get the activations of the prompts at the specified layer. Used later for activation additions.
    """
    with residual_stream(model, layers=[layer]) as stream:
        inputs = tokenizer(prompts, return_tensors='pt', padding=True)
        device = _device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

    return stream[layer]


def get_diff_vector(model: nn.Module, tokenizer, prompt_add: str, prompt_sub: str, layer: int):
    """
    Get the difference vector between the activations of prompt_add and prompt_sub at the specified layer. 
    Convinience function for
        s = get_vectors(..., prompts)
        return s[0] - s[1]
    """
    stream = get_vectors(model, tokenizer, [prompt_add, prompt_sub], layer)
    return (stream[0] - stream[1]).unsqueeze(0)


def get_hook_fn(act_add: t.Tensor) -> PreHookFn:
    """
    Get a hook function that adds an activation addition vector.
    """

    def _hook(_: nn.Module, inputs: Tuple[t.Tensor]):
        resid_pre, = inputs
        if resid_pre.shape[1] == 1:
            return None # caching for new tokens in generate()

        # We only add to the prompt (first call), not the generated tokens.
        ppos, apos = resid_pre.shape[1], act_add.shape[1]
        assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

        # TODO: Make this a function-wrapper for flexibility.
        resid_pre[:, :apos, :] += act_add
        return resid_pre

    return _hook

