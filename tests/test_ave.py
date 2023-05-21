# %%
# TODO: Make a separate script to generate constants from ave repo so we don't have to deal with dependency issues.


import ave
import pytest
import torch as t
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from algebraic_value_editing import hook_utils, prompt_utils, lenses
from transformer_lens import HookedTransformer


# %%

# gpt2 fixture
@pytest.fixture(name="models", scope="module")
def fixture_models():
    model_path = 'gpt2-xl'
    device = 'cpu'

    t.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval();

    hooked_model = HookedTransformer.from_pretrained(model_path, hf_model=model)
    hooked_model.to(device)
    hooked_model.eval()

    return (hooked_model, model, tokenizer)


# %%


def test_transformerlens_huggingface_same(models):
    """
    Not really an AVE test, but a sanity check that the hooked model gives the same outputs as the original.
    """
    hooked_model, model, tokenizer = models

    # Test that model logits are the same.
    # TODO: Batch of prompts
    prompt = "I hate you because"
    prompt_tokens = ave.tokenize(tokenizer, [prompt])
    labels = prompt_tokens['input_ids']
    with ave.residual_stream(model) as stream:
        outputs = model(**prompt_tokens, labels=labels)

    # Run HookedTransformer with cached residual stream
    cache, caching_hooks, _ = hooked_model.get_caching_hooks(names_filter=lambda n: "resid_pre" in n)
    logits_hooked, loss_hooked = hooked_model(prompt, return_type='both')
    loss_hooked

    # Compare the modified outputs ensuring they are the same.
    # I think softmax is required because transformerlens & huggingface handle final layernorm differently.
    assert t.allclose(loss_hooked, outputs.loss), f'Losses are not close: {loss_hooked} vs {outputs.loss}'
    assert ((t.log_softmax(logits_hooked, dim=-1) - t.log_softmax(outputs.logits, dim=-1)).abs() < 1e-4).all(), f'Logprobs are not (1e-04)-close'


# %%

def test_ave_same(models):
    # Ave settings
    hooked_model, model, tokenizer = models
    prompt_add, prompt_sub = 'Love ', 'Hate'
    act_name = 6
    prompt = "I hate you because"

    # Tokenize
    prompt_tokens = ave.tokenize(tokenizer, [prompt])
    labels = prompt_tokens['input_ids']

    # Get the difference vector, stream and outputs
    act_diff = ave.get_diff_vector(model, tokenizer, prompt_add, prompt_sub, act_name)
    hook_fn = ave.get_hook_fn(act_diff)
    layer = ave.get_blocks(model)[act_name]
    with ave.pre_hooks([(layer, hook_fn)]):
        outputs = model(**prompt_tokens, labels=labels)

    # Now the same, using the algebraic_value_editing library
    additions = prompt_utils.get_x_vector(prompt_add, prompt_sub, 1., act_name, hooked_model, pad_method=None, custom_pad_id=hooked_model.to_single_token(' '))
    activ_dict = hook_utils.get_activation_dict(hooked_model, additions)
    hook_fns = hook_utils.hook_fns_from_act_dict(activ_dict)
    with hooked_model.hooks(fwd_hooks=lenses.fwd_hooks_from_activ_hooks(hook_fns)):
        hooked_logits, hooked_loss = hooked_model(prompt_tokens['input_ids'], return_type='both')


    # Compare the modified outputs ensuring they are the same
    assert t.allclose(outputs.loss, hooked_loss), f'Losses are not close: {outputs.loss} vs {hooked_loss}'

    # Compare the modified logits ensuring they are the same
    assert ((t.log_softmax(outputs.logits, dim=-1) - t.log_softmax(hooked_logits, dim=-1)).abs() < 1e-4).all(), f'Logprobs are not (1e-04)-close'
