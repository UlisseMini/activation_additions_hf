# %%

import ave
import pytest
import torch as t; t.set_grad_enabled(False)
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# Ave settings

prompt = "I hate you because"
prompt_add, prompt_sub = 'Love ', 'Hate'
act_name = 6
coeff = 2.

# %%
# Run this cell to recompute loss constants we compare against

"""
from transformer_lens import HookedTransformer
from algebraic_value_editing import hook_utils, prompt_utils, lenses

hooked_model = HookedTransformer.from_pretrained('gpt2-xl')
hooked_model.eval()

# uses default TransformerLens tokenization, which matches `ave.tokenize`
vanilla_loss = hooked_model(prompt, return_type='loss')
print(f'vanilla loss {vanilla_loss.item()}')

# compute ave loss
additions = prompt_utils.get_x_vector(prompt_add, prompt_sub, coeff, act_name, hooked_model, pad_method=None, custom_pad_id=hooked_model.to_single_token(' '))
activ_dict = hook_utils.get_activation_dict(hooked_model, additions)
hook_fns = hook_utils.hook_fns_from_act_dict(activ_dict)
with hooked_model.hooks(fwd_hooks=lenses.fwd_hooks_from_activ_hooks(hook_fns)):
    ave_loss = hooked_model(prompt, return_type='loss')

print(f'ave loss {ave_loss.item()}')
"""

# %%

@pytest.fixture(name="models", scope="module")
def fixture_models():
    model_path, device = 'gpt2-xl', 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device); model.eval()

    return (model, tokenizer)


# %%


def test_transformerlens_huggingface_same(models):
    """
    Not really an AVE test, but a sanity check that the hooked model gives the same outputs as the original.
    """
    model, tokenizer = models

    # Test that model logits are the same.
    # TODO: Batch of prompts
    prompt = "I hate you because"
    prompt_tokens = ave.tokenize(tokenizer, [prompt])
    labels = prompt_tokens['input_ids']
    with ave.residual_stream(model) as stream:
        outputs = model(**prompt_tokens, labels=labels)

    # See cell above to compute
    loss_target = t.tensor(4.819790363311768)

    # Compare the modified outputs ensuring they are the same.
    assert t.allclose(loss_target, outputs.loss), f'Losses are not close: {loss_target} vs {outputs.loss}'


# %%

def test_ave_same(models):
        # Ave settings
    model, tokenizer = models
    prompt_add, prompt_sub = 'Love ', 'Hate'
    act_name = 6
    prompt = "I hate you because"

    # Tokenize
    prompt_tokens = ave.tokenize(tokenizer, [prompt])
    labels = prompt_tokens['input_ids']

    # Get the difference vector, stream and outputs
    act_diff = coeff * ave.get_diff_vector(model, tokenizer, prompt_add, prompt_sub, act_name)
    hook_fn = ave.get_hook_fn(act_diff)
    layer = ave.get_blocks(model)[act_name]
    with ave.pre_hooks([(layer, hook_fn)]):
        outputs = model(**prompt_tokens, labels=labels)

    # See cell above to compute
    loss_target = t.tensor(7.975721836090088)

    # Compare the modified outputs ensuring they are the same
    assert t.allclose(outputs.loss, loss_target), f'Losses are not close: {outputs.loss} vs target {loss_target}'