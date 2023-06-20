# %%
# Compatibility layer for algebraic_value_editing, such that notebooks can be ran without changes.

from typing import Callable, Dict, Optional, Tuple, Union, Any, List
from functools import partial
import torch as t
import activation_additions
import pandas as pd
from dataclasses import dataclass
import prettytable
import numpy as np
from transformers import LogitsProcessor


# %%

Model = Any # TODO: better types


@dataclass
class ActivationAddition:
    """
    Add coeff*vec to `layer` in the hooked transformer residual stream. prompt is only for logging.
    NOTE: This is suboptimal, we aren't storing the tokenized prompt. But this is fine for now.
    """
    prompt: str
    coeff: float
    layer: int
    act: t.Tensor

    # check act shape is right
    def __post_init__(self):
        assert len(self.act.shape) == 3, f"act must be (batch, seq_len, dim) but got shape {self.act.shape}"

    def __repr__(self):
        return f"ActivationAddition(coeff={self.coeff}, layer={self.layer}, prompt='{self.prompt}', act.shape={self.act.shape})"


# Unlike in the other repo, we compute vectors in the get_x_vector function.
def get_x_vector(
    prompt1: str,
    prompt2: str,
    coeff: float,
    act_name: Union[int, str],
    model: Optional[Model] = None,
    pad_method: Optional[str] = None,
    custom_pad_id: Optional[int] = None,
) -> List[ActivationAddition]:
    assert hasattr(model, 'tokenizer'), "Model must have a model.tokenizer"
    if pad_method is not None and pad_method != "tokens_right":
        raise NotImplementedError('pad_method != "tokens_right" is not implemented')
    
    if custom_pad_id is not None:
        model.tokenizer.pad_token_id = custom_pad_id

    act = activation_additions.get_vectors(model, model.tokenizer, [prompt1, prompt2], act_name)

    return [
        ActivationAddition(prompt=prompt1, coeff=coeff, layer=act_name, act=act[0].unsqueeze(0)),
        ActivationAddition(prompt=prompt2, coeff=-coeff, layer=act_name, act=act[1].unsqueeze(0)),
    ]


# %%


class FrequencyPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, frequency_penalty: float):
        self.frequency_penalty = frequency_penalty

    def __call__(self, current_tokens: t.LongTensor, scores: t.FloatTensor) -> t.FloatTensor:
        for batch_index in range(scores.shape[0]):
            scores[batch_index] = scores[batch_index] - self.frequency_penalty * t.bincount(
                current_tokens[batch_index], minlength=scores.shape[-1]
            )
        return scores


def port_sampling_kwargs(sampling_kwargs: Dict[str, float]) -> Dict[str, float]:
    sampling_kwargs = sampling_kwargs.copy()
    logit_processors = []

    if 'freq_penalty' in sampling_kwargs:
        logit_processors.append(FrequencyPenaltyLogitsProcessor(sampling_kwargs['freq_penalty']))

        del sampling_kwargs['freq_penalty']

    if 'seed' in sampling_kwargs:
        t.manual_seed(sampling_kwargs['seed'])
        del sampling_kwargs['seed']

    if 'tokens_to_generate' in sampling_kwargs:
        sampling_kwargs['max_new_tokens'] = sampling_kwargs['tokens_to_generate']
        del sampling_kwargs['tokens_to_generate']

    # argmax is default, need to switch to sampling
    sampling_kwargs['do_sample'] = True
    sampling_kwargs['logits_processor'] = logit_processors
    
    return sampling_kwargs



def get_n_comparisons(prompts: List[str], model: Model, additions: List[ActivationAddition], **sampling_kwargs) -> pd.DataFrame:
    """
    Print n completions from the modified and unmodified model. Format results in a table. **kwargs are passed to model.generate().
    """
    assert hasattr(model, 'tokenizer'), "Model must have a model.tokenizer"
    def _to_df(tokens: t.Tensor, modified: bool):
        completions = [model.tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in tokens]
        trimmed = [c[len(p):] for p, c in zip(prompts, completions)]

        return pd.DataFrame({
            'prompts': prompts,
            'completions': trimmed,
            'is_modified': modified,
        })

    inputs, device = model.tokenizer(prompts, return_tensors='pt', padding=True), activation_additions._device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate unmodified completions
    # FIXME: "Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation." should not happen. tokenizer has a set token?
    nom_tokens = model.generate(**inputs, **port_sampling_kwargs(sampling_kwargs))

    # Generate modified completions
    blocks = activation_additions.get_blocks(model)
    hooks = [(blocks[a.layer], activation_additions.get_hook_fn(a.coeff*a.act)) for a in additions]
    with activation_additions.pre_hooks(hooks):
        mod_tokens = model.generate(**inputs, **port_sampling_kwargs(sampling_kwargs))
    
    nom_df, mod_df = _to_df(nom_tokens, modified=False), _to_df(mod_tokens, modified=True)
    return pd.concat([nom_df, mod_df], ignore_index=True)


def print_n_comparisons(
    prompt: str,
    model: Model,
    num_comparisons: int = 5,
    activation_additions: Optional[List[ActivationAddition]] = None,
    addition_location: str = "front",
    res_stream_slice: slice = slice(None),
    **kwargs,
) -> None:
    assert num_comparisons > 0, "num_comparisons must be positive"
    assert hasattr(model, 'tokenizer'), "Model must have a model.tokenizer"
    if addition_location != "front":
        raise NotImplementedError("addition_location != front not implemented yet") # trivial to add, but not needed yet
    if res_stream_slice != slice(None):
        raise NotImplementedError("res_stream_slice != slice(None) not implemented yet")


    prompt_batch: List[str] = [prompt] * num_comparisons
    
    results = get_n_comparisons(prompts=prompt_batch, model=model, additions=activation_additions, **kwargs)
    pretty_print_completions(results=results)


# Display utils #
def bold_text(text: str) -> str:
    """Returns a string with ANSI bold formatting."""
    return f"\033[1m{text}\033[0m"

def _remove_eos(completion: str) -> str:
    """If completion ends with multiple <|endoftext|> strings, return a
    new string in which all but one are removed."""
    has_eos: bool = completion.endswith("<|endoftext|>")
    new_completion: str = completion.rstrip("<|endoftext|>")
    if has_eos:
        new_completion += "<|endoftext|>"
    return new_completion


def pretty_print_completions(
    results: pd.DataFrame,
    normal_title: str = "Unsteered completions",
    mod_title: str = "Steered completions",
    normal_prompt_override: Optional[str] = None,
    mod_prompt_override: Optional[str] = None,
) -> None:
    """Pretty-print the given completions.

    args:
        `results`: A `DataFrame` with the completions.

        `normal_title`: The title to use for the normal completions.

        `mod_title`: The title to use for the modified completions.

        `normal_prompt_override`: If not `None`, use this prompt for the
            normal completions.

        `mod_prompt_override`: If not `None`, use this prompt for the
            modified completions.
    """
    assert all(
        col in results.columns
        for col in ("prompts", "completions", "is_modified")
    )

    # Assert that an equal number of rows have `is_modified` True and
    # False
    n_rows_mod, n_rows_unmod = [
        len(results[results["is_modified"] == cond]) for cond in [True, False]
    ]
    all_modified: bool = n_rows_unmod == 0
    all_normal: bool = n_rows_mod == 0
    assert all_normal or all_modified or (n_rows_mod == n_rows_unmod), (
        "The number of modified and normal completions must be the same, or we"
        " must be printing all (un)modified completions."
    )

    # Figure out which columns to add
    completion_cols: List[str] = []
    completion_cols += [normal_title] if n_rows_unmod > 0 else []
    completion_cols += [mod_title] if n_rows_mod > 0 else []
    completion_dict: dict = {}
    for col in completion_cols:
        is_mod = col == mod_title
        completion_dict[col] = results[results["is_modified"] == is_mod][
            "completions"
        ]

    # Format the DataFrame for printing
    prompt: str = results["prompts"].tolist()[0]

    # Generate the table
    table = prettytable.PrettyTable()
    table.align = "c"
    table.field_names = map(bold_text, completion_cols)
    table.min_width = table.max_width = 60

    # Separate completions
    table.hrules = prettytable.ALL

    # Put into table
    for row in zip(*completion_dict.values()):
        # Bold the appropriate prompt
        normal_str = bold_text(
            prompt
            if normal_prompt_override is None
            else normal_prompt_override
        )
        mod_str = bold_text(
            prompt if mod_prompt_override is None else mod_prompt_override
        )
        if all_modified:
            new_row = [mod_str + _remove_eos(row[0])]
        elif all_normal:
            new_row = [normal_str + _remove_eos(row[0])]
        else:
            normal_str += _remove_eos(row[0])
            mod_str += _remove_eos(row[1])
            new_row = [normal_str, mod_str]

        table.add_row(new_row)
    print(table)


# %%
# Import model and tokenizer (dev)

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained('gpt2-xl')
    tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
    setattr(model, 'tokenizer', tokenizer) # bit cursed but w/e

    from transformer_lens import HookedTransformer
    hooked_model = HookedTransformer.from_pretrained('gpt2-xl')



# %%
# Example usage (dev)

if __name__ == '__main__':
    sampling_kwargs: Dict[str, Union[float, int]] = {
        "temperature": 1.0,
        "top_p": 0.3,
        "freq_penalty": 1.0,
        "num_comparisons": 3,
        "tokens_to_generate": 50,
        "seed": 0,  # For reproducibility
    }
    get_x_vector_preset: Callable = partial(
        get_x_vector,
        pad_method="tokens_right",
        model=model,
        custom_pad_id=int(tokenizer.encode(" ")[0]),
    )

    summand: List[ActivationAddition] = [
        *get_x_vector_preset(
            prompt1="Love",
            prompt2="Hate",
            coeff=5,
            act_name=6,
        )
    ]
    HATE_PROMPT = "I hate you because"
    print_n_comparisons(
        model=model,
        prompt=HATE_PROMPT,
        activation_additions=summand,
        **sampling_kwargs,
    )
