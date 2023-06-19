# %%

from ipywidgets import HTML
from IPython.display import display
from typing import List

# Similar to https://alan-cooney.github.io/CircuitsVis/?path=/docs/tokens-coloredtokens--code-example
# But written from scratch for flexibility, speed, and to avoid dependencies.
def colored_tokens(tokens: List[str], colors: List[float]):
    """
    Args:
        tokens: List of tokens to color. Generally obtained from tokenizer.
        colors: List of floats between 0 and 1, one for each token. Typically probabilities.
    """
    assert len(tokens) == len(colors)
    for token, color in zip(tokens, colors):
        assert 0 <= color <= 1, f'color {color} for {token} is not between 0 and 1'

    # TODO: More sophisticated color contrasting scheme
    # TODO: Javascript hovering
    return ''.join([
        f'<span style="color: rgb({255*(1-c)}, {255*c}, 0);">{s}</span>'
        for s, c in zip(tokens, colors)
    ])


# %%
# Test it

if __name__ == '__main__':
    from transformers import GPT2Tokenizer
    import torch
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    prompt = "I like to eat cheese and crackers"
    tokens = tokenizer.batch_decode([[t] for t in tokenizer.encode(prompt)])[1:]
    losses = torch.softmax(torch.tensor([-22.31, -20., -5., -2., -10., -2., -4.]), -1)

    display(HTML(f'<p>{colored_tokens(tokens, losses)}</p>'))
