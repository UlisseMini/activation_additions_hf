# %%

from ipywidgets import HTML
from IPython.display import display
from typing import List
from html import escape


tooltip_style = """
<style>
.token > .hover {
    display: none;
    white-space: normal;
    position: absolute;
    line-height: 1.2;
    background-color: #555;
    color: #fff;
    padding: 5px;
    border-radius: 5px;
    z-index: 1;
}
.token:hover > .hover {
    display: inline-block;
}
</style>
"""

# Similar to https://alan-cooney.github.io/CircuitsVis/?path=/docs/tokens-coloredtokens--code-example
# But written from scratch for flexibility, speed, and to avoid dependencies.
def colored_tokens(tokens: List[str], raw_colors: List[float], tooltips: List[str] = None, high=None, low=None, inject_css=True):
    """
    Args:
        tokens: List of tokens to color. Generally obtained from tokenizer.
        raw_colors: List of floats for coloring. Later normalized to [0, 1].
        high: Maximum value for color normalization. Defaults to max(raw_colors).
        low: Minimum value for color normalization. Defaults to min(raw_colors).
    """
    assert len(tokens) == len(raw_colors), f'len(tokens) ({len(tokens)}) != len(raw_colors) ({len(raw_colors)})'
    if tooltips is None:
        tooltips = [f'{c:.2f}' for c in raw_colors]

    # Normalize colors linearly
    high, low = high or max(raw_colors), low or min(raw_colors)
    colors = [(c - low) / (high - low) for c in raw_colors]

    # TODO: More sophisticated color contrasting scheme
    return (tooltip_style if inject_css else '') + ''.join([
        f'''
        <span class="token" style="color: rgb({255*(1-c)}, {255*c}, 0);">
            {escape(s)}
            <span class="hover">{t}</span>
        </span>
        '''.strip()
        for s, c, t in zip(tokens, colors, tooltips)
    ])



if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    if 'model' not in locals():
        tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        model = AutoModelForCausalLM.from_pretrained('gpt2-xl')

    prompt = tokenizer.bos_token + "I like to eat cheese and crackers"
    tokens = tokenizer.batch_decode([[t] for t in tokenizer.encode(prompt)])[1:]
    # tokens = [t.replace(' ', '_') for t in tokens]

    inputs = tokenizer(prompt, return_tensors='pt')
    output = model(**inputs)

    logprobs = torch.log_softmax(output.logits, -1)
    print(logprobs.shape, inputs['input_ids'].shape)

    # Experimental topk view. Probably want nested html instead of content() tooltips.
    k = 5
    topk = torch.topk(logprobs[0], k, dim=-1)
    topk_strs = [
        ' '.join([
            f'{tokenizer.decode(topk.indices.tolist()[t][i])}|{topk.values[t][i].exp():.2f}'
            for i in range(k)
        ])
        for t in range(logprobs.shape[1])
    ]
    topk_tokens = [tokenizer.batch_decode(topk.indices[i]) for i in range(logprobs.shape[1])]
    topk_probs = topk.values.exp()
    topk_htmls = [colored_tokens(topk_tokens[pos], topk_probs[pos], inject_css=False, low=0, high=1) for pos in range(logprobs.shape[1])]

    token_logprobs = logprobs[..., :-1, :].gather(-1, inputs['input_ids'][..., 1:, None])[0, ..., 0]
    token_probs = torch.exp(token_logprobs)

    tokens_html = colored_tokens(
        tokens,
        token_logprobs.tolist(),
        tooltips=[
            f'P: {p:.4f}<br>Topk: {topk_html}' for p, topk_html in zip(token_probs.tolist(), topk_htmls)
        ],
        high=1, low=0,
    )
    display(HTML(f'<p>{tokens_html}</p>'))

# %%
