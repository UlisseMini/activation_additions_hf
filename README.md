# Activation Additions

Reproducing the core of [Steering GPT-2-XL by adding an activation vector](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector). The [existing codebase](https://github.com/montemac/algebraic_value_editing) is tied to TransformerLens which has [memory issues](https://github.com/neelnanda-io/TransformerLens/issues/252) when loading large models. Since the core of AVE is simple, reimplementing it from scratch was easier than trying to fix TransformerLens.

Notebooks:
- `qualitative.ipynb` - A port of the qualitative notebook from the original codebase to LLAMA.
- `playground.ipynb` - The core research playground. Includes all the code for playing with x-vectors.
- `results.ipynb` - The notebook with all the results for the bigmodels post (currently WIP).