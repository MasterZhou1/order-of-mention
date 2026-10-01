# The Geometry of Contextual Relations: Language Models Address Facts by Order of Mention

**Yufa Zhou**, Duke University

arXiv: (link to be added)

Human reasoning depends on how objects are related within propositions. How do relations organize the language
representations of contextual contents? We give an LLM a list of facts in its context (*Alice eats an apple. Bob eats
a pear.*) and measure how its hidden state changes when the question switches from what Alice eats to what Bob eats.
Averaged over many lists, this change is a steering vector, the **ordinal vector**. It points to a fact by its
**order of mention**, the order in which the facts were stated, not by the name the question contains. The
**ordinal addressing hypothesis** states that each order of mention has a **fact address** in the model's state,
shared by all contexts: a question moves the state to the address of the fact it asks about, while the context
supplies what that fact says, h(C, q) = c_C + a_{p_C(q)}. Across Qwen2.5-7B-Instruct, Gemma-3-4B-IT and
Llama-3.1-8B-Instruct, fact addresses are (1) *ordered by mention*: query states are organized by the order of facts,
not of names, even when one fact has several subjects; (2) *steerable*: added once to a question about the first fact
of a new list, the ordinal vector makes the model answer with that list's second fact; (3) *low-rank*: they span a
low-rank subspace in which the first-mentioned fact is the easiest to reach, surprisingly similar to human recall; and
(4) *emergent*: they are shared in late-middle layers, hold from 1.5B to 32B parameters, and form early in
pretraining (OLMo-2 checkpoints).

This repository contains the code for every experiment, the compact data and scripts that draw every figure and
table of the paper, and the ordinal vectors of the three main models.

## Setup

```bash
pip install -r requirements.txt          # Python >= 3.10
huggingface-cli login                     # Llama and Gemma weights are gated
```

All computation is FP32 without TF32, as in the paper. **Hardware:** every model except Qwen2.5-32B fits on one
80 GB GPU in FP32. Qwen2.5-32B needs about 131 GB; with several GPUs visible, the model is split across them by
whole decoder blocks (`device_map="auto"`). Nothing is trained; the scripts only run forward passes and greedy
decoding.

## Layout

```
relgeom/
  vocab.py       word pools: names, surnames, dishes, animals, cities, jobs, instruments, sports, prose frames
  materials.py   word sets, contexts and questions (lists, two-fact, person x relation, coordinated subjects,
                 two-place relations, prose, lengthened facts, prefixes, inputs without facts)
  models.py      load a HF model, locate decoder blocks (Qwen, Llama, Gemma-3, OLMo-2), chat vs plain prompts,
                 registered main blocks
  capture.py     query states at the last prompt token, any set of blocks
  steering.py    ordinal vectors (Def. 3.2), random directions, once-only add/replace edits during prefill
  scoring.py     complete-answer scoring, stated-content scoring, first-token scoring, word-set bootstrap
  geometry.py    templates and held-out R^2, address subspace, composition, spacing, interaction, alignment
  pipeline.py    shared CLI options and the calibrate -> edit -> score loop
experiments/     one script per result (below); each writes one JSON file
plots/           quick plots of experiment outputs: depth.py (Fig. 5), spacing.py (Fig. 4c-d)
paper/           the paper's figures and tables from compact data (no model needed; see below)
vectors/         the ordinal vectors of Table 1A at the main block, with a loader and an example
```

Every experiment script takes `--model` (a Hugging Face id), `--out`, and optionally `--block` (default: the
registered main block of the model), `--prompt-format {chat,plain}`, `--revision`, `--seed` (random seed for drawing
word sets), `--by-content`, `--batch-size`, and `--save-records` (keep every scored answer). Run them from the
repository root as modules, e.g.

```bash
for m in Qwen/Qwen2.5-7B-Instruct google/gemma-3-4b-it meta-llama/Llama-3.1-8B-Instruct; do
  python -m experiments.steering_transfer --model $m --out results/transfer_${m##*/}.json
done
```

Main blocks (zero-based block outputs): Qwen2.5-7B 20, Gemma-3-4B 22, Llama-3.1-8B 15; the rule that chose them is
`experiments/select_block.py` (App. A).

## Paper ↔ code

| Paper | Claim | Command (`python -m experiments.<script> --model M --out OUT`) |
|---|---|---|
| Sec. 3, Fig. 2c-d, App. B | query change keeps its direction on new names, reverses with the order (Prop. 3.4) | `invariance_reversal` |
| Sec. 4, Fig. 3, App. C.1, Fig. 7, Fig. 13 | query states group by order of mention; within one word set: person → fact address → content; answer alignment | `order_of_mention` |
| Sec. 4-5, App. C.2 | coordinated subjects: the address belongs to the fact, not the name | `fact_vs_mention` |
| Sec. 4, App. C.3, Fig. 8 | two-place facts: fact × argument slot | `two_place_relations` |
| Sec. 5, Table 1A | ordinal vector transfers to new contexts (Prop. 3.6); random and ceiling | `steering_transfer` |
| Sec. 5, Table 1B, App. D.1 | directions without the facts select nothing | `no_facts` |
| Sec. 5, Table 1C, App. D.1 | whole donor states carry which fact to read | `depth_window --blocks main` |
| App. D.2, Table 2 | rearranged facts: the vector follows the new orders of mention | `rearranged_facts` |
| App. D.3, Fig. 9; App. E.1 (relations) | six relations share one ordinal vector and one subspace | `relation_generality` |
| App. D.4, Fig. 10 | order of mention, not token distance | `token_distance` |
| Sec. 6.1, Fig. 4a, App. E.1 | low-rank address subspace; composition along paths (Cor. 3.7) | `address_subspace` |
| Sec. 6.2, Fig. 4b-d, Fig. 11, App. E.2 | longer lists: crowding, serial-position curve, spacing vs transfer (Prop. 3.9) | `longer_lists`, then `python plots/spacing.py results/longer_lists_*.json --out spacing.pdf` |
| Sec. 6.2, App. E.3 | addresses fixed once their fact is read | `prefix_addresses` |
| App. E.4, Table 3, Fig. 12 | person and relation do not add; Lemma E.1 interaction | `person_relation` |
| Sec. 7.1, Fig. 5, App. F.1 | depth window: shared address → whole-state address → content | `depth_window --blocks all`, then `python plots/depth.py results/depth_*.json --out depth.pdf` |
| Sec. 7.2, Fig. 6, Table 4, App. F.2 | OLMo-2-7B checkpoints: when the address forms | `olmo_development --out OUT` (all ten checkpoints; `--checkpoints 5B 42B final` for a subset) |
| App. A | main-block rule | `select_block` |
| App. G.1, Table 5 | scale: Qwen2.5 1.5B-32B, Qwen3-8B, Llama-3.2-3B | `scale_sweep` |
| App. G.2, Table 6 | base vs instruction-tuned weights, plain prompt | `scale_sweep --prompt-format plain --no-depth` (add `--by-content` for `google/gemma-3-4b-pt`) |
| App. G.3, Table 7 | facts stated in prose | `prose` |

Models used in App. G: `Qwen/Qwen2.5-{1.5B,3B,14B,32B}-Instruct`, `Qwen/Qwen3-8B` (chat template with thinking
switched off), `meta-llama/Llama-3.2-3B-Instruct`, and the base weights `Qwen/Qwen2.5-{1.5B,3B}`,
`Qwen/Qwen3-8B-Base`, `google/gemma-3-4b-pt`, `meta-llama/Llama-3.2-3B`. OLMo-2 checkpoints are branches of
`allenai/OLMo-2-1124-7B` (stage-1 steps 1k, 4k, 10k, 25k, 50k, 99k, 200k, 400k, 928,646, and `main`), pinned to
commits in `experiments/olmo_development.py`.

## Protocol in brief

- **Word sets.** Each experiment draws (with `--seed`) its own calibration and test word sets from `relgeom/vocab.py`; they share
  no complete name or content string. Within a word set, the answers have pairwise distinct first tokens under the
  model's tokenizer (colliding contents are replaced from a reserve), so materials can differ slightly by model.
- **Contexts.** Person k holds content (k + assignment) mod n, and order of mention p holds person
  (p + arrangement) mod n; every context is asked every question.
- **Ordinal vector.** v^{p→p'} = mean over calibration word sets of each word set's mean of
  h_ℓ(C, q_C(p')) − h_ℓ(C, q_C(p)), at the last prompt token after block ℓ.
- **Edit.** The receiver's query state is edited once, at the last prompt token while the prompt is processed
  (add v, or replace the whole state for the ceiling and donor conditions); later blocks and positions are
  recomputed and greedy decoding continues for at most 16 tokens.
- **Scoring.** The complete normalized answer must equal the designated content; invalid and truncated answers stay
  in every denominator. Random controls are Gaussian directions with the ordinal vector's length. Intervals are
  95% percentile intervals from 5,000 bootstrap draws over test word sets (donor and receiver word sets resampled
  independently for donor states).


## Figures and tables of the paper

`paper/` regenerates every figure and table of the paper on a CPU in about half a minute, from the compact summaries
in `paper/data/` (JSON, about 1 MB in total); no model is loaded.

```bash
pip install numpy matplotlib
python paper/make_figures.py      # -> paper/figures/*.pdf, *.png  (Figures 1-13)
python paper/make_tables.py       # -> paper/tables/*.tex          (Tables 1-7)
```

The figures use Liberation Sans. Install it (e.g. `apt install fonts-liberation`) or put the `LiberationSans-*.ttf`
files in `paper/fonts/` (or point `FIG_FONT_DIR` at them); otherwise matplotlib falls back to DejaVu Sans and text
widths change slightly.

| Paper | Output | Script (`paper/scripts/`) | Data (`paper/data/`) |
|---|---|---|---|
| Fig. 1a | `order-cartoon` | `fig_order_cartoon.py` | (illustration) |
| Fig. 1b | `access-teaser` | `fig_access_teaser.py` | `teaser_example.json` |
| Fig. 2a | `setting` | `fig_setting.py` | (illustration) |
| Fig. 2b-d | `relational-hypothesis` | `fig_relational_hypothesis.py` | `invariance.json`, `scale_sweep.json` |
| Fig. 3, Fig. 12 | `query-organization`, `query-controls` | `fig_query_organization.py` | `query_geometry.json`, `transfer.json`, `relations.json` |
| Fig. 4 | `address-linear` | `fig_address_linear.py` | `address_geometry.json` |
| Fig. 5, Fig. 13 | `selection-depth`, `depth-geometry` | `fig_selection_depth.py` | `query_geometry.json`, `transfer.json`, `depth_curves.json`, `scale_sweep.json` |
| Fig. 6 | `development` | `fig_development.py` | `olmo2_checkpoints.json` |
| Fig. 7 | `within-world` | `fig_within_word_set.py` | `within_word_set.json`, `depth_curves.json` |
| Fig. 8 | `role-address` | `fig_role_address.py` | `two_place_relations.json` |
| Fig. 9 | `relation-generality` | `fig_relation_generality.py` | `relations.json` |
| Fig. 10 | `address-controls` | `fig_address_controls.py` | `token_distance.json` |
| Fig. 11 | `longer-lists` | `fig_longer_lists.py` | `longer_lists.json` |
| Table 1 | `selection-evidence-table.tex` | `tables.py` | `transfer.json` |
| Table 2 | `rearranged-table.tex` | `tables.py` | `transfer.json` |
| Table 3 | `composition-table.tex` | `tables.py` | `composition.json` |
| Table 4 | `development-table.tex` | `tables.py` | `olmo2_checkpoints.json` |
| Table 5 | `generality-scale-table.tex` | `tables.py` | `scale_sweep.json` |
| Table 6 | `generality-base-table.tex` | `tables.py` | `scale_sweep.json`, `invariance.json`, `query_geometry.json` |
| Table 7 | `generality-prose-table.tex` | `tables.py` | `prose.json` |

Each data file starts with a `description` of its fields. Rates are fractions of cases with the designated answer,
with 95% word-set bootstrap intervals (`ci`); models are keyed `qwen7b`, `gemma4b`, `llama8b` (and, in
`scale_sweep.json`, the further models of App. G). Orders of mention are zero-based in the data files and one-based
in the paper.

## Ordinal vectors

`vectors/` holds the ordinal vectors v^{p→q} of Table 1A for the three main models at their main blocks, as float16
(`ordinal_vectors[p-1, q-1]` = v^{p→q} for four-fact lists; estimated from 8 calibration word sets × 4 arrangements ×
4 content assignments of *eats* lists, 128 query changes per pair).

| File | Model | Main block | Hidden size |
|---|---|---|---|
| `qwen2.5-7b-instruct.npz` | `Qwen/Qwen2.5-7B-Instruct` | 20 | 3584 |
| `gemma-3-4b-it.npz` | `google/gemma-3-4b-it` | 22 | 2560 |
| `llama-3.1-8b-instruct.npz` | `meta-llama/Llama-3.1-8B-Instruct` | 15 | 4096 |

```python
import sys; sys.path.insert(0, "vectors")
from load_vectors import ordinal_vector
v, block, model_id = ordinal_vector("qwen", 1, 2)       # v^{1->2}, block 20, "Qwen/Qwen2.5-7B-Instruct"
```

To steer, add v to the output of decoder block `block` (zero-based) at the last prompt token, once, while the prompt
is processed, and decode greedily. `vectors/example.py` does this with the code of `relgeom/` on a context whose
relation, contents and names were never used for calibration:

```bash
python vectors/example.py --model qwen          # prints the answer without and with v^{1->2}
```

The vectors are model- and block-specific: they live in the residual stream of the instruction-tuned model, read
with its native chat template in FP32.

## Citation

```bibtex
@article{zhou2026geometry,
  title={The Geometry of Contextual Relations: Language Models Address Facts by Order of Mention},
  author={Zhou, Yufa},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT (see `LICENSE`).
