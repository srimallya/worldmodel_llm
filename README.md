# Worldmodel LLM

A small character-level latent world-model language experiment in PyTorch.

The model trains a normal autoregressive language objective alongside a JEPA-style consequence objective:

- `prefix`: observed text state
- `action`: candidate continuation
- `future`: consequence after the action
- teacher: sees `prefix + action + future`
- student: sees `prefix + action` and predicts future latent consequences

The goal is not only to improve next-character loss, but to learn a latent consequence model that can be used during generation.

## What Is In The Script

[worldmodel_llm.py](worldmodel_llm.py) includes:

- character-level causal Transformer LM
- EMA teacher model
- multi-horizon future latent prediction
- JEPA prediction loss
- contrastive future loss
- VICReg-style variance and covariance regularization
- same-prefix counterfactual action diversity
- same-prefix true-action classification against fake actions
- candidate-action planning probes
- planned generation using predicted future latents
- top/bottom candidate action ranking diagnostics
- best-LM and best-world checkpointing
- timestamped Markdown training logs with hyperparameters

## Files

- `worldmodel_llm.py`: training, evaluation, probing, checkpointing, and generation
- `input.txt`: training corpus
- `LICENSE`: MIT license

Generated runtime outputs are ignored by Git:

- `checkpoints/`
- `training_logs/`
- Python cache files

## Run

Install PyTorch for your machine, then run:

```bash
python3 worldmodel_llm.py
```

The script auto-selects CUDA, Apple MPS, or CPU.

## Checkpoints

The script keeps two best checkpoints:

- `checkpoints/best_lm.pt`: lowest validation LM loss
- `checkpoints/best_world.pt`: lowest combined world-model score

The world-model score is:

```python
world_score = (
    val_jepa
    + 0.25 * val_nce
    + 0.10 * val_var
    + 0.01 * val_cov
    + 0.50 * val_cf_div
    + 0.25 * val_action_ce
)
```

Lower is better.

## Training Logs

Each run writes a Markdown log to:

```text
training_logs/training_log_YYYYMMDD_HHMMSS.md
```

The log includes:

- run id
- start timestamp with timezone
- working directory
- Python, platform, and PyTorch versions
- full hyperparameter table
- copied console training output

## Planned Generation

Normal generation samples the next token directly. Planned generation samples several candidate action chunks, predicts the future latent for each candidate, scores candidates by average continuation NLL, moderate latent novelty, and a text degeneracy penalty, then appends the best candidate.

This tests whether the learned consequence model is useful at generation time, not just as an auxiliary training chart.

The script also prints a ranked candidate-action diagnostic: sample many actions from the same prefix, rank them with the planner score, then show the top and bottom candidates. This is a more direct check of whether the planner score is meaningful than one long generated sample.

## Action Contrast

The world objective includes a same-prefix action classifier. Each counterfactual group uses the true corpus action as candidate `0` and random negative actions as candidates `1..K-1`. The model predicts future latents for every candidate action and an `action_scorer` learns to identify the true action with cross-entropy.

This pushes the latent transition model toward action-conditioned branching, not only generic future-manifold matching.

## License

MIT.
