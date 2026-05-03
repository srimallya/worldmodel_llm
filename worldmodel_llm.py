# ============================
# Latent Text World Model JEPA Character LM
# Corpus: input.txt
#
# Core idea:
#   prefix = observed state
#   action = candidate continuation
#   future = consequence after action
#
#   Teacher sees:
#       prefix + action + future
#
#   Student sees:
#       prefix + action
#
#   Student predicts teacher's latent future consequences.
#
# World-model interpretation:
#       W(s_t, a_t) -> z_future
#
# Improvements over simple version:
#   - multi-horizon future latents
#   - JEPA warmup and ramp
#   - VICReg-style anti-collapse regularization
#   - candidate-action consequence probing
#   - cleaner logging
#
# No RL. No reward circus. Just latent consequence prediction.
# ============================

import os
import math
import copy
import sys
import platform
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# Hyperparameters
# ============================

config = dict(
    input_path="/Users/srimallyamaitra/codes/worldmodel/input.txt",

    val_frac=0.10,

    # sequence layout
    prefix_len=192,
    action_len=32,
    future_len=64,

    # must cover prefix + action + future
    block_size=288,

    batch_size=8,

    # model
    n_layer=4,
    n_head=8,
    n_embd=384,
    dropout=0.15,

    # latent world model
    latent_dim=256,
    projector_hidden=512,
    predictor_hidden=512,

    # multi-horizon future prediction
    # horizons are measured inside future span
    # e.g. horizon 8 means first 8 chars of future
    future_horizons=(8, 16, 32, 64),

    # training
    max_steps=50000,
    eval_interval=200,
    learning_rate=3e-4,
    weight_decay=0.01,
    grad_clip=1.0,

    # loss weights
    token_weight=1.0,

    # JEPA ramps from 0 to this value
    jepa_weight=0.30,

    # VICReg-style anti-collapse terms
    variance_weight=0.08,
    covariance_weight=0.01,

    # Future discrimination terms
    contrastive_weight=0.08,
    contrastive_temp=0.15,
    diversity_weight=0.01,
    counterfactual_diversity_weight=0.02,
    counterfactual_group_size=4,
    counterfactual_margin=0.80,
    action_ce_weight=0.05,
    counterfactual_negative_mode="random",
    model_negative_temperature=0.9,
    model_negative_top_k=80,

    # warmup/ramp
    jepa_warmup_steps=1500,
    jepa_ramp_steps=8000,

    # latent norm control to prevent explosion after JEPA ramp
    max_latent_norm=10.0,
    latent_norm_weight=0.005,

    # action_ce delay: only start after match_gap > threshold or step > threshold
    action_ce_match_gap_threshold=0.05,
    action_ce_step_threshold=3000,
    ema_decay=0.995,

    # sampling
    sample_tokens=400,
    top_k=90,
    temperature=0.8,

    # checkpointing
    checkpoint_dir="checkpoints",
    best_lm_checkpoint_name="best_lm.pt",
    best_world_checkpoint_name="best_world.pt",

    # run documentation
    training_log_dir="training_logs",

    # planning probe
    planning_prefix_chars=160,
    planning_candidates=6,
    planning_action_tokens=64,

    # planned generation
    planned_generation_prefix="Harry looked at Hermione and",
    planned_sample_tokens=400,
    planned_action_tokens=32,
    planned_candidates=8,
    planned_temperature=0.9,
    planned_top_k=80,
    planned_diversity_bonus=0.10,
    planned_degeneracy_penalty_weight=0.75,
    planned_horizon_instability_weight=0.25,

    # candidate ranking diagnostic
    rank_action_tokens=64,
    rank_candidates=16,

    seed=1337,
)


torch.manual_seed(config["seed"])


# ============================
# Run documentation
# ============================

run_started_at = datetime.now().astimezone()
run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
training_log_path = os.path.join(
    config["training_log_dir"],
    f"training_log_{run_id}.md",
)

action_ce_enabled = False


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_training_log():
    os.makedirs(config["training_log_dir"], exist_ok=True)

    log_file = open(training_log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    print("# Latent Text World Model Training Log")
    print("")
    print("## Run")
    print("")
    print(f"- run_id: `{run_id}`")
    print(f"- started_at: `{run_started_at.isoformat()}`")
    print(f"- working_directory: `{os.getcwd()}`")
    print(f"- python: `{platform.python_version()}`")
    print(f"- platform: `{platform.platform()}`")
    print(f"- torch: `{torch.__version__}`")
    print("")
    print("## Hyperparameters")
    print("")
    print("| key | value |")
    print("|-----|-------|")

    for key in sorted(config):
        print(f"| `{key}` | `{repr(config[key])}` |")

    print("")
    print("## Console Log")
    print("")
    print("```text")

    return log_file, original_stdout, original_stderr


training_log_file, original_stdout, original_stderr = setup_training_log()


# ============================
# Device
# ============================

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()
print(f"using device: {device}")


# ============================
# Data
# ============================

path = config["input_path"]

if not os.path.exists(path):
    raise FileNotFoundError(
        f"input.txt not found at {path}. Set config['input_path'] correctly. "
        "The model cannot train on vibes. Tragic, but reproducible."
    )

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    ids = []
    for c in s:
        if c in stoi:
            ids.append(stoi[c])
    return torch.tensor(ids, dtype=torch.long)


def decode(t):
    return "".join([itos[int(i)] for i in t])


data = encode(text)

n = int((1.0 - config["val_frac"]) * len(data))
train_data = data[:n]
val_data = data[n:]


def get_lm_batch(split):
    src = train_data if split == "train" else val_data
    T = config["block_size"]

    if len(src) <= T + 1:
        raise ValueError("Corpus too small for block_size. Feed the tiny beast more text.")

    ix = torch.randint(0, len(src) - T - 1, (config["batch_size"],))

    x = torch.stack([src[i:i + T] for i in ix]).to(device)
    y = torch.stack([src[i + 1:i + T + 1] for i in ix]).to(device)

    return x, y


def get_world_batch(split):
    src = train_data if split == "train" else val_data

    prefix_len = config["prefix_len"]
    action_len = config["action_len"]
    future_len = config["future_len"]

    total = prefix_len + action_len + future_len + 1

    if len(src) <= total:
        raise ValueError(
            "Corpus too small for prefix + action + future. "
            "The toy universe lacks enough universe."
        )

    ix = torch.randint(0, len(src) - total, (config["batch_size"],))

    seq = torch.stack([src[i:i + total] for i in ix]).to(device)

    prefix = seq[:, :prefix_len]
    action = seq[:, prefix_len:prefix_len + action_len]
    future = seq[:, prefix_len + action_len:prefix_len + action_len + future_len]

    student_visible = torch.cat([prefix, action], dim=1)
    teacher_visible = torch.cat([prefix, action, future], dim=1)

    lm_x = seq[:, :prefix_len + action_len + future_len]
    lm_y = seq[:, 1:prefix_len + action_len + future_len + 1]

    return {
        "prefix": prefix,
        "action": action,
        "future": future,
        "student_visible": student_visible,
        "teacher_visible": teacher_visible,
        "lm_x": lm_x,
        "lm_y": lm_y,
    }


def get_counterfactual_action_batch(split, model=None):
    src = train_data if split == "train" else val_data

    prefix_len = config["prefix_len"]
    action_len = config["action_len"]
    group_size = config["counterfactual_group_size"]
    total = prefix_len + action_len + 1
    negative_mode = config["counterfactual_negative_mode"]

    if group_size < 2:
        raise ValueError("counterfactual_group_size must be at least 2.")

    if negative_mode not in ("random", "model"):
        raise ValueError("counterfactual_negative_mode must be 'random' or 'model'.")

    if len(src) <= total:
        raise ValueError(
            "Corpus too small for counterfactual prefix/action batches."
        )

    ix = torch.randint(0, len(src) - total, (config["batch_size"],))

    prefixes = []
    actions = []

    for i in ix:
        prefix = src[i:i + prefix_len]
        true_action = src[i + prefix_len:i + prefix_len + action_len]

        if negative_mode == "model" and model is not None:
            was_training = model.training
            model.eval()
            neg_actions = []

            with torch.no_grad():
                prefix_device = prefix.unsqueeze(0).to(device)

                for _ in range(group_size - 1):
                    generated = model.generate(
                        prefix_device.clone(),
                        max_new_tokens=action_len,
                        temperature=config["model_negative_temperature"],
                        top_k_val=config["model_negative_top_k"],
                    )
                    neg_actions.append(generated[0, prefix_len:].detach().cpu())

            if was_training:
                model.train()
        else:
            neg_ix = torch.randint(0, len(src) - action_len - 1, (group_size - 1,))
            neg_actions = [src[j:j + action_len] for j in neg_ix]

        all_actions = [true_action] + neg_actions

        prefixes.append(prefix.unsqueeze(0).repeat(group_size, 1))
        actions.append(torch.stack(all_actions))

    prefixes = torch.cat(prefixes, dim=0).to(device)
    actions = torch.cat(actions, dim=0).to(device)

    return torch.cat([prefixes, actions], dim=1)


# ============================
# Utilities
# ============================

def top_k_filter(logits, k):
    if k is None or k <= 0:
        return logits

    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    cutoff = v[..., -1, None]

    return torch.where(logits < cutoff, torch.full_like(logits, -1e10), logits)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def schedule_value(step, warmup_steps, ramp_steps, max_value):
    """
    Returns a smooth ramp:
      0 before warmup
      linearly ramps after warmup
      max_value after warmup + ramp
    """

    if step < warmup_steps:
        return 0.0

    t = (step - warmup_steps) / max(1, ramp_steps)
    t = max(0.0, min(1.0, t))

    return max_value * t


def variance_loss(z, eps=1e-4):
    """
    VICReg-style variance loss.
    Prevents each latent dimension from collapsing.

    z: [B, D]
    """

    std = torch.sqrt(z.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))

    return loss


def covariance_loss(z):
    """
    VICReg-style covariance loss.
    Discourages redundant latent dimensions.

    z: [B, D]
    """

    B, D = z.shape

    if B <= 1:
        return torch.tensor(0.0, device=z.device)

    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)

    loss = off_diagonal(cov).pow(2).sum() / D

    return loss


def latent_norm_loss(z_pred, max_norm=10.0):
    """
    Prevents latent norm explosion after JEPA ramp.
    Penalizes latent norms that exceed max_norm.

    z_pred: [B, H, D]
    """
    norms = z_pred.norm(dim=-1)
    loss = F.relu(norms - max_norm).pow(2).mean()
    return loss


def jepa_prediction_loss(z_pred, z_teacher):
    """
    Directional match plus bounded raw-latent match.

    The cosine term keeps consequence direction aligned while the Smooth L1
    term prevents latent norm drift from being invisible to the objective.
    """

    z_pred_n = F.normalize(z_pred, dim=-1)
    z_teacher_n = F.normalize(z_teacher, dim=-1)

    cos_loss = 1.0 - (z_pred_n * z_teacher_n).sum(dim=-1).mean()
    raw_loss = F.smooth_l1_loss(
        torch.tanh(z_pred / 5.0),
        torch.tanh(z_teacher / 5.0),
    )

    return cos_loss + 0.25 * raw_loss


def jepa_prediction_loss_per_horizon(z_pred, z_teacher):
    """
    Compute JEPA loss per horizon.

    z_pred: [B, H, D]
    z_teacher: [B, H, D]

    Returns: list of losses, one per horizon
    """
    horizons = config["future_horizons"]
    losses = {}
    for h_idx, h in enumerate(horizons):
        z_p = z_pred[:, h_idx, :]
        z_t = z_teacher[:, h_idx, :]
        z_p_n = F.normalize(z_p, dim=-1)
        z_t_n = F.normalize(z_t, dim=-1)
        cos_loss = 1.0 - (z_p_n * z_t_n).sum(dim=-1).mean()
        raw_loss = F.smooth_l1_loss(
            torch.tanh(z_p / 5.0),
            torch.tanh(z_t / 5.0),
        )
        losses[h] = (cos_loss + 0.25 * raw_loss).item()
    return losses


def horizon_cosine_distances(z_pred):
    """
    Compute mean cosine distance between adjacent predicted horizons.

    z_pred: [B, H, D]

    Returns: float mean adjacent cosine distance
    """
    z_normed = F.normalize(z_pred, dim=-1)
    adjacent_cos = (z_normed[:, :-1, :] * z_normed[:, 1:, :]).sum(dim=-1)
    return float(adjacent_cos.mean().detach().cpu())


def diversity_loss(z, margin=0.15):
    """
    z: [B, H, D]
    Encourages different samples in the batch to have separated final-horizon
    consequence latents.
    """

    zf = F.normalize(z[:, -1, :], dim=-1)
    sim = zf @ zf.T
    B = zf.size(0)

    if B <= 1:
        return torch.tensor(0.0, device=z.device)

    mask = ~torch.eye(B, dtype=torch.bool, device=zf.device)
    off = sim[mask]

    return F.relu(off - margin).mean()


def grouped_action_diversity_loss(z, group_size, margin=0.80):
    """
    z: [B*K, H, D]
    Each group has the same prefix with K different actions.
    Encourages different actions under the same prefix to have different
    final-horizon consequence latents.
    """

    zf = F.normalize(z[:, -1, :], dim=-1)
    num_groups = zf.size(0) // group_size

    if num_groups == 0 or group_size <= 1:
        return torch.tensor(0.0, device=z.device)

    zf = zf[:num_groups * group_size]
    losses = []

    for g in range(num_groups):
        group = zf[g * group_size:(g + 1) * group_size]
        sim = group @ group.T
        mask = ~torch.eye(group_size, dtype=torch.bool, device=z.device)
        off = sim[mask]
        losses.append(F.relu(off - margin).mean())

    return torch.stack(losses).mean()


def contrastive_future_loss(z_pred, z_teacher, temperature=0.1):
    """
    z_pred: [B, H, D]
    z_teacher: [B, H, D]
    For each horizon, predicted future should match its own teacher future,
    not other batch futures.
    """

    B, H, D = z_pred.shape
    labels = torch.arange(B, device=z_pred.device)
    total = 0.0

    for h in range(H):
        p = F.normalize(z_pred[:, h, :], dim=-1)
        t = F.normalize(z_teacher[:, h, :], dim=-1)
        logits = (p @ t.T) / temperature
        total = total + F.cross_entropy(logits, labels)

    return total / H


def action_classification_loss(model, z_pred, group_size):
    """
    Same-prefix counterfactual action classification.

    Candidate 0 in each group is the true corpus action. Candidates 1..K-1
    are random negative actions sampled for the same prefix.
    """

    zf = z_pred[:, -1, :]
    num_groups = zf.size(0) // group_size

    if num_groups == 0 or group_size <= 1:
        return torch.tensor(0.0, device=zf.device)

    zf = zf[:num_groups * group_size]
    scores = model.action_scorer(zf).view(num_groups, group_size)
    labels = torch.zeros(num_groups, dtype=torch.long, device=zf.device)

    return F.cross_entropy(scores, labels)


def teacher_delta_future_latents(teacher, teacher_visible):
    teacher_hidden = teacher.encode_hidden(teacher_visible)

    target_latents = []
    context_end = config["prefix_len"] + config["action_len"]

    z_context_t = teacher.latent_from_hidden_span(
        teacher_hidden,
        start=0,
        end=context_end,
    )

    for horizon in config["future_horizons"]:
        future_end = context_end + horizon

        z_future_t = teacher.latent_from_hidden_span(
            teacher_hidden,
            start=context_end,
            end=future_end,
        )

        target_latents.append(z_future_t - z_context_t)

    return torch.stack(target_latents, dim=1)


def latent_stats(z):
    with torch.no_grad():
        return {
            "z_norm": float(z.norm(dim=-1).mean().detach().cpu()),
            "z_std": float(z.std(dim=0).mean().detach().cpu()),
        }


# ============================
# Model
# ============================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(scores, dim=-1)
        att = self.attn_drop(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)

        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LatentTextWorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        n_embd = config["n_embd"]
        block_size = config["block_size"]

        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(
                n_embd=n_embd,
                n_head=config["n_head"],
                dropout=config["dropout"],
                block_size=block_size,
            )
            for _ in range(config["n_layer"])
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.projector = nn.Sequential(
            nn.Linear(n_embd, config["projector_hidden"]),
            nn.GELU(),
            nn.Linear(config["projector_hidden"], config["latent_dim"]),
        )

        self.state_action_projector = nn.Sequential(
            nn.Linear(3 * n_embd, config["projector_hidden"]),
            nn.GELU(),
            nn.Linear(config["projector_hidden"], config["latent_dim"]),
        )

        # One predictor shared across horizons.
        # Horizon identity is injected with a learned embedding.
        self.horizon_embed = nn.Embedding(
            len(config["future_horizons"]),
            config["latent_dim"],
        )

        self.predictor = nn.Sequential(
            nn.Linear(config["latent_dim"], config["predictor_hidden"]),
            nn.GELU(),
            nn.Linear(config["predictor_hidden"], config["latent_dim"]),
        )

        self.action_scorer = nn.Sequential(
            nn.Linear(config["latent_dim"], config["projector_hidden"]),
            nn.GELU(),
            nn.Linear(config["projector_hidden"], 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def encode_hidden(self, idx):
        B, T = idx.shape

        assert T <= self.block_size, f"T={T} exceeds block_size={self.block_size}"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.wte(idx) + self.wpe(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        return x

    def forward(self, idx, targets=None):
        hidden = self.encode_hidden(idx)
        logits = self.lm_head(hidden)

        loss = None

        if targets is not None:
            B, T = targets.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, -1),
                targets.reshape(B * T),
            )

        return logits, loss, hidden

    def latent_from_hidden_span(self, hidden, start=None, end=None):
        if start is None:
            start = 0

        if end is None:
            end = hidden.size(1)

        pooled = hidden[:, start:end, :].mean(dim=1)
        z = self.projector(pooled)

        return z

    def latent_from_span(self, idx, start=None, end=None):
        hidden = self.encode_hidden(idx)
        return self.latent_from_hidden_span(hidden, start=start, end=end)

    def predict_future_latents(self, student_visible):
        """
        Student sees prefix + action.

        Returns:
            preds: [B, H, D]
        """

        hidden = self.encode_hidden(student_visible)

        prefix_end = min(config["prefix_len"], hidden.size(1))

        prefix_pool = hidden[:, :prefix_end, :].mean(dim=1)

        if prefix_end < hidden.size(1):
            action_pool = hidden[:, prefix_end:, :].mean(dim=1)
        else:
            action_pool = hidden[:, -1:, :].mean(dim=1)

        boundary_state = hidden[:, -1, :]

        pooled = torch.cat([prefix_pool, action_pool, boundary_state], dim=-1)
        z_context = self.state_action_projector(pooled)

        preds = []

        for h_idx in range(len(config["future_horizons"])):
            h_id = torch.full(
                (z_context.size(0),),
                h_idx,
                dtype=torch.long,
                device=z_context.device,
            )

            z_h = z_context + self.horizon_embed(h_id)
            z_pred = self.predictor(z_h)

            preds.append(z_pred)

        preds = torch.stack(preds, dim=1)

        return preds

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k_val=0):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _, _ = self(idx_cond, targets=None)

            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = top_k_filter(logits, top_k_val)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

        return idx


# ============================
# Teacher update
# ============================

@torch.no_grad()
def update_ema_teacher(teacher, student, decay):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


# ============================
# Training / Evaluation
# ============================

def make_models():
    student = LatentTextWorldModel().to(device)
    teacher = copy.deepcopy(student).to(device)

    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    return student, teacher


@torch.no_grad()
def estimate_lm_loss(model):
    model.eval()

    out = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(10):
            xb, yb = get_lm_batch(split)
            _, loss, _ = model(xb, yb)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()

    return out


@torch.no_grad()
def estimate_world_loss(student, teacher):
    student.eval()
    teacher.eval()

    out = {}

    for split in ["train", "val"]:
        jepa_losses = []
        var_losses = []
        cov_losses = []
        nce_losses = []
        div_losses = []
        cf_div_losses = []
        action_ce_losses = []

        for _ in range(10):
            batch = get_world_batch(split)

            student_visible = batch["student_visible"]
            teacher_visible = batch["teacher_visible"]

            z_pred = student.predict_future_latents(student_visible)

            z_teacher = teacher_delta_future_latents(teacher, teacher_visible)

            jepa = jepa_prediction_loss(z_pred, z_teacher)
            nce = contrastive_future_loss(
                z_pred,
                z_teacher,
                temperature=config["contrastive_temp"],
            )
            div = diversity_loss(z_pred)

            cf_visible = get_counterfactual_action_batch(split, model=student)
            cf_pred = student.predict_future_latents(cf_visible)
            cf_div = grouped_action_diversity_loss(
                cf_pred,
                group_size=config["counterfactual_group_size"],
                margin=config["counterfactual_margin"],
            )
            action_ce = action_classification_loss(
                student,
                cf_pred,
                group_size=config["counterfactual_group_size"],
            )

            B, H, D = z_pred.shape
            z_flat = z_pred.reshape(B * H, D)

            v_loss = variance_loss(z_flat)
            c_loss = covariance_loss(z_flat)

            jepa_losses.append(jepa.item())
            var_losses.append(v_loss.item())
            cov_losses.append(c_loss.item())
            nce_losses.append(nce.item())
            div_losses.append(div.item())
            cf_div_losses.append(cf_div.item())
            action_ce_losses.append(action_ce.item())

        out[split] = {
            "jepa": sum(jepa_losses) / len(jepa_losses),
            "var": sum(var_losses) / len(var_losses),
            "cov": sum(cov_losses) / len(cov_losses),
            "nce": sum(nce_losses) / len(nce_losses),
            "div": sum(div_losses) / len(div_losses),
            "cf_div": sum(cf_div_losses) / len(cf_div_losses),
            "action_ce": sum(action_ce_losses) / len(action_ce_losses),
        }

    student.train()

    return out


@torch.no_grad()
def sample_text(model, prefix="", steps=None):
    if steps is None:
        steps = config["sample_tokens"]

    if len(prefix) == 0:
        start_id = torch.randint(0, vocab_size, (1, 1), device=device)
    else:
        start_id = encode(prefix).unsqueeze(0).to(device)

    out = model.generate(
        start_id,
        max_new_tokens=steps,
        temperature=config["temperature"],
        top_k_val=config["top_k"],
    )[0].tolist()

    print(decode(out))


def visible_prefix_action(tokens):
    visible_len = config["prefix_len"] + config["action_len"]
    visible = tokens[:, -min(tokens.size(1), visible_len):]

    if visible.size(1) < visible_len:
        pad_len = visible_len - visible.size(1)
        pad = visible[:, :1].repeat(1, pad_len)
        visible = torch.cat([pad, visible], dim=1)

    return visible


def text_degeneracy_penalty(token_ids):
    """
    Penalizes visibly ugly action chunks before the planner commits to them.
    """

    s = decode(token_ids.tolist())

    if len(s) == 0:
        return 1.0

    penalty = 0.0

    repeats = 0
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            repeats += 1
    penalty += repeats / max(1, len(s))

    space_ratio = s.count(" ") / max(1, len(s))
    if space_ratio > 0.25:
        penalty += (space_ratio - 0.25) * 2.0

    upper_ratio = sum(c.isupper() for c in s) / max(1, len(s))
    if upper_ratio > 0.20:
        penalty += (upper_ratio - 0.20) * 2.0

    alpha_ratio = sum(c.isalpha() for c in s) / max(1, len(s))
    if alpha_ratio < 0.55:
        penalty += (0.55 - alpha_ratio) * 2.0

    return penalty


def horizon_shift_distance(z_pred):
    if z_pred.size(1) <= 1:
        return 0.0

    z_normed = F.normalize(z_pred, dim=-1)
    adjacent_cos = (z_normed[:, :-1, :] * z_normed[:, 1:, :]).sum(dim=-1)
    shift = 1.0 - adjacent_cos

    return float(shift.mean().detach().cpu())


@torch.no_grad()
def continuation_nll(model, full_ids, prefix_len):
    """
    Average NLL of generated continuation tokens that are visible in the
    current context window.

    full_ids: [1, T]
    prefix_len: token length before this candidate action was sampled
    """

    T = full_ids.size(1)

    if T <= prefix_len:
        return 999.0

    start = max(0, T - 1 - model.block_size)
    x = full_ids[:, start:T - 1]
    y = full_ids[:, start + 1:T]

    if x.numel() == 0 or y.numel() == 0:
        return 999.0

    logits, _, _ = model(x, targets=None)
    logp = F.log_softmax(logits, dim=-1)

    first_y_global_pos = start + 1
    cont_start = max(0, prefix_len - first_y_global_pos)

    cont_logp = logp[:, cont_start:, :].gather(
        -1,
        y[:, cont_start:].unsqueeze(-1),
    ).squeeze(-1)

    if cont_logp.numel() == 0:
        return 999.0

    return float((-cont_logp.mean()).detach().cpu())


@torch.no_grad()
def planned_generate(
    model,
    prefix,
    total_new_tokens=400,
    action_tokens=32,
    candidates=8,
    temperature=0.9,
    top_k_val=80,
    diversity_bonus=0.05,
    degeneracy_penalty_weight=0.75,
    horizon_instability_weight=0.25,
):
    model.eval()

    idx = encode(prefix).unsqueeze(0).to(device)

    if idx.numel() == 0:
        idx = torch.randint(0, vocab_size, (1, 1), device=device)

    target_len = idx.size(1) + total_new_tokens

    while idx.size(1) < target_len:
        step_tokens = min(action_tokens, target_len - idx.size(1))
        candidate_infos = []

        for _ in range(candidates):
            prefix_len = idx.size(1)
            cand = model.generate(
                idx.clone(),
                max_new_tokens=step_tokens,
                temperature=temperature,
                top_k_val=top_k_val,
            )

            visible = visible_prefix_action(cand)
            z_pred = model.predict_future_latents(visible)
            z_final = z_pred[:, -1, :]
            horizon_shift = horizon_shift_distance(z_pred)

            action_segment = cand[0, prefix_len:]
            deg_penalty = text_degeneracy_penalty(action_segment)
            nll = continuation_nll(model, cand, prefix_len)

            candidate_infos.append({
                "cand": cand,
                "z": z_final.squeeze(0),
                "nll": nll,
                "deg_penalty": deg_penalty,
                "horizon_shift": horizon_shift,
            })

        Z = torch.stack([x["z"] for x in candidate_infos], dim=0)
        Z_n = F.normalize(Z, dim=-1)
        sim = Z_n @ Z_n.T

        scores = []

        for i, info in enumerate(candidate_infos):
            if len(candidate_infos) > 1:
                others = torch.cat([sim[i, :i], sim[i, i + 1:]], dim=0)
                novelty = 1.0 - others.mean()
                novelty = float(novelty.detach().cpu())
            else:
                novelty = 0.0

            score = (
                -info["nll"]
                + diversity_bonus * novelty
                - degeneracy_penalty_weight * info["deg_penalty"]
                - horizon_instability_weight * info["horizon_shift"]
            )
            scores.append(score)

        best_i = max(range(len(scores)), key=lambda i: scores[i])
        idx = candidate_infos[best_i]["cand"]

    return decode(idx[0].tolist())


def train_step(student, teacher, optimizer, step):
    global action_ce_enabled
    batch = get_world_batch("train")

    student_visible = batch["student_visible"]
    teacher_visible = batch["teacher_visible"]
    lm_x = batch["lm_x"]
    lm_y = batch["lm_y"]

    # ----------------------------
    # 1. Token competence
    # ----------------------------

    _, token_loss, _ = student(lm_x, lm_y)

    # ----------------------------
    # 2. Teacher sees future
    # ----------------------------

    with torch.no_grad():
        teacher.eval()

        z_teacher = teacher_delta_future_latents(teacher, teacher_visible).detach()

    # ----------------------------
    # 3. Student predicts future consequences
    # ----------------------------

    z_pred = student.predict_future_latents(student_visible)

    # ----------------------------
    # 4. JEPA latent prediction loss
    # ----------------------------

    jepa_loss = jepa_prediction_loss(z_pred, z_teacher)
    nce_loss = contrastive_future_loss(
        z_pred,
        z_teacher,
        temperature=config["contrastive_temp"],
    )
    d_loss = diversity_loss(z_pred)

    cf_visible = get_counterfactual_action_batch("train", model=student)
    student.train()
    cf_pred = student.predict_future_latents(cf_visible)
    cf_d_loss = grouped_action_diversity_loss(
        cf_pred,
        group_size=config["counterfactual_group_size"],
        margin=config["counterfactual_margin"],
    )
    action_ce_loss = action_classification_loss(
        student,
        cf_pred,
        group_size=config["counterfactual_group_size"],
    )

    # ----------------------------
    # 5. Anti-collapse latent regularization
    # ----------------------------

    B, H, D = z_pred.shape
    z_flat = z_pred.reshape(B * H, D)

    v_loss = variance_loss(z_flat)
    c_loss = covariance_loss(z_flat)
    ln_loss = latent_norm_loss(z_pred, max_norm=config["max_latent_norm"])

    # ----------------------------
    # 6. Loss schedule
    # ----------------------------

    jepa_w = schedule_value(
        step=step,
        warmup_steps=config["jepa_warmup_steps"],
        ramp_steps=config["jepa_ramp_steps"],
        max_value=config["jepa_weight"],
    )

    var_w = schedule_value(
        step=step,
        warmup_steps=config["jepa_warmup_steps"],
        ramp_steps=config["jepa_ramp_steps"],
        max_value=config["variance_weight"],
    )

    cov_w = schedule_value(
        step=step,
        warmup_steps=config["jepa_warmup_steps"],
        ramp_steps=config["jepa_ramp_steps"],
        max_value=config["covariance_weight"],
    )

    ln_w = config["latent_norm_weight"]

    nce_w = schedule_value(
        step=step,
        warmup_steps=config["jepa_warmup_steps"],
        ramp_steps=config["jepa_ramp_steps"],
        max_value=config["contrastive_weight"],
    )

    div_w = schedule_value(
        step=step,
        warmup_steps=config["jepa_warmup_steps"],
        ramp_steps=config["jepa_ramp_steps"],
        max_value=config["diversity_weight"],
    )

    cf_div_w = schedule_value(
        step=step,
        warmup_steps=config["jepa_warmup_steps"],
        ramp_steps=config["jepa_ramp_steps"],
        max_value=config["counterfactual_diversity_weight"],
    )

    action_ce_w = schedule_value(
        step=step,
        warmup_steps=config["jepa_warmup_steps"],
        ramp_steps=config["jepa_ramp_steps"],
        max_value=config["action_ce_weight"],
    )

    if not action_ce_enabled:
        action_ce_w = 0.0

    # ----------------------------
    # 7. Total loss
    # ----------------------------

    loss = (
        config["token_weight"] * token_loss
        + jepa_w * jepa_loss
        + var_w * v_loss
        + cov_w * c_loss
        + ln_w * ln_loss
        + nce_w * nce_loss
        + div_w * d_loss
        + cf_div_w * cf_d_loss
        + action_ce_w * action_ce_loss
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = nn.utils.clip_grad_norm_(
        student.parameters(),
        config["grad_clip"],
    )

    optimizer.step()

    update_ema_teacher(
        teacher=teacher,
        student=student,
        decay=config["ema_decay"],
    )

    stats = latent_stats(z_flat)

    return {
        "loss": float(loss.detach().cpu()),
        "token": float(token_loss.detach().cpu()),
        "jepa": float(jepa_loss.detach().cpu()),
        "var": float(v_loss.detach().cpu()),
        "cov": float(c_loss.detach().cpu()),
        "nce": float(nce_loss.detach().cpu()),
        "div": float(d_loss.detach().cpu()),
        "cf_div": float(cf_d_loss.detach().cpu()),
        "action_ce": float(action_ce_loss.detach().cpu()),
        "jepa_w": float(jepa_w),
        "var_w": float(var_w),
        "cov_w": float(cov_w),
        "ln": float(ln_loss.detach().cpu()),
        "ln_w": float(ln_w),
        "nce_w": float(nce_w),
        "div_w": float(div_w),
        "cf_div_w": float(cf_div_w),
        "action_ce_w": float(action_ce_w),
        "grad": float(grad_norm.detach().cpu()),
        "z_norm": stats["z_norm"],
        "z_std": stats["z_std"],
    }


# ============================
# Planning Probe
# ============================

@torch.no_grad()
def planning_probe(model):
    """
    This is not training.
    This is a probe to see whether the model can attach different predicted
    consequence latents to different candidate actions.

    It samples K actions from the same prefix, then predicts future latents
    for each action.
    """

    model.eval()

    prefix_chars = config["planning_prefix_chars"]

    if len(val_data) > prefix_chars + 1:
        start = torch.randint(0, len(val_data) - prefix_chars - 1, (1,)).item()
        prefix = val_data[start:start + prefix_chars].unsqueeze(0).to(device)
    else:
        prefix = train_data[:prefix_chars].unsqueeze(0).to(device)

    prefix_text = decode(prefix[0].tolist())

    candidates = []
    latents = []

    for k in range(config["planning_candidates"]):
        generated = model.generate(
            prefix.clone(),
            max_new_tokens=config["planning_action_tokens"],
            temperature=config["temperature"],
            top_k_val=config["top_k"],
        )

        action = generated[:, prefix.size(1):]

        visible = visible_prefix_action(generated)

        z_pred = model.predict_future_latents(visible)
        z_final = z_pred[:, -1, :]
        latents.append(z_final.squeeze(0))

        z_norm = float(z_final.norm(dim=-1).mean().detach().cpu())
        z_std = float(z_pred.reshape(-1, z_pred.size(-1)).std(dim=0).mean().detach().cpu())

        candidate_text = decode(action[0].tolist())

        candidates.append({
            "text": candidate_text,
            "z_norm": z_norm,
            "z_std": z_std,
        })

    print("\n----- planning probe -----")
    print("prefix:")
    print(prefix_text[-500:])
    print("")

    for i, cand in enumerate(candidates):
        cleaned = cand["text"].replace("\n", "\\n")
        print(
            f"[candidate {i + 1}] "
            f"z_norm={cand['z_norm']:.3f} "
            f"z_std={cand['z_std']:.3f}"
        )
        print(cleaned[:500])
        print("")

    if len(latents) > 1:
        Z = torch.stack(latents, dim=0)
        Z = F.normalize(Z, dim=-1)
        S = Z @ Z.T
        mask = ~torch.eye(S.size(0), dtype=torch.bool, device=S.device)
        off = S[mask]
        print(
            "candidate latent cosine: "
            f"mean={off.mean().item():.3f} "
            f"min={off.min().item():.3f} "
            f"max={off.max().item():.3f}"
        )

    print("--------------------------\n")

    model.train()


@torch.no_grad()
def planning_probe_summary(model, num_prefixes=8):
    """
    Aggregates candidate-action latent separation across multiple prefixes.
    This is less noisy than trusting one sampled planning prefix.
    """

    model.eval()

    means = []
    mins = []
    maxs = []
    zstds = []

    for _ in range(num_prefixes):
        prefix_chars = config["planning_prefix_chars"]

        if len(val_data) > prefix_chars + 1:
            start = torch.randint(0, len(val_data) - prefix_chars - 1, (1,)).item()
            prefix = val_data[start:start + prefix_chars].unsqueeze(0).to(device)
        else:
            prefix = train_data[:prefix_chars].unsqueeze(0).to(device)

        latents = []

        for _ in range(config["planning_candidates"]):
            generated = model.generate(
                prefix.clone(),
                max_new_tokens=config["planning_action_tokens"],
                temperature=config["temperature"],
                top_k_val=config["top_k"],
            )

            visible = visible_prefix_action(generated)

            z_pred = model.predict_future_latents(visible)
            z_final = z_pred[:, -1, :]
            latents.append(z_final.squeeze(0))

        Z = torch.stack(latents, dim=0)
        zstds.append(float(Z.std(dim=0).mean().detach().cpu()))

        Z = F.normalize(Z, dim=-1)
        S = Z @ Z.T
        off = S[~torch.eye(S.size(0), dtype=torch.bool, device=S.device)]

        means.append(float(off.mean().detach().cpu()))
        mins.append(float(off.min().detach().cpu()))
        maxs.append(float(off.max().detach().cpu()))

    print("\n----- planning summary -----")
    print(
        f"cos_mean={sum(means) / len(means):.3f} "
        f"cos_min_avg={sum(mins) / len(mins):.3f} "
        f"cos_min_best={min(mins):.3f} "
        f"cos_max_avg={sum(maxs) / len(maxs):.3f} "
        f"zstd_avg={sum(zstds) / len(zstds):.3f}"
    )
    print("----------------------------\n")

    model.train()


@torch.no_grad()
def rank_candidate_actions(
    model,
    prefix,
    action_tokens=64,
    candidates=16,
    temperature=0.8,
    top_k_val=80,
):
    model.eval()

    idx = encode(prefix).unsqueeze(0).to(device)

    if idx.numel() == 0:
        idx = torch.randint(0, vocab_size, (1, 1), device=device)

    rows = []
    cand_latents = []

    for _ in range(candidates):
        cand = model.generate(
            idx.clone(),
            max_new_tokens=action_tokens,
            temperature=temperature,
            top_k_val=top_k_val,
        )

        action = cand[0, idx.size(1):]
        action_text = decode(action.tolist())
        visible = visible_prefix_action(cand)
        z_all = model.predict_future_latents(visible)
        z_pred = z_all[:, -1, :]
        nll = continuation_nll(model, cand, idx.size(1))
        deg = text_degeneracy_penalty(action)
        horizon_shift = horizon_shift_distance(z_all)

        cand_latents.append(z_pred.squeeze(0))
        rows.append({
            "text": action_text,
            "nll": nll,
            "deg": deg,
            "horizon_shift": horizon_shift,
        })

    if len(cand_latents) > 1:
        Z = torch.stack(cand_latents, dim=0)
        Z_n = F.normalize(Z, dim=-1)
        S = Z_n @ Z_n.T

        for i in range(candidates):
            others = torch.cat([S[i, :i], S[i, i + 1:]], dim=0)
            novelty = float((1.0 - others.mean()).detach().cpu())
            rows[i]["novelty"] = novelty
            rows[i]["score"] = (
                -1.00 * rows[i]["nll"]
                + config["planned_diversity_bonus"] * novelty
                - config["planned_degeneracy_penalty_weight"] * rows[i]["deg"]
                - config["planned_horizon_instability_weight"] * rows[i]["horizon_shift"]
            )
    else:
        rows[0]["novelty"] = 0.0
        rows[0]["score"] = (
            -rows[0]["nll"]
            - config["planned_degeneracy_penalty_weight"] * rows[0]["deg"]
            - config["planned_horizon_instability_weight"] * rows[0]["horizon_shift"]
        )

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)

    print("\n----- ranked candidate actions -----")
    print("prefix:")
    print(prefix[-500:])
    print("")
    print("TOP")

    for r in rows[:3]:
        print(
            f"score={r['score']:.3f} "
            f"nll={r['nll']:.3f} "
            f"novelty={r['novelty']:.3f} "
            f"deg={r['deg']:.3f} "
            f"hshift={r['horizon_shift']:.3f}"
        )
        print(r["text"].replace("\n", "\\n")[:500])
        print("")

    print("BOTTOM")

    for r in rows[-3:]:
        print(
            f"score={r['score']:.3f} "
            f"nll={r['nll']:.3f} "
            f"novelty={r['novelty']:.3f} "
            f"deg={r['deg']:.3f} "
            f"hshift={r['horizon_shift']:.3f}"
        )
        print(r["text"].replace("\n", "\\n")[:500])
        print("")

    print("------------------------------------\n")
    model.train()


def pairwise_cosine_stats(z):
    if z.size(0) <= 1:
        return {
            "cos_mean": 1.0,
            "cos_min": 1.0,
            "cos_max": 1.0,
            "dist_mean": 0.0,
        }

    z_n = F.normalize(z, dim=-1)
    sim = z_n @ z_n.T
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    off = sim[mask]

    return {
        "cos_mean": float(off.mean().detach().cpu()),
        "cos_min": float(off.min().detach().cpu()),
        "cos_max": float(off.max().detach().cpu()),
        "dist_mean": float((1.0 - off).mean().detach().cpu()),
    }


@torch.no_grad()
def world_model_diagnostics(student, teacher, split="val", batches=10):
    student_was_training = student.training
    teacher_was_training = teacher.training

    student.eval()
    teacher.eval()

    true_jepas = []
    shuffled_jepas = []
    action_sensitivities = []
    context_sensitivities = []
    cand_cos_means = []
    cand_cos_mins = []
    cand_cos_maxs = []
    horizon_cos_dists = []

    per_horizon_true_jepas = {h: [] for h in config["future_horizons"]}
    per_horizon_shuffled_jepas = {h: [] for h in config["future_horizons"]}

    for _ in range(max(1, batches)):
        batch = get_world_batch(split)
        z_pred = student.predict_future_latents(batch["student_visible"])
        z_teacher = teacher_delta_future_latents(teacher, batch["teacher_visible"])

        true_jepas.append(jepa_prediction_loss(z_pred, z_teacher).item())

        true_jepa_per_h = jepa_prediction_loss_per_horizon(z_pred, z_teacher)
        for h, loss in true_jepa_per_h.items():
            per_horizon_true_jepas[h].append(loss)

        if z_teacher.size(0) > 1:
            shuffle_ix = torch.randperm(z_teacher.size(0), device=z_teacher.device)
            shuffled_teacher = z_teacher[shuffle_ix]
        else:
            shuffled_teacher = z_teacher

        shuffled_jepas.append(jepa_prediction_loss(z_pred, shuffled_teacher).item())

        shuffled_jepa_per_h = jepa_prediction_loss_per_horizon(z_pred, shuffled_teacher)
        for h, loss in shuffled_jepa_per_h.items():
            per_horizon_shuffled_jepas[h].append(loss)

        prefix = batch["prefix"][:1]
        action_latents = []

        for _ in range(max(2, config["planning_candidates"])):
            generated = student.generate(
                prefix.clone(),
                max_new_tokens=config["action_len"],
                temperature=config["model_negative_temperature"],
                top_k_val=config["model_negative_top_k"],
            )
            visible = visible_prefix_action(generated)
            z_action = student.predict_future_latents(visible)[:, -1, :]
            action_latents.append(z_action.squeeze(0))

        action_stats = pairwise_cosine_stats(torch.stack(action_latents, dim=0))
        action_sensitivities.append(action_stats["dist_mean"])

        shared_action = batch["action"][:1].repeat(batch["prefix"].size(0), 1)
        context_visible = torch.cat([batch["prefix"], shared_action], dim=1)
        context_z = student.predict_future_latents(context_visible)[:, -1, :]
        context_stats = pairwise_cosine_stats(context_z)
        context_sensitivities.append(context_stats["dist_mean"])

        candidate_latents = []

        for _ in range(max(2, config["planning_candidates"])):
            generated = student.generate(
                prefix.clone(),
                max_new_tokens=config["planning_action_tokens"],
                temperature=config["planned_temperature"],
                top_k_val=config["planned_top_k"],
            )
            visible = visible_prefix_action(generated)
            candidate_latents.append(student.predict_future_latents(visible)[:, -1, :].squeeze(0))

        candidate_stats = pairwise_cosine_stats(torch.stack(candidate_latents, dim=0))
        cand_cos_means.append(candidate_stats["cos_mean"])
        cand_cos_mins.append(candidate_stats["cos_min"])
        cand_cos_maxs.append(candidate_stats["cos_max"])

        horizon_cos_dists.append(horizon_cosine_distances(z_pred))

    if student_was_training:
        student.train()
    if teacher_was_training:
        teacher.train()

    true_jepa = sum(true_jepas) / len(true_jepas)
    shuffled_future_jepa = sum(shuffled_jepas) / len(shuffled_jepas)

    per_horizon_results = {}
    for h in config["future_horizons"]:
        tj = sum(per_horizon_true_jepas[h]) / len(per_horizon_true_jepas[h])
        sj = sum(per_horizon_shuffled_jepas[h]) / len(per_horizon_shuffled_jepas[h])
        per_horizon_results[h] = {
            "true_jepa": tj,
            "shuffled_jepa": sj,
            "match_gap": sj - tj,
        }

    return {
        "true_jepa": true_jepa,
        "shuffled_future_jepa": shuffled_future_jepa,
        "match_gap": shuffled_future_jepa - true_jepa,
        "action_sensitivity": sum(action_sensitivities) / len(action_sensitivities),
        "context_sensitivity": sum(context_sensitivities) / len(context_sensitivities),
        "candidate_cos_mean": sum(cand_cos_means) / len(cand_cos_means),
        "candidate_cos_min": min(cand_cos_mins),
        "candidate_cos_max": max(cand_cos_maxs),
        "horizon_cos_dist": sum(horizon_cos_dists) / len(horizon_cos_dists),
        "per_horizon": per_horizon_results,
    }


# ============================
# Checkpointing
# ============================

def world_score(world_stats):
    val = world_stats["val"]

    return (
        val["jepa"]
        + 0.25 * val["nce"]
        + 0.10 * val["var"]
        + 0.01 * val["cov"]
        + 0.50 * val["cf_div"]
        + 0.25 * val["action_ce"]
    )


def save_checkpoint(
    path,
    student,
    teacher,
    optimizer,
    step,
    lm_stats,
    world_stats,
    score,
    diagnostics=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "step": step,
            "config": config,
            "vocab": {
                "chars": chars,
                "stoi": stoi,
                "itos": itos,
            },
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lm_stats": lm_stats,
            "world_stats": world_stats,
            "world_score": score,
            "world_diagnostics": diagnostics,
            "run_id": run_id,
            "training_log_path": training_log_path,
        },
        path,
    )


# ============================
# Logging
# ============================

def print_train_log(step, log):
    ace_flag = "*" if log["action_ce_w"] > 0 else ""
    print(
        f"[step {step:5d}] "
        f"loss={log['loss']:.4f} "
        f"token={log['token']:.4f} "
        f"jepa={log['jepa']:.4f} "
        f"var={log['var']:.4f} "
        f"cov={log['cov']:.4f} "
        f"ln={log['ln']:.4f} "
        f"nce={log['nce']:.4f} "
        f"div={log['div']:.4f} "
        f"cf_div={log['cf_div']:.4f} "
        f"action_ce={log['action_ce']:.4f}{ace_flag} "
        f"jw={log['jepa_w']:.3f} "
        f"vw={log['var_w']:.3f} "
        f"cw={log['cov_w']:.4f} "
        f"nw={log['nce_w']:.3f} "
        f"dw={log['div_w']:.3f} "
        f"cfw={log['cf_div_w']:.3f} "
        f"acew={log['action_ce_w']:.3f} "
        f"z_norm={log['z_norm']:.3f} "
        f"z_std={log['z_std']:.3f} "
        f"grad={log['grad']:.2f}"
    )


def print_eval(step, lm_stats, world_stats, best_lm, best_world, current_world_score):
    print(f"\n===== eval step {step} =====")

    print("LM loss")
    print("split | lm")
    print("------|--------")
    print(f"train | {lm_stats['train']:.4f}")
    print(f"val   | {lm_stats['val']:.4f}")
    print(f"best  | {best_lm['val_lm']:.4f} @ step {best_lm['step']}")
    print("")

    print("World-model loss")
    print("split | jepa   | var    | cov    | nce    | div    | cf_div | action_ce")
    print("------|--------|--------|--------|--------|--------|--------|----------")
    print(
        f"train | {world_stats['train']['jepa']:.4f} "
        f"| {world_stats['train']['var']:.4f} "
        f"| {world_stats['train']['cov']:.4f} "
        f"| {world_stats['train']['nce']:.4f} "
        f"| {world_stats['train']['div']:.4f} "
        f"| {world_stats['train']['cf_div']:.4f} "
        f"| {world_stats['train']['action_ce']:.4f}"
    )
    print(
        f"val   | {world_stats['val']['jepa']:.4f} "
        f"| {world_stats['val']['var']:.4f} "
        f"| {world_stats['val']['cov']:.4f} "
        f"| {world_stats['val']['nce']:.4f} "
        f"| {world_stats['val']['div']:.4f} "
        f"| {world_stats['val']['cf_div']:.4f} "
        f"| {world_stats['val']['action_ce']:.4f}"
    )
    print(
        f"world_score | current={current_world_score:.4f} "
        f"| best={best_world['score']:.4f} @ step {best_world['step']}"
    )
    print("")


def print_world_diagnostics(diagnostics):
    print("World diagnostics")
    print(
        "true_jepa | shuffled_jepa | match_gap | action_sens | context_sens "
        "| cand_cos_mean | cand_cos_min | cand_cos_max | h_cos_dist"
    )
    print(
        "----------|---------------|-----------|-------------|--------------"
        "|---------------|--------------|-------------|------------"
    )
    print(
        f"{diagnostics['true_jepa']:.4f} "
        f"| {diagnostics['shuffled_future_jepa']:.4f} "
        f"| {diagnostics['match_gap']:.4f} "
        f"| {diagnostics['action_sensitivity']:.4f} "
        f"| {diagnostics['context_sensitivity']:.4f} "
        f"| {diagnostics['candidate_cos_mean']:.4f} "
        f"| {diagnostics['candidate_cos_min']:.4f} "
        f"| {diagnostics['candidate_cos_max']:.4f} "
        f"| {diagnostics['horizon_cos_dist']:.4f}"
    )
    print("Per-horizon JEPA:")
    print("horizon | true_jepa | shuffled_jepa | match_gap")
    print("--------|-----------|---------------|----------")
    for h in config["future_horizons"]:
        ph = diagnostics["per_horizon"][h]
        print(
            f"h{h:4d}   | {ph['true_jepa']:.4f}    | {ph['shuffled_jepa']:.4f}       | {ph['match_gap']:.4f}"
        )
    print("")


# ============================
# Main
# ============================

student, teacher = make_models()

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)

n_params = sum(p.numel() for p in student.parameters())

print(f"params: {n_params:,}")
print(f"vocab_size: {vocab_size}")
print(f"prefix_len: {config['prefix_len']}")
print(f"action_len: {config['action_len']}")
print(f"future_len: {config['future_len']}")
print(f"block_size: {config['block_size']}")
print(f"latent_dim: {config['latent_dim']}")
print(f"future_horizons: {config['future_horizons']}")
print(f"ema_decay: {config['ema_decay']}")
print(f"token_weight: {config['token_weight']}")
print(f"jepa_weight: {config['jepa_weight']}")
print(f"variance_weight: {config['variance_weight']}")
print(f"covariance_weight: {config['covariance_weight']}")
print(f"contrastive_weight: {config['contrastive_weight']}")
print(f"contrastive_temp: {config['contrastive_temp']}")
print(f"diversity_weight: {config['diversity_weight']}")
print(f"counterfactual_diversity_weight: {config['counterfactual_diversity_weight']}")
print(f"counterfactual_group_size: {config['counterfactual_group_size']}")
print(f"counterfactual_margin: {config['counterfactual_margin']}")
print(f"action_ce_weight: {config['action_ce_weight']}")
print(f"counterfactual_negative_mode: {config['counterfactual_negative_mode']}")
print(f"model_negative_temperature: {config['model_negative_temperature']}")
print(f"model_negative_top_k: {config['model_negative_top_k']}")
print(f"jepa_warmup_steps: {config['jepa_warmup_steps']}")
print(f"jepa_ramp_steps: {config['jepa_ramp_steps']}")
print(f"checkpoint_dir: {config['checkpoint_dir']}")
print(f"planned_generation_prefix: {config['planned_generation_prefix']!r}")
print(f"planned_candidates: {config['planned_candidates']}")
print(f"planned_action_tokens: {config['planned_action_tokens']}")
print(f"planned_diversity_bonus: {config['planned_diversity_bonus']}")
print(f"planned_degeneracy_penalty_weight: {config['planned_degeneracy_penalty_weight']}")
print(f"planned_horizon_instability_weight: {config['planned_horizon_instability_weight']}")
print(f"rank_candidates: {config['rank_candidates']}")
print(f"rank_action_tokens: {config['rank_action_tokens']}")
print("mode: latent text world model")
print("teacher: sees prefix + action + future, provides delta future targets")
print("student: sees prefix + action, predicts future consequence deltas")
print("rl: absent, mercifully")

best_lm = {
    "val_lm": float("inf"),
    "train_lm": float("inf"),
    "step": 0,
}

best_world = {
    "score": float("inf"),
    "step": 0,
}

student.train()

for step in range(1, config["max_steps"] + 1):
    log = train_step(student, teacher, optimizer, step)

    if step % config["eval_interval"] == 0 or step == 1:
        print_train_log(step, log)

        lm_stats = estimate_lm_loss(student)
        world_stats = estimate_world_loss(student, teacher)
        diagnostics = world_model_diagnostics(student, teacher, split="val", batches=10)
        current_world_score = world_score(world_stats)

        if (
            step >= config["action_ce_step_threshold"]
            or diagnostics["match_gap"] > config["action_ce_match_gap_threshold"]
        ):
            action_ce_enabled = True

        if lm_stats["val"] < best_lm["val_lm"]:
            best_lm = {
                "val_lm": lm_stats["val"],
                "train_lm": lm_stats["train"],
                "step": step,
            }
            save_checkpoint(
                path=os.path.join(config["checkpoint_dir"], config["best_lm_checkpoint_name"]),
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                step=step,
                lm_stats=lm_stats,
                world_stats=world_stats,
                score=current_world_score,
                diagnostics=diagnostics,
            )

        if current_world_score < best_world["score"]:
            best_world = {
                "score": current_world_score,
                "step": step,
            }
            save_checkpoint(
                path=os.path.join(config["checkpoint_dir"], config["best_world_checkpoint_name"]),
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                step=step,
                lm_stats=lm_stats,
                world_stats=world_stats,
                score=current_world_score,
                diagnostics=diagnostics,
            )

        print_eval(step, lm_stats, world_stats, best_lm, best_world, current_world_score)

        planning_probe(student)
        planning_probe_summary(student, num_prefixes=8)

        print("----- sample -----")
        sample_text(student, prefix="", steps=config["sample_tokens"])
        print("------------------\n")

        print("----- normal sample -----")
        sample_text(
            student,
            prefix=config["planned_generation_prefix"],
            steps=config["planned_sample_tokens"],
        )
        print("----- planned sample -----")
        print(planned_generate(
            student,
            prefix=config["planned_generation_prefix"],
            total_new_tokens=config["planned_sample_tokens"],
            action_tokens=config["planned_action_tokens"],
            candidates=config["planned_candidates"],
            temperature=config["planned_temperature"],
            top_k_val=config["planned_top_k"],
            diversity_bonus=config["planned_diversity_bonus"],
            degeneracy_penalty_weight=config["planned_degeneracy_penalty_weight"],
            horizon_instability_weight=config["planned_horizon_instability_weight"],
        ))
        print("--------------------------\n")

        rank_candidate_actions(
            student,
            prefix=config["planned_generation_prefix"],
            action_tokens=config["rank_action_tokens"],
            candidates=config["rank_candidates"],
            temperature=config["planned_temperature"],
            top_k_val=config["planned_top_k"],
        )
        print_world_diagnostics(diagnostics)


print("\n===== best checkpoint =====")
print("kind  | metric        | step | path")
print("------|---------------|------|-----")
print(
    f"lm    | val_lm={best_lm['val_lm']:.4f} "
    f"| {best_lm['step']} "
    f"| {os.path.join(config['checkpoint_dir'], config['best_lm_checkpoint_name'])}"
)
print(
    f"world | score={best_world['score']:.4f} "
    f"| {best_world['step']} "
    f"| {os.path.join(config['checkpoint_dir'], config['best_world_checkpoint_name'])}"
)
print(f"\ntraining_log | {training_log_path}")
print("```")
sys.stdout = original_stdout
sys.stderr = original_stderr
training_log_file.close()
