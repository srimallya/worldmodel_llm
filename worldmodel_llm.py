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
    jepa_weight=0.50,

    # VICReg-style anti-collapse terms
    variance_weight=0.15,
    covariance_weight=0.005,

    # Future discrimination terms
    contrastive_weight=0.08,
    contrastive_temp=0.15,
    diversity_weight=0.01,
    counterfactual_diversity_weight=0.02,
    counterfactual_group_size=4,
    counterfactual_margin=0.80,

    # warmup/ramp
    jepa_warmup_steps=1500,
    jepa_ramp_steps=5000,

    # teacher
    ema_decay=0.995,

    # sampling
    sample_tokens=400,
    top_k=90,
    temperature=0.8,

    # checkpointing
    checkpoint_dir="checkpoints",
    best_lm_checkpoint_name="best_lm.pt",
    best_world_checkpoint_name="best_world.pt",

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
    planned_diversity_bonus=0.05,

    seed=1337,
)


torch.manual_seed(config["seed"])


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


def get_counterfactual_action_batch(split):
    src = train_data if split == "train" else val_data

    prefix_len = config["prefix_len"]
    action_len = config["action_len"]
    group_size = config["counterfactual_group_size"]
    total = prefix_len + action_len + 1

    if group_size < 2:
        raise ValueError("counterfactual_group_size must be at least 2.")

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

        pooled = hidden.mean(dim=1)
        z_context = self.projector(pooled)

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

            cf_visible = get_counterfactual_action_batch(split)
            cf_pred = student.predict_future_latents(cf_visible)
            cf_div = grouped_action_diversity_loss(
                cf_pred,
                group_size=config["counterfactual_group_size"],
                margin=config["counterfactual_margin"],
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

        out[split] = {
            "jepa": sum(jepa_losses) / len(jepa_losses),
            "var": sum(var_losses) / len(var_losses),
            "cov": sum(cov_losses) / len(cov_losses),
            "nce": sum(nce_losses) / len(nce_losses),
            "div": sum(div_losses) / len(div_losses),
            "cf_div": sum(cf_div_losses) / len(cf_div_losses),
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
            cand = model.generate(
                idx.clone(),
                max_new_tokens=step_tokens,
                temperature=temperature,
                top_k_val=top_k_val,
            )

            visible = visible_prefix_action(cand)
            z_pred = model.predict_future_latents(visible)
            z_final = z_pred[:, -1, :]

            logits, _, _ = model(cand[:, -model.block_size:], targets=None)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits / max(temperature, 1e-6), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

            candidate_infos.append({
                "cand": cand,
                "z": z_final.squeeze(0),
                "entropy": float(entropy.detach().cpu()),
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

            score = -info["entropy"] + diversity_bonus * novelty
            scores.append(score)

        best_i = max(range(len(scores)), key=lambda i: scores[i])
        idx = candidate_infos[best_i]["cand"]

    return decode(idx[0].tolist())


def train_step(student, teacher, optimizer, step):
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

    cf_visible = get_counterfactual_action_batch("train")
    cf_pred = student.predict_future_latents(cf_visible)
    cf_d_loss = grouped_action_diversity_loss(
        cf_pred,
        group_size=config["counterfactual_group_size"],
        margin=config["counterfactual_margin"],
    )

    # ----------------------------
    # 5. Anti-collapse latent regularization
    # ----------------------------

    B, H, D = z_pred.shape
    z_flat = z_pred.reshape(B * H, D)

    v_loss = variance_loss(z_flat)
    c_loss = covariance_loss(z_flat)

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

    # ----------------------------
    # 7. Total loss
    # ----------------------------

    loss = (
        config["token_weight"] * token_loss
        + jepa_w * jepa_loss
        + var_w * v_loss
        + cov_w * c_loss
        + nce_w * nce_loss
        + div_w * d_loss
        + cf_div_w * cf_d_loss
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
        "jepa_w": float(jepa_w),
        "var_w": float(var_w),
        "cov_w": float(cov_w),
        "nce_w": float(nce_w),
        "div_w": float(div_w),
        "cf_div_w": float(cf_div_w),
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
    )


def save_checkpoint(path, student, teacher, optimizer, step, lm_stats, world_stats, score):
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
        },
        path,
    )


# ============================
# Logging
# ============================

def print_train_log(step, log):
    print(
        f"[step {step:5d}] "
        f"loss={log['loss']:.4f} "
        f"token={log['token']:.4f} "
        f"jepa={log['jepa']:.4f} "
        f"var={log['var']:.4f} "
        f"cov={log['cov']:.4f} "
        f"nce={log['nce']:.4f} "
        f"div={log['div']:.4f} "
        f"cf_div={log['cf_div']:.4f} "
        f"jw={log['jepa_w']:.3f} "
        f"vw={log['var_w']:.3f} "
        f"cw={log['cov_w']:.4f} "
        f"nw={log['nce_w']:.3f} "
        f"dw={log['div_w']:.3f} "
        f"cfw={log['cf_div_w']:.3f} "
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
    print("split | jepa   | var    | cov    | nce    | div    | cf_div")
    print("------|--------|--------|--------|--------|--------|--------")
    print(
        f"train | {world_stats['train']['jepa']:.4f} "
        f"| {world_stats['train']['var']:.4f} "
        f"| {world_stats['train']['cov']:.4f} "
        f"| {world_stats['train']['nce']:.4f} "
        f"| {world_stats['train']['div']:.4f} "
        f"| {world_stats['train']['cf_div']:.4f}"
    )
    print(
        f"val   | {world_stats['val']['jepa']:.4f} "
        f"| {world_stats['val']['var']:.4f} "
        f"| {world_stats['val']['cov']:.4f} "
        f"| {world_stats['val']['nce']:.4f} "
        f"| {world_stats['val']['div']:.4f} "
        f"| {world_stats['val']['cf_div']:.4f}"
    )
    print(
        f"world_score | current={current_world_score:.4f} "
        f"| best={best_world['score']:.4f} @ step {best_world['step']}"
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
print(f"jepa_warmup_steps: {config['jepa_warmup_steps']}")
print(f"jepa_ramp_steps: {config['jepa_ramp_steps']}")
print(f"checkpoint_dir: {config['checkpoint_dir']}")
print(f"planned_generation_prefix: {config['planned_generation_prefix']!r}")
print(f"planned_candidates: {config['planned_candidates']}")
print(f"planned_action_tokens: {config['planned_action_tokens']}")
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
        current_world_score = world_score(world_stats)

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
            )

        print_eval(step, lm_stats, world_stats, best_lm, best_world, current_world_score)

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
        ))
        print("--------------------------\n")

        planning_probe(student)
        planning_probe_summary(student, num_prefixes=8)


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
