"""Pretrain a GPT-2 (124M) style model on FineWeb-Edu, following Karpathy's build-nanogpt.

Additions over the lecture version: the data is reshuffled every epoch (shard order and the
order of documents inside each shard), training runs for a fixed number of epochs, and a
checkpoint is saved at the end of every epoch.

Single process:  python train_tiny_llama.py
DDP:             torchrun --standalone --nproc_per_node=8 train_tiny_llama.py

Reads the token shards from <repo>/resources/edu_fineweb10B (and, through 04_gpt2/hellaswag.py,
the HellaSwag data from <repo>/resources/hellaswag). Writes the metrics log (parsed by plot_log.ipynb)
and the checkpoints to log/ next to this script.
"""
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESOURCES_DIR = os.path.join(REPO_ROOT, "resources")  # shared datasets, not in git

sys.path.append(os.path.join(REPO_ROOT, "04_gpt2"))  # reuse the HellaSwag eval from the GPT-2 lecture
from hellaswag import iterate_examples, render_example  # noqa: E402


@dataclass
class TrainConfig:
    data_root: str = os.path.join(RESOURCES_DIR, "edu_fineweb10B")
    log_dir: str = os.path.join(SCRIPT_DIR, "log")
    # batch: fixed ~0.5M tokens per optimizer step, no ramp (as in build-nanogpt)
    total_batch_size: int = 524288  # 2**19 tokens, reached with gradient accumulation
    micro_batch_size: int = 32      # sequences per GPU per forward (Karpathy: 64 on 80GB, 32 fits 40GB)
    seq_len: int = 1024
    num_epochs: int = 5
    # optimizer: linear warmup to max_lr, then cosine decay to 10% of it at the last step
    max_lr: float = 6e-4
    warmup_steps: int = 715         # ~375M tokens, the GPT-3 warmup
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # evaluation (val loss, HellaSwag, samples) every eval_interval steps and before the last step
    eval_interval: int = 250
    val_loss_steps: int = 20
    use_compile: bool = True
    seed: int = 42                  # model init and sampling
    data_seed: int = 1337           # data shuffling

    @property
    def min_lr(self):
        return self.max_lr * 0.1


# -----------------------------------------------------------------------------
# Model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # q, k, v for all heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble the heads
        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # projections back into the residual stream: keep its variance from growing with depth
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)  # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=False):
        """AdamW that decays only the 2D+ tensors (matmul weights, embeddings), not biases or LayerNorms."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        if verbose:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
            print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)


# -----------------------------------------------------------------------------
# Data

EOT_TOKEN = 50256  # <|endoftext|>, the first token of every document


def shuffle_documents(tokens, rng):
    """Randomly reorders the documents in a shard. Documents start at EOT tokens. The piece
    before the first EOT (the tail of a document that began in the previous shard) is dropped,
    so the result is leading_fragment_len(...) tokens shorter than the shard."""
    starts = np.flatnonzero(tokens == EOT_TOKEN)
    docs = np.split(tokens, starts)[1:]  # [0] is the leading fragment (empty if the shard starts with EOT)
    docs = [docs[i] for i in rng.permutation(len(docs))]
    return np.concatenate(docs)


def leading_fragment_len(filename, chunk=1 << 20):
    """Number of tokens before the first EOT in a shard, read via mmap without loading the shard."""
    a = np.load(filename, mmap_mode="r")
    for start in range(0, len(a), chunk):
        hits = np.flatnonzero(np.asarray(a[start:start + chunk]) == EOT_TOKEN)
        if len(hits):
            return start + int(hits[0])
    return len(a)


class DataLoaderLite:
    """Streams (B, T) batches from the token shards of one split.

    All ranks share current_position; each rank reads its own B*T slice after it. A shard's tail
    too short for a full batch on every rank is skipped. `epoch` counts completed passes over
    all shards.

    With shuffle=True, every epoch (including the first) uses a new random shard order and a random
    document order within each shard; each shard's leading partial document is dropped. Both are
    seeded by (seed, epoch[, shard]), so every rank sees exactly the same data order.
    """

    def __init__(self, data_root, split, B, T, process_rank, num_processes, shuffle=False, seed=1337):
        assert split in ("train", "val")
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.global_batch_tokens = B * T * num_processes  # what one next_batch() consumes across all ranks
        self.shuffle = shuffle
        self.seed = seed

        self.shards = sorted(os.path.join(data_root, s) for s in os.listdir(data_root) if split in s)
        assert len(self.shards) > 0, f"no shards found for split {split}"
        self.shard_lens = [len(np.load(s, mmap_mode="r")) for s in self.shards]  # header only, cheap
        if shuffle:  # shuffling drops each shard's leading fragment (see shuffle_documents)
            self.shard_lens = [n - leading_fragment_len(s) for n, s in zip(self.shard_lens, self.shards)]
        assert all(self._fits(0, n) for n in self.shard_lens), "every shard must hold at least one batch"
        self.reset()

    @property
    def batches_per_epoch(self):
        """Number of next_batch() calls in one pass over all shards (the same for every epoch)."""
        return sum((n - 1) // self.global_batch_tokens for n in self.shard_lens)

    def reset(self):
        self.epoch = 0
        self._start_epoch()

    def next_batch(self):
        """Returns this rank's inputs and targets (x, y), both (B, T); y is x shifted by one token."""
        if self.tokens is None:
            self.tokens = self._load_current_shard()
        B, T = self.B, self.T
        start = self.current_position + B * T * self.process_rank
        buf = self.tokens[start : start + B * T + 1]
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        self.current_position += self.global_batch_tokens
        # move on as soon as the next batch doesn't fit, so an epoch ends on its last batch
        if not self._fits(self.current_position, self.shard_lens[self.shard_order[self.current_shard]]):
            self._next_shard()
        return x, y

    def _fits(self, position, shard_len):
        # the last rank reads up to position + global_batch_tokens + 1 (the extra token is the last target)
        return position + self.global_batch_tokens + 1 <= shard_len

    def _start_epoch(self):
        if self.shuffle:
            self.shard_order = np.random.default_rng((self.seed, self.epoch)).permutation(len(self.shards))
        else:
            self.shard_order = np.arange(len(self.shards))
        self.current_shard = 0  # index into shard_order
        self.current_position = 0
        self.tokens = None  # loaded on the first next_batch()

    def _next_shard(self):
        if self.current_shard + 1 == len(self.shards):
            self.epoch += 1
            self._start_epoch()
        else:
            self.current_shard += 1
            self.current_position = 0
            self.tokens = None

    def _load_current_shard(self):
        shard_idx = int(self.shard_order[self.current_shard])
        tokens = np.load(self.shards[shard_idx])
        if self.shuffle:
            tokens = shuffle_documents(tokens, np.random.default_rng((self.seed, self.epoch, shard_idx)))
        return torch.tensor(tokens, dtype=torch.long)


# -----------------------------------------------------------------------------
# Distributed setup

@dataclass
class DistEnv:
    ddp: bool
    rank: int
    local_rank: int
    world_size: int
    device: str

    @property
    def is_master(self):
        return self.rank == 0

    @property
    def device_type(self):
        return "cuda" if self.device.startswith("cuda") else "cpu"

    def autocast(self):
        return torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)

    def all_reduce(self, tensor, op=dist.ReduceOp.AVG):
        if self.ddp:
            dist.all_reduce(tensor, op=op)
        return tensor

    def print0(self, *args):
        if self.is_master:
            print(*args)


def setup_distributed():
    """Joins the DDP process group when launched by torchrun, otherwise picks the best single device."""
    if int(os.environ.get("RANK", -1)) == -1:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"using device: {device}")
        return DistEnv(ddp=False, rank=0, local_rank=0, world_size=1, device=device)

    assert torch.cuda.is_available(), "DDP training requires CUDA"
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    return DistEnv(ddp=True, rank=int(os.environ["RANK"]), local_rank=local_rank,
                   world_size=int(os.environ["WORLD_SIZE"]), device=device)


# -----------------------------------------------------------------------------
# Schedule

def plan_epochs(batches_per_epoch, grad_accum_steps, num_epochs):
    """Returns the total number of steps and {step: epoch} for the steps on which an epoch ends.
    Epochs don't have to end on step boundaries: a step may finish one epoch and start the next."""
    def last_step_of(epoch):  # the step that consumes the last batch of `epoch`
        return (epoch * batches_per_epoch - 1) // grad_accum_steps
    epoch_end_steps = {last_step_of(epoch): epoch for epoch in range(1, num_epochs + 1)}
    return last_step_of(num_epochs) + 1, epoch_end_steps


def get_lr(step, max_steps, cfg):
    """Linear warmup to max_lr, then cosine decay that would reach min_lr at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    decay_ratio = (step - cfg.warmup_steps) / (max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)


# -----------------------------------------------------------------------------
# Training and evaluation

def train_step(model, optimizer, train_loader, grad_accum_steps, lr, grad_clip, env):
    """One optimizer step over grad_accum_steps micro-batches. Returns (mean loss, grad norm)."""
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(env.device), y.to(env.device)
        with env.autocast():
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if env.ddp:  # all-reduce the gradients only on the last micro-step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    env.all_reduce(loss_accum)
    norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    return loss_accum.item(), norm.item()


@torch.no_grad()
def evaluate_val_loss(model, val_loader, num_steps, env):
    """Mean loss over the first num_steps validation batches (the same ones every time)."""
    model.eval()
    val_loader.reset()
    loss_accum = 0.0
    for _ in range(num_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(env.device), y.to(env.device)
        with env.autocast():
            _, loss = model(x, y)
        loss_accum += loss.detach() / num_steps
    env.all_reduce(loss_accum)
    return loss_accum.item()


def get_most_likely_row(tokens, mask, logits):
    """Index of the completion with the lowest average per-token loss (HellaSwag acc_norm)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_tokens.view(-1), reduction="none")
    losses = losses.view(tokens.size(0), -1)
    # the mask is shifted too, so it starts at the last prompt token (which predicts the first completion token)
    shift_mask = mask[..., 1:].contiguous()
    avg_loss = (losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()


@torch.no_grad()
def evaluate_hellaswag(model, env):
    """Returns (num_correct, num_total) on HellaSwag val, with the examples split across ranks."""
    model.eval()
    num_correct, num_total = 0, 0
    for i, example in enumerate(iterate_examples("val")):
        if i % env.world_size != env.rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(env.device), mask.to(env.device)
        with env.autocast():
            logits, _ = model(tokens)
        num_correct += int(get_most_likely_row(tokens, mask, logits) == label)
        num_total += 1
    counts = torch.tensor([num_correct, num_total], dtype=torch.long, device=env.device)
    env.all_reduce(counts, op=dist.ReduceOp.SUM)
    num_correct, num_total = counts.tolist()
    return num_correct, num_total


@torch.no_grad()
def generate(model, enc, prompt, num_samples, max_length, seed, env):
    """Top-50 sampling of num_samples continuations of prompt, max_length tokens in total each."""
    model.eval()
    x = torch.tensor(enc.encode(prompt), dtype=torch.long, device=env.device)
    x = x.unsqueeze(0).repeat(num_samples, 1)
    rng = torch.Generator(device=env.device)
    rng.manual_seed(seed)
    while x.size(1) < max_length:
        with env.autocast():
            logits, _ = model(x)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=rng)
        x = torch.cat((x, torch.gather(topk_indices, -1, ix)), dim=1)
    return [enc.decode(row.tolist()) for row in x]


def save_checkpoint(path, model, optimizer, step, epoch, tokens_seen, val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.config,
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "val_loss": val_loss,  # from the most recent eval
    }, path)


# -----------------------------------------------------------------------------

def main(cfg, model_config):
    env = setup_distributed()
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    enc = tiktoken.get_encoding("gpt2")

    tokens_per_micro_step = cfg.micro_batch_size * cfg.seq_len * env.world_size
    assert cfg.total_batch_size % tokens_per_micro_step == 0
    grad_accum_steps = cfg.total_batch_size // tokens_per_micro_step
    env.print0(f"total batch size: {cfg.total_batch_size}, grad accumulation steps: {grad_accum_steps}")

    loaders = {}
    for split in ("train", "val"):
        loaders[split] = DataLoaderLite(cfg.data_root, split, cfg.micro_batch_size, cfg.seq_len,
                                        env.rank, env.world_size, shuffle=(split == "train"), seed=cfg.data_seed)
        env.print0(f"found {len(loaders[split].shards)} shards for split {split}")
    train_loader, val_loader = loaders["train"], loaders["val"]

    max_steps, epoch_end_steps = plan_epochs(train_loader.batches_per_epoch, grad_accum_steps, cfg.num_epochs)
    assert max_steps > cfg.warmup_steps, "the run is shorter than the LR warmup"
    env.print0(f"epochs: {cfg.num_epochs}, total tokens: {max_steps * cfg.total_batch_size:,}, steps: {max_steps}")
    env.print0(f"lr: warmup {cfg.warmup_steps} steps to {cfg.max_lr:.1e}, cosine to {cfg.min_lr:.1e} at step {max_steps}")
    env.print0(f"epoch ends (checkpoints) at steps: {sorted(epoch_end_steps)}")

    model = GPT(model_config).to(env.device)
    # raw_model: the plain eager module (no compile, no DDP), sharing parameters with `model`. Used for
    # the optimizer, checkpoints, and the evals with variable sequence lengths (HellaSwag, sampling),
    # which would otherwise trigger repeated recompiles.
    raw_model = model
    if cfg.use_compile:
        model = torch.compile(model)
    if env.ddp:
        model = DDP(model, device_ids=[env.local_rank])
    optimizer = raw_model.configure_optimizers(cfg.weight_decay, cfg.max_lr, env.device_type, verbose=env.is_master)

    log_file = os.path.join(cfg.log_dir, "log.txt")
    if env.is_master:
        os.makedirs(cfg.log_dir, exist_ok=True)
        open(log_file, "w").close()

    def log(line):  # metrics log, one "<step> <train|val|hella> <value>" line per entry
        if env.is_master:
            with open(log_file, "a") as f:
                f.write(line + "\n")

    val_loss = None
    for step in range(max_steps):
        last_step = step == max_steps - 1

        if step % cfg.eval_interval == 0 or last_step:
            val_loss = evaluate_val_loss(model, val_loader, cfg.val_loss_steps, env)
            env.print0(f"validation loss: {val_loss:.4f}")
            log(f"{step} val {val_loss:.4f}")

            num_correct, num_total = evaluate_hellaswag(raw_model, env)
            acc_norm = num_correct / num_total
            env.print0(f"HellaSwag accuracy: {num_correct}/{num_total}={acc_norm:.4f}")
            log(f"{step} hella {acc_norm:.4f}")

            if step > 0:
                samples = generate(raw_model, enc, "Hello, I'm a language model,", num_samples=4,
                                   max_length=32, seed=cfg.seed + env.rank, env=env)
                for i, text in enumerate(samples):
                    print(f"rank {env.rank} sample {i}: {text}")

        t0 = time.time()
        lr = get_lr(step, max_steps, cfg)
        loss, norm = train_step(model, optimizer, train_loader, grad_accum_steps, lr, cfg.grad_clip, env)
        if env.device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        env.print0(f"step {step:5d} | loss: {loss:.6f} | lr: {lr:.4e} | batch: {cfg.total_batch_size:,} | "
                   f"norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/s: {cfg.total_batch_size / dt:.2f}")
        log(f"{step} train {loss:.6f}")

        if step in epoch_end_steps:
            epoch = epoch_end_steps[step]
            assert train_loader.epoch == epoch, "data loader out of sync with the epoch plan"
            if env.is_master:
                path = os.path.join(cfg.log_dir, f"model_epoch{epoch}.pt")
                save_checkpoint(path, raw_model, optimizer, step, epoch,
                                tokens_seen=(step + 1) * cfg.total_batch_size, val_loss=val_loss)
                print(f"saved checkpoint for epoch {epoch} to {path}")

    if env.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    # vocab padded from 50257 to 50304 (a multiple of 128) for faster kernels
    main(TrainConfig(), GPTConfig(vocab_size=50304))
