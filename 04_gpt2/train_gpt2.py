from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy
import tiktoken
import time
import inspect
import os
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from hellaswag import render_example, iterate_examples

# datasets live in <repo>/resources; the log and checkpoints go next to this script
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "edu_fineweb10B"))
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")


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
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


EOT_TOKEN = 50256  # <|endoftext|>, the first token of every document


def load_tokens(filename, shuffle_rng=None):
    npt = np.load(filename)
    if shuffle_rng is not None:
        npt = shuffle_documents(npt, shuffle_rng)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


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
    """Streams (B, T) batches from token shards; B is passed on each call.
    current_position is the same on all ranks; each rank reads its own B*T slice after it.
    self.epoch counts completed passes over all shards.
    With shuffle=True, every epoch (including the first) uses a new random shard order and
    a random document order within each shard; each shard's leading partial document is dropped. Both are seeded by (seed, epoch[, shard]), so
    every rank, and every copy of the loader, sees exactly the same data order."""
    def __init__(self, T, process_rank, num_processes, split, shuffle=False, seed=1337):
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.shuffle = shuffle
        self.seed = seed
        assert split in ("train", "val")

        data_root = DATA_ROOT
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.shard_lens = [len(np.load(s, mmap_mode="r")) for s in shards]  # header only, cheap
        if shuffle:  # shuffling drops each shard's leading fragment (see shuffle_documents)
            self.shard_lens = [L - leading_fragment_len(s) for L, s in zip(self.shard_lens, shards)]
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.epoch = 0
        self._start_epoch()

    def _start_epoch(self):
        if self.shuffle:
            self.shard_order = np.random.default_rng((self.seed, self.epoch)).permutation(len(self.shards))
        else:
            self.shard_order = np.arange(len(self.shards))
        self.current_shard = 0  # index into shard_order
        self.current_position = 0
        self.tokens = None  # loaded lazily, so dry runs never touch token data

    def _shard_len(self):
        return self.shard_lens[self.shard_order[self.current_shard]]

    def _load_current_shard(self):
        shard_idx = self.shard_order[self.current_shard]
        rng = None
        if self.shuffle:
            rng = np.random.default_rng((self.seed, self.epoch, int(shard_idx)))
        return load_tokens(self.shards[shard_idx], shuffle_rng=rng)

    def _fits(self, B):
        # the last rank reads up to current_position + B*T*num_processes + 1
        return self.current_position + B * self.T * self.num_processes + 1 <= self._shard_len()

    def _next_shard(self):
        if self.current_shard + 1 == len(self.shards):
            self.epoch += 1
            self._start_epoch()  # new shard order for the next epoch
        else:
            self.current_shard += 1
            self.current_position = 0
            self.tokens = None

    def next_batch(self, B, dry_run=False):
        """Returns (x, y) of shape (B, T). With dry_run=True only advances the position (returns None)."""
        T = self.T
        if not self._fits(B):  # B may have grown since the last call
            self._next_shard()
        out = None
        if not dry_run:
            if self.tokens is None:
                self.tokens = self._load_current_shard()
            start = self.current_position + B * T * self.process_rank
            buf = self.tokens[start : start + B * T + 1]
            out = (buf[:-1]).view(B, T), (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if not self._fits(B):  # switch right away so an epoch ends on its last batch
            self._next_shard()
        return out


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
        
    

    # ---- batch size: fixed 0.5M tokens per step, no ramp (as in Karpathy's build-nanogpt) ----
    total_batch_size = 524288       # 2**19, ~0.5M tokens per step
    B_max = 32                      # micro-batch per GPU (Karpathy used 64 on 80GB A100s; 32 fits 40GB)
    T = 1024
    assert total_batch_size % (B_max * T * ddp_world_size) == 0
    grad_accum = total_batch_size // (B_max * T * ddp_world_size)
    if master_process:
        print(f"total batch size: {total_batch_size}, grad accumulation steps: {grad_accum}")

    train_loader = DataLoaderLite(T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", shuffle=True)
    val_loader = DataLoaderLite(T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # ---- precompute the whole run with a dry run of the data loader ----
    # Deterministic and identical on every rank: the exact number of steps in num_epochs passes over
    # the train shards (each shard's tail is skipped) and the steps on which each epoch ends.
    num_epochs = 5
    plan = []             # per step: (B, grad_accum_steps, tokens_seen_before_step)
    epoch_end_steps = {}  # step -> number of the epoch that ends on that step
    sim = copy.copy(train_loader)
    tokens_seen = 0
    while sim.epoch < num_epochs:
        B, accum = B_max, grad_accum
        for _ in range(accum):
            sim.next_batch(B, dry_run=True)
        plan.append((B, accum, tokens_seen))
        tokens_seen += B * T * accum * ddp_world_size
        if sim.epoch > len(epoch_end_steps):
            epoch_end_steps[len(plan) - 1] = sim.epoch
    max_steps = len(plan)
    total_tokens = tokens_seen

    torch.set_float32_matmul_precision("high")
    
    # model = GPT.from_pretrained("gpt2")
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    use_compile = True
    # raw_model: the plain eager module (no compile, no DDP). Shares parameters with `model`.
    # Used for the optimizer, checkpoints, and evals with variable sequence lengths
    # (HellaSwag, generation) which would otherwise trigger repeated recompiles.
    raw_model = model
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # ---- learning rate schedule (Karpathy's build-nanogpt) ----
    # linear warmup over 715 steps (~375M tokens, the GPT-3 warmup), then cosine decay to 10% of
    # max_lr over the rest of the run, so the decay always completes at the last step
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    if master_process:
        print(f"epochs: {num_epochs}, total tokens: {total_tokens:,}, steps: {max_steps}")
        print(f"lr: warmup {warmup_steps} steps to {max_lr:.1e}, cosine to {min_lr:.1e} at step {max_steps}")
        print(f"epoch ends (checkpoints) at steps: {sorted(epoch_end_steps)}")

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass
    last_val_loss = None

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch(B_max)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            last_val_loss = val_loss_accum.item()
            if master_process:
                print(f"validation loss: {last_val_loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {last_val_loss:.4f}\n")

        if step % 250 == 0 or last_step:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = raw_model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")


        if (step > 0 and step % 250 == 0) or last_step:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = raw_model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")


        B, grad_accum_steps, tokens_seen = plan[step]
        step_tokens = B * T * grad_accum_steps * ddp_world_size
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(B)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)
        tokens_per_sec = step_tokens / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | batch: {step_tokens:,} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/s: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

        # checkpoint at the end of every epoch
        if step in epoch_end_steps and master_process:
            epoch = epoch_end_steps[step]
            assert train_loader.epoch == epoch, "data loader out of sync with the precomputed plan"
            checkpoint_path = os.path.join(log_dir, f"model_epoch{epoch}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': raw_model.config,
                'step': step,
                'epoch': epoch,
                'tokens_seen': tokens_seen + step_tokens,
                'val_loss': last_val_loss,  # from the most recent eval (every 250 steps)
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"saved checkpoint for epoch {epoch} to {checkpoint_path}")


    if ddp:
        destroy_process_group()
