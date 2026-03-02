########################################################################################################
# Bird's Nest — RWKV Engine
# Unified v6 "Finch" + v7 "Goose" inference on Apple MPS GPU
# Auto-detects model version from weight keys
########################################################################################################

import os, time, copy, types
from typing import Generator, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from birdsnest.engine import InferenceEngine

# ── Model architecture configs by n_embd ────────────────────────────────────
RWKV_CONFIGS = {
    768:   (12, "0.1B"),
    1024:  (24, "0.4B"), 
    1280:  (24, "0.5B"),
    2048:  (24, "1.5B"),
    2560:  (32, "2.9B"),
    4096:  (32, "7.2B"),
    5120:  (40, "13.3B"),
}

# ── Tokenizer ───────────────────────────────────────────────────────────────

class RWKVTokenizer:
    """RWKV World tokenizer (65536 vocab)."""
    
    def __init__(self, file_name: str):
        self.idx2token = {}
        sorted_tokens = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            sorted_tokens.append(x)
            self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}

        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for i in reversed(range(len(sorted_tokens))):
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0, s1 = int(s[0]), int(s[1])
                self.table[s0][s1].append(s)
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encode(self, src: str) -> list:
        src = src.encode("utf-8")
        tokens = []
        i = 0
        while i < len(src):
            s = src[i:i+1]
            if i < len(src) - 1:
                s0, s1 = int(src[i]), int(src[i+1])
                if s1 in self.good[s0]:
                    sss = src[i:i+self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decode(self, tokens: list) -> str:
        return b''.join(self.idx2token[t] for t in tokens).decode('utf-8', errors='replace')


# ── RWKV-7 RNN Functions ───────────────────────────────────────────────────

def _v7_time_mixing(layer_id, H, N, x, x_prev, v_first, state, 
                     x_r, x_w, x_k, x_v, x_a, x_g,
                     w0, w1, w2, a0, a1, a2, v0, v1, v2,
                     g1, g2, k_k, k_a, r_k,
                     kw, vw, rw, ow, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = rw @ xr
    w = torch.tanh(xw @ w1) @ w2
    k = kw @ xk
    v = vw @ xv
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = k * k_k
    kk = F.normalize(kk.view(H, N), dim=-1, p=2.0).view(-1)
    k = k * (1 + (a-1) * k_a)

    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = w0 + w.float()
    w = torch.exp(-0.606531 * torch.sigmoid(w))

    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk*a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(dtype=x.dtype) @ r.view(H, N, 1)

    out = F.group_norm(out.view(1, H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(H*N)
    out = out + ((r * k * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H*N)
    return ow @ (out * g), x, state, v_first


def _v7_channel_mixing(x, x_prev, x_k, kw, vw):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(kw @ k) ** 2
    return vw @ k, x


# ── RWKV-6 RNN Functions ───────────────────────────────────────────────────

def _v6_time_mixing(x, state, i, n_head, head_size,
                     time_maa_x, time_maa_w, time_maa_k, time_maa_v, time_maa_r, time_maa_g,
                     time_maa_w1, time_maa_w2,
                     time_decay_w1, time_decay_w2, time_faaaa, time_decay,
                     kw, vw, rw, gw, ow, ln_w, ln_b):
    H = n_head
    N = head_size
    
    sx = state[i*3+0] - x
    state[i*3+0] = x
    
    xxx = x + sx * time_maa_x
    xxx = torch.tanh(xxx @ time_maa_w1).view(5, 1, -1)
    xxx = torch.bmm(xxx, time_maa_w2).view(5, -1)
    mw, mk, mv, mr, mg = xxx.unbind(dim=0)
    
    xw = x + sx * (time_maa_w + mw)
    xk = x + sx * (time_maa_k + mk)
    xv = x + sx * (time_maa_v + mv)
    xr = x + sx * (time_maa_r + mr)
    xg = x + sx * (time_maa_g + mg)
    
    w = time_decay + (torch.tanh(xw @ time_decay_w1) @ time_decay_w2).float()
    w = torch.exp(-torch.exp(w.clamp(-100, 5)))
    
    r = rw @ xr
    k = kw @ xk
    v = vw @ xv
    g = F.silu(gw @ xg)
    
    kk = k.view(H, N)
    vv = v.view(H, N)
    rr = r.view(H, N)
    
    s = state[i*3+1].view(H, N, N).float()
    ww = w.view(H, N)
    
    # Per-head outer product: (H, N, 1) @ (H, 1, N) → (H, N, N)
    a = kk.float().unsqueeze(-1) @ vv.float().unsqueeze(-2)
    # Per-head query: (H, 1, N) @ (H, N, N) → (H, 1, N) → (H, N)
    out = (rr.float().unsqueeze(-2) @ s).squeeze(-2)
    s = s * ww.unsqueeze(-1).float() + a
    state[i*3+1] = s.view(H, N, N)
    
    out = out.view(1, H*N)
    out = F.group_norm(out, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(H*N)
    return ow @ (out.to(dtype=x.dtype) * g), state


def _v6_channel_mixing(x, state, i, time_maa_k, time_maa_r, kw, vw, rw):
    sx = state[i*3+2] - x
    state[i*3+2] = x
    xk = x + sx * time_maa_k
    xr = x + sx * time_maa_r
    r = torch.sigmoid(rw @ xr)
    k = torch.relu(kw @ xk) ** 2
    return r * (vw @ k), state


# ── RWKV Engine ─────────────────────────────────────────────────────────────

class RWKVEngine(InferenceEngine):
    """Unified RWKV v6/v7 engine with MPS GPU acceleration."""

    name = "rwkv"
    architecture = "rwkv"

    def __init__(self):
        super().__init__()
        self.tokenizer: Optional[RWKVTokenizer] = None
        self.weights = None
        self.version = None  # 6 or 7
        self.n_layer = 0
        self.n_embd = 0
        self.n_head = 0
        self.head_size = 64
        self.state = None
        self.init_state = None
        self.init_out = None
        self._find_tokenizer()

    def _find_tokenizer(self):
        """Locate tokenizer file relative to project root."""
        # engines/rwkv_engine.py -> engines/ -> birdsnest/ -> ChatRWKV/
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tok_path = os.path.join(base, "tokenizer", "rwkv_vocab_v20230424.txt")
        if os.path.exists(tok_path):
            self.tokenizer = RWKVTokenizer(tok_path)

    def _detect_version(self, keys: list) -> int:
        """Detect v6 vs v7 from weight key names."""
        for k in keys:
            if 'att.x_a' in k:  # v7 has x_a (in-context learning rate)
                return 7
            if 'time_maa_x' in k:  # v6 has time_maa_x
                return 6
        return 6  # fallback

    def load(self, model_path: str) -> Dict[str, Any]:
        """Load an RWKV model (.pth) with MPS GPU acceleration."""
        self.device = self.detect_device()
        dtype = torch.float32

        t0 = time.time()
        
        # Load weights to CPU first, then move to device
        z = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Detect version
        self.version = self._detect_version(list(z.keys()))
        
        # Get architecture params
        self.n_embd = z['blocks.0.ln1.weight'].shape[0]
        if self.n_embd in RWKV_CONFIGS:
            self.n_layer, size_name = RWKV_CONFIGS[self.n_embd]
        else:
            self.n_layer = 32
            size_name = f"~{self.n_embd // 256 * 256 / 1000:.1f}B?"

        if self.version == 7:
            self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        else:
            # v6: derive from time_faaaa
            self.n_head = z['blocks.0.att.time_faaaa'].shape[0]
            self.head_size = self.n_embd // self.n_head

        # Process weights based on version
        if self.version == 7:
            # Move all to device
            for k in z:
                z[k] = z[k].to(device=self.device)
            keys = list(z.keys())
            for k in keys:
                if k.endswith('att.w0'):
                    z[k] = z[k].float().to(device=self.device)
                else:
                    z[k] = z[k].to(dtype=dtype, device=self.device)
                z[k] = z[k].squeeze()
                if k.endswith('att.r_k'):
                    z[k] = z[k].flatten()

            # Fuse ln0 into embeddings
            z['emb.weight'] = F.layer_norm(
                z['emb.weight'], (self.n_embd,),
                weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias']
            )
            # v0/v1/v2 placeholders for layer 0
            z['blocks.0.att.v0'] = z['blocks.0.att.a0']
            z['blocks.0.att.v1'] = z['blocks.0.att.a1']
            z['blocks.0.att.v2'] = z['blocks.0.att.a2']
        else:
            # v6: convert and move
            for k in z:
                z[k] = z[k].float().to(device=self.device)
                if '.time_' in k:
                    z[k] = z[k].squeeze()
                if '.time_faaaa' in k:
                    z[k] = z[k].unsqueeze(-1)

        self.weights = z
        self.model_path = model_path
        self.model_name = os.path.basename(model_path).replace('.pth', '')
        self.is_loaded = True

        load_time = time.time() - t0
        self.model_info = {
            "version": f"v{self.version}",
            "size": size_name,
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "head_size": self.head_size,
            "device": self.device,
            "load_time": round(load_time, 1),
            "file_size_gb": round(os.path.getsize(model_path) / 1024**3, 1),
        }

        # Prime with system prompt
        self._init_state()

        return self.model_info

    def _init_state(self):
        """Initialize state and prime with system prompt."""
        dtype = torch.float32
        self.state = self._make_state(dtype)

        system_prompt = (
            "\nYou are a helpful, knowledgeable AI assistant. "
            "Answer questions clearly and concisely.\n\n"
            "User: Hello!\n\n"
            "Assistant: Hello! How can I help you today?\n\n"
        )

        out = None
        for token in self.tokenizer.encode(system_prompt):
            out = self._forward(token)

        self.init_out = out.clone() if out is not None else None
        self.init_state = copy.deepcopy(self.state)

    def _make_state(self, dtype=torch.float32) -> list:
        """Create fresh zero state."""
        state = [None] * (self.n_layer * 3)
        for i in range(self.n_layer):
            state[i*3+0] = torch.zeros(self.n_embd, dtype=dtype, device=self.device)
            state[i*3+1] = torch.zeros(
                (self.n_head, self.head_size, self.head_size),
                dtype=torch.float, device=self.device
            )
            state[i*3+2] = torch.zeros(self.n_embd, dtype=dtype, device=self.device)
        return state

    def _forward(self, token: int) -> torch.Tensor:
        """Single-token forward pass through the RNN."""
        z = self.weights
        
        if self.version == 7:
            return self._forward_v7(token)
        else:
            return self._forward_v6(token)

    def _forward_v7(self, token: int) -> torch.Tensor:
        """RWKV-7 Goose forward pass."""
        with torch.no_grad():
            z = self.weights
            x = z['emb.weight'][token]
            v_first = torch.empty_like(x)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, self.state[i*3+0], self.state[i*3+1], v_first = _v7_time_mixing(
                    i, self.n_head, self.head_size, xx, self.state[i*3+0], v_first, self.state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'],
                    z[att+'a0'], z[att+'a1'], z[att+'a2'],
                    z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'key.weight'], z[att+'value.weight'],
                    z[att+'receptance.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx, self.state[i*3+2] = _v7_channel_mixing(
                    xx, self.state[i*3+2], z[ffn+'x_k'],
                    z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            return z['head.weight'] @ x

    def _forward_v6(self, token: int) -> torch.Tensor:
        """RWKV-6 Finch forward pass."""
        with torch.no_grad():
            z = self.weights
            x = z['emb.weight'][token]
            x = F.layer_norm(x, (self.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, self.state = _v6_time_mixing(
                    xx, self.state, i, self.n_head, self.head_size,
                    z[att+'time_maa_x'], z[att+'time_maa_w'], z[att+'time_maa_k'],
                    z[att+'time_maa_v'], z[att+'time_maa_r'], z[att+'time_maa_g'],
                    z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],
                    z[att+'time_faaaa'], z[att+'time_decay'],
                    z[att+'key.weight'], z[att+'value.weight'],
                    z[att+'receptance.weight'], z[att+'gate.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx, self.state = _v6_channel_mixing(
                    xx, self.state, i,
                    z[ffn+'time_maa_k'], z[ffn+'time_maa_r'],
                    z[ffn+'key.weight'], z[ffn+'value.weight'], z[ffn+'receptance.weight'])
                x = x + xx

            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            return z['head.weight'] @ x

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        """Sample next token from logits."""
        probs = F.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)

        if top_p < 1:
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff_idx = torch.searchsorted(cumulative, top_p)
            cutoff = sorted_probs[cutoff_idx]
            probs[probs < cutoff] = 0

        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)

        probs = probs / probs.sum()
        return torch.multinomial(probs, num_samples=1).item()

    def generate_stream(self, prompt: str, temperature=1.0, top_p=0.7, max_tokens=500, system_prefix: str = "") -> Generator[str, None, None]:
        """Generate text token-by-token, yielding decoded pieces."""
        if not self.is_loaded:
            yield "[Error: No model loaded]"
            return

        # Feed system context into state BEFORE the User: tag (silent, no generation)
        if system_prefix:
            sys_tokens = self.tokenizer.encode(system_prefix)
            for t in sys_tokens:
                self._forward(t)

        # Detect G1 thinking model
        is_g1 = self.model_name and 'g1' in self.model_name.lower() and 'rwkv7' in self.model_name.lower()

        # Encode and feed the actual user prompt
        if is_g1:
            # G1 thinking format — seed with <think to trigger reasoning
            prompt_tokens = self.tokenizer.encode(f"User: {prompt}\n\nAssistant: <think\n")
            yield "<think>\n"  # Signal frontend that thinking mode is active
        else:
            prompt_tokens = self.tokenizer.encode(f"User: {prompt}\n\nAssistant:")
        out = None
        for t in prompt_tokens:
            out = self._forward(t)

        if out is None:
            yield "[Error: Forward pass failed]"
            return

        # Generate
        all_tokens = []
        import re as _re
        _stop_pattern = _re.compile(r'\n\s*[Uu]ser\s*[:\!\?]?')
        # Normalize prompt for echo detection
        _prompt_norm = prompt.lower().strip()
        
        for _ in range(max_tokens):
            token = self._sample(out, temperature, top_p)
            all_tokens.append(token)

            # Check stop BEFORE yielding
            decoded = self.tokenizer.decode(all_tokens)
            stop_match = _stop_pattern.search(decoded)
            if stop_match:
                # Yield only the text before the stop sequence
                remaining = decoded[:stop_match.start()]
                prev_decoded = self.tokenizer.decode(all_tokens[:-1]) if len(all_tokens) > 1 else ""
                new_text = remaining[len(prev_decoded):]
                if new_text and '\ufffd' not in new_text:
                    yield new_text
                break
            if decoded.endswith('\n\n') and len(all_tokens) > 10:
                break

            # Echo detection — if the model is repeating the user's prompt, stop
            if len(all_tokens) > 20:
                gen_norm = decoded.lower().strip()
                if len(gen_norm) > 20 and _prompt_norm[:30] in gen_norm:
                    break

            # Decode just this token and yield
            try:
                word = self.tokenizer.decode([token])
                if '\ufffd' not in word:
                    yield word
            except Exception:
                pass

            out = self._forward(token)

    def reset(self):
        """Reset conversation to initial state."""
        if self.init_state is not None:
            self.state = copy.deepcopy(self.init_state)

    def unload(self):
        """Free model weights and GPU memory."""
        self.weights = None
        self.state = None
        self.init_state = None
        self.init_out = None
        self.is_loaded = False
        self.model_name = None
        self.model_path = None
        self.model_info = {}
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)
