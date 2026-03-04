########################################################################################################
# RWKV-7 "Goose" Chat — Mac CPU/MPS Optimized
# Adapted from official rwkv_v7_demo_rnn.py by BlinkDL
# Supports all RWKV-7 x070 models (0.1B to 13.3B)
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time, os, sys
from typing import List
import torch.nn as nn
from torch.nn import functional as F

# ── Device Selection ────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
    print("⚡ Using Apple Metal GPU (MPS)")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    print("🐌 Using CPU fp32")

# ── Config ──────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
TOKENIZER_FILE = os.path.join(os.path.dirname(__file__), "tokenizer", "rwkv_vocab_v20230424.txt")

TEMPERATURE = 1.0
TOP_P = 0.7
MAX_TOKENS = 500

# ── Model Architecture Configs (auto-detected from weights) ────────────────
MODEL_CONFIGS = {
    # n_embd -> (n_layer, name)
    768:   (12, "0.1B"),
    1024:  (24, "0.4B"),
    2048:  (24, "1.5B"),
    2560:  (32, "2.9B"),
    4096:  (32, "7.2B"),
}

# ── Available Models ────────────────────────────────────────────────────────
def list_models():
    models = []
    if os.path.exists(MODELS_DIR):
        for f in sorted(os.listdir(MODELS_DIR)):
            if f.endswith('.pth'):
                path = os.path.join(MODELS_DIR, f)
                size_gb = os.path.getsize(path) / 1024**3
                name = f.replace('.pth', '')
                models.append((name, path, size_gb))
    return models

# ── Tokenizer ───────────────────────────────────────────────────────────────

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted_tokens += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted_tokens))):
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]
            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens)).decode('utf-8', errors='replace')

# ── Sampling ────────────────────────────────────────────────────────────────

def sample_logits(logits, temperature=1.0, top_p=0.7):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0
        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()

# ── RWKV-7 Core Functions ──────────────────────────────────────────────────

def time_mixing(layer_id, H, N, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, kw, vw, rw, ow, ln_w, ln_b):
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

    # Fused decay calculation
    w = w0 + w.float()
    w = torch.exp(-0.606531 * torch.sigmoid(w))  # 0.606531 = exp(-0.5)
    
    # RWKV-7 kernel (RNN mode)
    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk*a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(dtype=x.dtype) @ r.view(H, N, 1)

    out = F.group_norm(out.view(1, H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(H*N)    
    out = out + ((r * k * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H*N)
    return ow @ (out * g), x, state, v_first

def channel_mixing(x, x_prev, x_k, kw, vw):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(kw @ k) ** 2
    return vw @ k, x

# ── RWKV-7 RNN Model ───────────────────────────────────────────────────────

class RWKV_RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        # Move all weights to GPU
        for k in self.z:
            self.z[k] = self.z[k].to(device=DEVICE)
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape

        keys = list(z.keys())
        for k in keys:
            if k.endswith('att.w0'):
                z[k] = z[k].float().to(device=DEVICE)
            else:
                z[k] = z[k].to(dtype=DTYPE, device=DEVICE)
            z[k] = z[k].squeeze()
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        # Fuse embedding with ln0 for efficiency
        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        # v0/v1/v2 are ignored for layer 0 — set to a0/a1/a2 as placeholders
        z['blocks.0.att.v0'] = z['blocks.0.att.a0']
        z['blocks.0.att.v1'] = z['blocks.0.att.a1']
        z['blocks.0.att.v2'] = z['blocks.0.att.a2']

    def forward(self, token: int, state: List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][token]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = time_mixing(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'key.weight'], z[att+'value.weight'], z[att+'receptance.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = channel_mixing(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = z['head.weight'] @ x

            return x, state

# ── Main Chat Loop ──────────────────────────────────────────────────────────

def main():
    print("""
╔════════════════════════════════════════════════╗
║   🐦 RWKV-7 "Goose" Chat — Mac Edition        ║
║   Non-Transformer AI on Apple Silicon          ║
╚════════════════════════════════════════════════╝
    """)

    if not os.path.exists(TOKENIZER_FILE):
        print(f"❌ Tokenizer not found: {TOKENIZER_FILE}")
        sys.exit(1)

    models = list_models()
    if not models:
        print(f"❌ No .pth models found in {MODELS_DIR}")
        sys.exit(1)

    # Filter to show only v7 (x070) models, but allow v6 too
    print("📦 Available Models:")
    print("─" * 60)
    for i, (name, path, size_gb) in enumerate(models):
        tag = "v7" if "x070" in name or "g1" in name else "v6" if "x060" in name else "?"
        print(f"  [{i+1}] [{tag}] {name}  ({size_gb:.1f} GB)")
    print("─" * 60)

    if len(models) == 1:
        choice = 0
        print(f"\n  Auto-selecting: {models[0][0]}")
    else:
        try:
            raw = input(f"\nSelect model [1-{len(models)}]: ").strip()
            choice = int(raw) - 1
            if choice < 0 or choice >= len(models):
                print("Invalid choice"); sys.exit(1)
        except (ValueError, KeyboardInterrupt):
            print("Invalid choice"); sys.exit(1)

    model_name, model_path, model_size = models[choice]
    
    # Check if this is a v7 model
    if 'x060' in model_name:
        print(f"\n⚠️  {model_name} is a v6 model. Use rwkv_chat_fast.py for v6 models.")
        print(f"   This script is optimized for v7 (x070) models.")
        sys.exit(1)

    print(f"\n📖 Loading tokenizer...")
    tokenizer = RWKV_TOKENIZER(TOKENIZER_FILE)

    # Auto-detect architecture from weights
    print(f"🧠 Loading {model_name} ({model_size:.1f} GB)...")
    t0 = time.time()
    
    # Peek at weights to get n_embd
    peek = torch.load(model_path, map_location='cpu', weights_only=False)
    n_embd = peek['blocks.0.ln1.weight'].shape[0]
    n_head, head_size = peek['blocks.0.att.r_k'].shape
    del peek
    
    if n_embd in MODEL_CONFIGS:
        n_layer, size_name = MODEL_CONFIGS[n_embd]
    else:
        print(f"⚠️  Unknown model size (n_embd={n_embd}), guessing n_layer=32")
        n_layer = 32
        size_name = "unknown"

    args = types.SimpleNamespace()
    args.MODEL_NAME = model_path.replace('.pth', '')
    args.n_layer = n_layer
    args.n_embd = n_embd
    args.vocab_size = 65536
    args.head_size = head_size

    print(f"   Architecture: {size_name} — {n_layer}L / {n_embd}D / {n_head}H")
    
    model = RWKV_RNN(args)
    print(f"✅ Model loaded in {time.time()-t0:.1f}s ({n_head} heads × {head_size} dim)")

    # Initialize state
    def make_state():
        state = [None for _ in range(n_layer * 3)]
        for i in range(n_layer):
            state[i*3+0] = torch.zeros(n_embd, dtype=DTYPE, device=DEVICE)
            state[i*3+1] = torch.zeros((n_embd // head_size, head_size, head_size), dtype=torch.float, device=DEVICE)
            state[i*3+2] = torch.zeros(n_embd, dtype=DTYPE, device=DEVICE)
        return state

    # System prompt
    system_prompt = "\nYou are a helpful, knowledgeable AI assistant. Answer questions clearly and concisely.\n\nUser: Hello!\n\nAssistant: Hello! I'm Raven, an AI assistant powered by RWKV-7, a non-transformer architecture. How can I help you today?\n\n"
    
    print("⚡ Warming up with system prompt...")
    state = make_state()
    out = None
    for token in tokenizer.encode(system_prompt):
        out, state = model.forward(token, state)
    
    # Save initial state for reset
    init_out = out.clone()
    init_state = copy.deepcopy(state)
    print("✅ Ready!\n")

    print("─" * 60)
    print(f"  Model: {model_name} ({size_name})")
    print(f"  Heads: {n_head} × {head_size}d | Layers: {n_layer}")
    print(f"  Commands: quit | reset")
    print("─" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        if user_input.lower() == 'reset':
            print("🔄 Resetting conversation...")
            out = init_out.clone()
            state = copy.deepcopy(init_state)
            print("✅ Memory cleared!\n")
            continue

        # Encode user message
        prompt_text = f"User: {user_input}\n\nAssistant:"
        for token in tokenizer.encode(prompt_text):
            out, state = model.forward(token, state)

        # Generate response
        print("\nRaven: ", end="", flush=True)
        t_start = time.time()
        all_tokens = []
        generated = 0
        
        for i in range(MAX_TOKENS):
            token = sample_logits(out, TEMPERATURE, TOP_P)
            all_tokens.append(token)
            generated += 1
            
            try:
                word = tokenizer.decode(all_tokens[-1:])
                if '\ufffd' not in word:
                    print(word, end="", flush=True)
            except:
                pass
            
            out, state = model.forward(token, state)
            
            decoded_so_far = tokenizer.decode(all_tokens)
            if '\n\nUser:' in decoded_so_far or '\n\nuser:' in decoded_so_far:
                break
            if decoded_so_far.endswith('\n\n') and len(all_tokens) > 10:
                break

        elapsed = time.time() - t_start
        tok_s = generated / elapsed if elapsed > 0 else 0
        print(f"\n  [{generated} tokens, {tok_s:.1f} tok/s]\n")

if __name__ == "__main__":
    main()
