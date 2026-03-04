########################################################################################################
# RWKV-6 Interactive Chat — Mac/CPU Edition
# Adapted from RWKV_v6_demo.py by BlinkDL
# No CUDA needed — runs on CPU (Apple Silicon optimized via PyTorch MPS)
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, time, os, sys
import torch.nn as nn
from torch.nn import functional as F

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_NAME = os.path.join(MODEL_DIR, "RWKV-x060-World-1B6-v2.1-20240328-ctx4096")
TOKENIZER_FILE = os.path.join(os.path.dirname(__file__), "tokenizer", "rwkv_vocab_v20230424.txt")

TEMPERATURE = 1.0
TOP_P = 0.7
MAX_TOKENS = 300

# Model architecture for 1.6B
N_LAYER = 24
N_EMBD = 2048
VOCAB_SIZE = 65536

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

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

# ── RWKV-6 RNN Model ───────────────────────────────────────────────────────

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval()
        
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')

        for k in w.keys():
            w[k] = w[k].float()
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head
        
        self.w = types.SimpleNamespace()
        self.w.blocks = {}
        for k in w.keys():
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
    def channel_mixing(self, x, state, i:int, time_maa_k, time_maa_r, kw, vw, rw):
        i0 = (2+self.head_size)*i+0
        sx = state[i0] - x
        xk = x + sx * time_maa_k
        xr = x + sx * time_maa_r
        state[i0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        return r * (vw @ k)

    @MyFunction
    def time_mixing(self, x, state, i:int, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
        H = self.n_head
        S = self.head_size

        i1 = (2+S)*i+1
        sx = state[i1] - x
        state[i1] = x
        xxx = x + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + sx * (w_maa + mw)
        xk = x + sx * (k_maa + mk)
        xv = x + sx * (v_maa + mv)
        xr = x + sx * (r_maa + mr)
        xg = x + sx * (g_maa + mg)

        w = (time_decay + (torch.tanh(xw @ td_w1) @ td_w2).float()).view(H, S, 1)
        w = torch.exp(-torch.exp(w.float()))

        r = (rw @ xr).view(H, 1, S)
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)
        g = F.silu(gw @ xg)

        s = state[(2+S)*i+2:(2+S)*(i+1), :].reshape(H, S, S)

        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (time_first * a + s)
        s = a + w * s
    
        state[(2+S)*i+2:(2+S)*(i+1), :] = s.reshape(S, -1)
        x = x.flatten()

        x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).squeeze(0) * g
        return ow @ x

    def forward(self, token, state):
        with torch.no_grad():
            if state == None:
                state = torch.zeros(self.args.n_layer * (2+self.head_size), self.args.n_embd)
            
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i,
                    att.time_maa_x, att.time_maa_w, att.time_maa_k, att.time_maa_v, att.time_maa_r, att.time_maa_g, att.time_maa_w1, att.time_maa_w2,
                    att.time_decay_w1, att.time_decay_w2, att.time_faaaa, att.time_decay,
                    att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                    att.ln_x.weight, att.ln_x.bias)
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_maa_k, ffn.time_maa_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
            
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

# ── Main Chat Loop ──────────────────────────────────────────────────────────

def main():
    print("""
╔═══════════════════════════════════════════╗
║     🐦 RWKV-6 Chat — Raven Engine        ║
║     Non-Transformer AI on Your Mac        ║
╚═══════════════════════════════════════════╝
    """)

    # Check for tokenizer
    if not os.path.exists(TOKENIZER_FILE):
        print(f"❌ Tokenizer not found: {TOKENIZER_FILE}")
        sys.exit(1)

    # Check for model
    model_path = MODEL_NAME + '.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"   Download from: https://huggingface.co/BlinkDL/rwkv-6-world")
        sys.exit(1)

    print("📖 Loading tokenizer...")
    tokenizer = RWKV_TOKENIZER(TOKENIZER_FILE)

    print(f"🧠 Loading RWKV-6 1.6B model (this takes ~30s on first load)...")
    t0 = time.time()
    args = types.SimpleNamespace()
    args.MODEL_NAME = MODEL_NAME
    args.n_layer = N_LAYER
    args.n_embd = N_EMBD
    args.vocab_size = VOCAB_SIZE
    model = RWKV_RNN(args)
    print(f"✅ Model loaded in {time.time()-t0:.1f}s")

    # Initialize with a system prompt
    system_prompt = "\nYou are a helpful, knowledgeable AI assistant. Answer questions clearly and concisely.\n\nUser: Hello!\n\nAssistant: Hello! I'm an AI assistant powered by RWKV, a non-transformer architecture. How can I help you today?\n\n"
    
    print("⚡ Warming up with system prompt...")
    state = None
    out = None
    for token in tokenizer.encode(system_prompt):
        out, state = model.forward(token, state)
    print("✅ Ready!\n")

    print("─" * 50)
    print("Type your message and press Enter.")
    print("Type 'quit' to exit, 'reset' to clear memory.")
    print("─" * 50)
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
            state = None
            for token in tokenizer.encode(system_prompt):
                out, state = model.forward(token, state)
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
            
            # Stream output
            try:
                word = tokenizer.decode(all_tokens[-1:])
                if '\ufffd' not in word:
                    print(word, end="", flush=True)
            except:
                pass
            
            out, state = model.forward(token, state)
            
            # Stop conditions
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
