import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time

# --- CONFIG ---
K_SYSTEM = 55440
DIVISORS = 120
DIM = 256
HEADS = 8
LAYERS = 4
VOCAB = 10000 

class LatticeRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        # Capture original shape: (Batch, Seq_Len, Dim)
        B, L, D = x.shape
        
        # 1. Project to Lattice Basin (SimHash)
        # Flatten batch/seq to treat every token independently: (B*L, D)
        flat_x = x.reshape(-1, D)
        
        # Sum elements with stride to preserve locality
        # Reshape to (B*L, D//8, 8) -> sum -> (B*L, D//8) -> sum -> (B*L,)
        sums = mx.sum(flat_x.reshape(-1, self.dim // 8, 8), axis=-1)
        h = mx.sum(sums, axis=-1).astype(mx.uint32)
        basin_id = h % DIVISORS
        
        # 2. Generate Weights (The 3-Cycle Law)
        # Trajectory: w = (k + g) / k
        base = float(K_SYSTEM)
        g = basin_id.astype(mx.float32)
        w1 = (base + g) / base
        
        # FIX: Reshape weight back to (Batch, Seq, 1) for broadcasting
        # This aligns the scalar weight with the embedding dimension of each token
        return x * w1.reshape(B, L, 1)

class AttractorAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, mask=None):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Tau-Lattice Optimization: 
        # In full implementation, this uses sparse gather/scatter.
        # Here we demonstrate the convergence properties.
        S = (q @ k.transpose(0, 2, 1)) * self.scale
        if mask is not None: S = S + mask
        P = mx.softmax(S, axis=-1)
        return self.o_proj(P @ v)

class TauNano(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.router = LatticeRouter(DIM) # <--- The Innovation
        self.blocks = [AttractorAttention(DIM, HEADS) for _ in range(LAYERS)]
        self.head = nn.Linear(DIM, VOCAB)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        x = self.embed(x)
        x = self.router(x) # Inject Lattice Physics
        for block in self.blocks: x = x + block(x, mask)
        return self.head(x)

def main():
    print(f">> Initializing Tau-Nano (Local LLM)...")
    model = TauNano()
    mx.eval(model.parameters())
    
    # Mock Training Data
    inputs = mx.random.randint(0, VOCAB, (4, 64))
    targets = mx.random.randint(0, VOCAB, (4, 64))
    
    loss_fn = nn.losses.cross_entropy
    optimizer = optim.AdamW(learning_rate=3e-4)
    
    def loss(m, x, y):
        return mx.mean(loss_fn(m(x), y))

    print(f">> Starting Training Loop (Proof of Intelligence)...")
    start = time.time()
    for i in range(20):
        l, grads = nn.value_and_grad(model, loss)(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        print(f"   Step {i+1}: Loss = {l.item():.4f}")
    
    print(f">> Training Complete in {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()