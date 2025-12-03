import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Low-rank attention module that projects Q,K into a low-dim basis B (d x r)
class LowRankSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, r, learn_basis=True, share_basis=True, dropout=0.1):
        """
        d_model: input embedding dim
        nhead: number of heads
        r: reduced dimension (per head will be r_head = r // nhead if r divisible)
        learn_basis: if True, basis B is a trainable parameter; else optionally initialize via PCA externally
        share_basis: share same B across heads (if False, have head-specific B)
        """
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.r = r
        # We will project per-head into r_head dims
        # Choose r_head = max(1, r // nhead) to keep things simple
        self.r_head = max(1, r // nhead)

        # standard linear projections for Q,K,V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.learn_basis = learn_basis
        self.share_basis = share_basis

        # Basis(es): shape (d_model, r_head), either shared or per-head
        if learn_basis:
            if share_basis:
                self.B = nn.Parameter(torch.randn(d_model, self.r_head))
            else:
                # shape (nhead, d_model, r_head)
                self.B = nn.Parameter(torch.randn(nhead, d_model, self.r_head))
            # orthonormalize initial basis
            self._init_orthonormal(self.B)
        else:
            # placeholder; expects caller to set self.B via set_basis()
            self.B = None

    def _init_orthonormal(self, B):
        with torch.no_grad():
            if B is None:
                return
            if self.share_basis:
                # QR
                Q, _ = torch.qr(B)
                B.copy_(Q[:, :self.r_head])
            else:
                # each head
                for h in range(self.nhead):
                    Q, _ = torch.qr(B[h])
                    B.data[h].copy_(Q[:, :self.r_head])

    def set_basis(self, B_tensor):
        """
        If learn_basis=False, you can set a precomputed basis tensor here.
        Expected shapes:
          - shared: (d_model, r_head)
          - per-head: (nhead, d_model, r_head)
        """
        assert not self.learn_basis, "set_basis should only be used when learn_basis=False"
        if self.share_basis:
            assert B_tensor.shape == (self.d_model, self.r_head)
        else:
            assert B_tensor.shape == (self.nhead, self.d_model, self.r_head)
        self.B = B_tensor.to(next(self.parameters()).device)

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, d_model]
        mask: [batch, seq_len] or [batch, seq_len, seq_len]
        returns: [batch, seq_len, d_model]
        """
        bsz, seq_len, _ = x.shape

        Q = self.W_q(x)  # [b, s, d]
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape for multi-head: [b, s, nhead, head_dim] -> [b, nhead, s, head_dim]
        Q = Q.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Project Q and K into reduced r_head dimension using basis B
        # B shape possibilities:
        #  - shared: (d_model, r_head) -> use same for every head but need to slice head-specific dims
        # Approach: create head-specific basis of shape (nhead, head_dim, r_head)
        if self.learn_basis:
            if self.share_basis:
                # Shared B is in d_model space; we slice to head dims by selecting corresponding weight rows
                # But easier: create head basis by linear mapping: extract head-specific slice indices
                # We'll reshape B to [d_model, r_head] then slice per head by isolating portion corresponding to head_dim
                # For simplicity, we create head-specific basis by slicing with head offsets
                B_shared = self.B  # [d_model, r_head]
                head_bases = []
                for h in range(self.nhead):
                    start = h * self.head_dim
                    end = (h + 1) * self.head_dim
                    head_bases.append(B_shared[start:end, :])  # [head_dim, r_head]
                B_heads = torch.stack(head_bases, dim=0).to(x.device)  # [nhead, head_dim, r_head]
            else:
                # B is already per-head: [nhead, d_model, r_head] -> slice head_dim
                B_heads = []
                for h in range(self.nhead):
                    start = h * self.head_dim
                    end = (h + 1) * self.head_dim
                    B_heads.append(self.B[h, start:end, :])  # [head_dim, r_head]
                B_heads = torch.stack(B_heads, dim=0).to(x.device)
        else:
            # user-provided B can be in two shapes
            if self.B is None:
                raise RuntimeError("No basis set — call set_basis() before forward when learn_basis=False")
            if self.share_basis:
                B_shared = self.B.to(x.device)
                head_bases = []
                for h in range(self.nhead):
                    start = h * self.head_dim
                    end = (h + 1) * self.head_dim
                    head_bases.append(B_shared[start:end, :])
                B_heads = torch.stack(head_bases, dim=0)
            else:
                # B is [nhead, d_model, r_head] -> slice head dims
                head_bases = []
                for h in range(self.nhead):
                    start = h * self.head_dim
                    end = (h + 1) * self.head_dim
                    head_bases.append(self.B[h, start:end, :].to(x.device))
                B_heads = torch.stack(head_bases, dim=0)

        # Project: for each head, Q_head: [b, s, head_dim], B_head: [head_dim, r_head]
        # result Qr: [b, nhead, s, r_head]
        Qr = torch.einsum('bnhd,hrc->bnhrc', Q, B_heads)  # temporarily using extra dim
        Kr = torch.einsum('bnhd,hrc->bnhrc', K, B_heads)

        # above einsum yields shape [b, nhead, h, r_head, ???] — too messy. Simpler: loop heads
        # Simpler (and clearer): compute with for loop across heads (nhead small)
        Qr_list = []
        Kr_list = []
        for h in range(self.nhead):
            Bh = B_heads[h]  # [head_dim, r_head]
            Qh = Q[:, h, :, :]  # [b, s, head_dim]
            Kh = K[:, h, :, :]
            Qr_list.append(Qh @ Bh)  # [b, s, r_head]
            Kr_list.append(Kh @ Bh)
        # stack: [b, nhead, s, r_head]
        Qr = torch.stack(Qr_list, dim=1)
        Kr = torch.stack(Kr_list, dim=1)

        # compute attention in reduced space
        # [b, nhead, s, r]
        attn_scores = torch.matmul(Qr, Kr.transpose(-1, -2)) / math.sqrt(self.r_head)
        if mask is not None:
            # mask expected [b, s] or [b, 1, s, s] etc.
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [b,1,1,s]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # apply attention weights to full-dim V (not reduced) -> achieve cross-dim mixing
        # attn: [b, nhead, s, s], V: [b, nhead, s, head_dim] -> out_head = attn @ V -> [b,nhead,s,head_dim]
        out_heads = torch.matmul(attn, V)  # [b, nhead, s, head_dim]
        out = out_heads.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(out)
        return out

# Small Transformer block using LowRankSelfAttention
class LowRankTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, r, dim_feedforward=2048, dropout=0.1, learn_basis=True):
        super().__init__()
        self.self_attn = LowRankSelfAttention(d_model, nhead, r, learn_basis=learn_basis)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.gelu

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, mask=src_mask)
        src = src + src2
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)
        return src

class LowRankTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, r=64, dim_feedforward=512, max_len=512, learn_basis=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            LowRankTransformerLayer(d_model, nhead, r, dim_feedforward, learn_basis=learn_basis)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [b, s] token ids
        b, s = x.shape
        positions = torch.arange(0, s, device=x.device).unsqueeze(0).expand(b, s)
        h = self.tok_emb(x) + self.pos_emb(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.ln(h)
        logits = self.head(h)
        return logits
