"""
Multi-rater segmentation metrics (MBH-Seg set + personalized) with robust shape handling.
Works with AESPA validation_sam and SPA variants.

Public API:
    compute_metrics(
        preds_single: (B,1,H,W) logits/probs/binary,
        raters:       (B,R,1,H,W) or (B,R,H,W),
        valid_raters: (B,R) bool/0-1, optional,
        pred_set:     (B,M,1,H,W) or (B,M,H,W), optional,
        follow_idx:   (B,), optional,
        dice_match_norm: "gt"|"max"|"match" (default: "gt"),
        dice_max_symmetric: bool (default: False),
        run: int, optional -> included in result dict if provided
    ) -> Dict[str, Tensor or float]

Returns per-batch-mean metrics as floats (Python float). If you need per-sample vectors,
adapt code to skip .mean().
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

# Optional Hungarian (SciPy); fallback to greedy if not installed.
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -------------------------- helpers (robust shapes) --------------------------

def _to_probs(x: torch.Tensor) -> torch.Tensor:
    """Convert logits/binary/probs to probabilities in [0,1] if needed."""
    if x.numel() == 0:
        return x
    if x.min() >= 0.0 and x.max() <= 1.0:
        return x
    return torch.sigmoid(x)

def _ensure_BM1HW(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure a set tensor has shape (B,M,1,H,W).
    Accepts (B,M,1,H,W) or (B,M,H,W) or (B,1,H,W) (-> M=1).
    """
    if x.dim() == 5:
        B, M, C, H, W = x.shape
        if C != 1:
            # if someone passed multi-channel preds, reduce by sigmoid->>0.5 then pick foreground
            # but in practice C should be 1 for a binary mask head.
            x = x[:, :, :1]  # keep first channel
        return x
    elif x.dim() == 4:
        # treat as (B,M=1,H,W)
        B, H, W, *rest = x.shape[0], x.shape[1], x.shape[2], []
        x = x.unsqueeze(1)  # (B,1,H,W)
        return x.unsqueeze(2)  # (B,1,1,H,W)
    else:
        raise ValueError(f"{name}: expected (B,M,1,H,W) or (B,M,H,W) or (B,1,H,W); got {tuple(x.shape)}")

def _ensure_BR1HW(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure raters tensor has shape (B,R,1,H,W).
    Accepts (B,R,1,H,W) or (B,R,H,W).
    """
    if x.dim() == 5:
        B, R, C, H, W = x.shape
        if C != 1:
            x = x[:, :, :1]
        return x
    elif x.dim() == 4:
        return x.unsqueeze(2)  # (B,R,1,H,W)
    else:
        raise ValueError(f"{name}: expected (B,R,1,H,W) or (B,R,H,W); got {tuple(x.shape)}")

def _flatten_set(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten spatial dims of a set: (M,1,H,W) -> (M,N)
    """
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"_flatten_set expects (M,1,H,W); got {tuple(x.shape)}")
    M, _, H, W = x.shape
    return x.view(M, 1, H * W)[:, 0, :]

def _pairwise_dice(P: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Dice between two sets of binary masks.
    P: (M,N) binary {0,1}; G: (R,N) binary {0,1} -> returns (M,R)
    """
    M, N = P.shape
    R, N2 = G.shape
    assert N == N2, "Prediction and GT must have same flattened size."

    P_exp = P.unsqueeze(1)   # (M,1,N)
    G_exp = G.unsqueeze(0)   # (1,R,N)
    inter = (P_exp * G_exp).sum(dim=2)  # (M,R)
    p_sum = P_exp.sum(dim=2)            # (M,1)
    g_sum = G_exp.sum(dim=2)            # (1,R)
    dice = (2.0 * inter) / (p_sum + g_sum + 1e-7)

    # empty-empty -> 1.0
    empty_pair = (p_sum == 0) * (g_sum == 0)  # (M,R) broadcast
    dice = torch.where(empty_pair, torch.ones_like(dice), dice)
    return dice


# -------------------------- set-level metrics --------------------------

def dice_max(pred_set: torch.Tensor, raters: torch.Tensor, symmetric: bool = False) -> torch.Tensor:
    """
    Dice_max: for each GT rater, take max Dice across prediction set.
    Optionally symmetric (average both directions).
    Returns per-batch mean (B,) tensor.
    """
    P = _ensure_BM1HW(pred_set, "dice_max/pred_set")
    G = _ensure_BR1HW(raters,   "dice_max/raters")
    B = P.size(0)

    out = []
    for b in range(B):
        P_b = _to_probs(P[b])          # (M,1,H,W)
        G_b = G[b].float()             # (R,1,H,W)
        P_bin = (P_b > 0.5).float()
        G_bin = (G_b > 0.5).float()
        P_flat = _flatten_set(P_bin)   # (M,N)
        G_flat = _flatten_set(G_bin)   # (R,N)
        D = _pairwise_dice(P_flat, G_flat)   # (M,R)

        cov_gt = D.max(dim=0).values.mean()
        if symmetric:
            cov_pred = D.max(dim=1).values.mean()
            out.append(0.5 * (cov_gt + cov_pred))
        else:
            out.append(cov_gt)

    return torch.stack(out).float()  # (B,)


def dice_match(pred_set: torch.Tensor, raters: torch.Tensor, normalize_by: str = "gt") -> torch.Tensor:
    """
    Dice_match via Hungarian (maximize sum of Dice).
    normalize_by: "gt" | "max" | "match"
    Returns (B,)
    """
    P = _ensure_BM1HW(pred_set, "dice_match/pred_set")
    G = _ensure_BR1HW(raters,   "dice_match/raters")
    B = P.size(0)
    out = []

    for b in range(B):
        P_b = _to_probs(P[b])
        G_b = (G[b] > 0.5).float()
        P_bin = (P_b > 0.5).float()

        P_flat = _flatten_set(P_bin)   # (M,N)
        G_flat = _flatten_set(G_b)     # (R,N)
        M, R = P_flat.size(0), G_flat.size(0)

        D = _pairwise_dice(P_flat, G_flat)    # (M,R)
        cost = (1.0 - D).cpu().numpy()

        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost)  # type: ignore
            matched = D[row_ind, col_ind].to(P.device)
        else:
            # greedy fallback
            matched_vals = []
            D_copy = D.clone()
            used_p = set(); used_g = set()
            while True:
                D_tmp = D_copy.clone()
                for i in used_p: D_tmp[i, :] = -1.0
                for j in used_g: D_tmp[:, j] = -1.0
                val, idx = torch.max(D_tmp.view(-1), dim=0)
                if val < 0:
                    break
                i = int(idx // D_tmp.size(1)); j = int(idx % D_tmp.size(1))
                if i in used_p or j in used_g:
                    break
                matched_vals.append(val)
                used_p.add(i); used_g.add(j)
                if len(used_p) == M or len(used_g) == R:
                    break
            matched = torch.stack(matched_vals).to(P.device) if matched_vals else torch.zeros((), device=P.device)

        if normalize_by == "gt":
            denom = max(R, 1)
        elif normalize_by == "max":
            denom = max(max(M, R), 1)
        elif normalize_by == "match":
            denom = max(min(M, R), 1)
        else:
            raise ValueError("normalize_by must be 'gt'|'max'|'match'.")

        out.append(matched.sum() / float(denom))

    return torch.stack(out).float()  # (B,)


def ged(pred_set: torch.Tensor, raters: torch.Tensor) -> torch.Tensor:
    """
    Generalized Energy Distance for sets with d(x,y)=1-Dice. Lower is better.
    Returns (B,)
    """
    P = _ensure_BM1HW(pred_set, "ged/pred_set")
    G = _ensure_BR1HW(raters,   "ged/raters")
    B = P.size(0)
    out = []

    for b in range(B):
        P_b = (_to_probs(P[b]) > 0.5).float()  # (M,1,H,W)
        G_b = (G[b] > 0.5).float()             # (R,1,H,W)

        P_flat = _flatten_set(P_b)  # (M,N)
        G_flat = _flatten_set(G_b)  # (R,N)

        D_pg = _pairwise_dice(P_flat, G_flat)  # (M,R)
        d_pg = (1.0 - D_pg).mean()

        if P_flat.size(0) > 0:
            D_pp = _pairwise_dice(P_flat, P_flat)
            d_pp = (1.0 - D_pp).mean()
        else:
            d_pp = torch.tensor(0.0, device=P.device)

        if G_flat.size(0) > 0:
            D_gg = _pairwise_dice(G_flat, G_flat)
            d_gg = (1.0 - D_gg).mean()
        else:
            d_gg = torch.tensor(0.0, device=P.device)

        out.append(d_pp + d_gg - 2.0 * d_pg)

    return torch.stack(out).float()


def dice_soft(preds_single: torch.Tensor, raters: torch.Tensor, valid_raters: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Soft Dice vs. neutral mean(valid raters). Returns (B,)
    """
    P = _to_probs(preds_single)
    if P.dim() == 3:
        P = P.unsqueeze(1)  # (B,1,H,W)
    elif P.dim() == 4 and P.size(1) != 1:
        P = P[:, :1]

    R = _ensure_BR1HW(raters, "dice_soft/raters")  # (B,R,1,H,W)
    B = P.size(0)
    if valid_raters is None:
        valid_raters = torch.ones((B, R.size(1)), dtype=torch.bool, device=P.device)

    w = valid_raters.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,R,1,1,1)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    gsoft = (R.float() * w).sum(dim=1, keepdim=True) / denom           # (B,1,1,H,W)
    gsoft = gsoft.squeeze(2)                                           # (B,1,H,W)

    P_flat = P.view(B, 1, -1)[:, 0, :]
    G_flat = gsoft.view(B, 1, -1)[:, 0, :]
    inter = (P_flat * G_flat).sum(dim=1)
    p_sum = P_flat.sum(dim=1)
    g_sum = G_flat.sum(dim=1)
    dice = (2.0 * inter + 1e-7) / (p_sum + g_sum + 1e-7)

    empty_pair = (p_sum == 0) & (g_sum == 0)
    dice = torch.where(empty_pair, torch.ones_like(dice), dice)
    return dice  # (B,)


# -------------------------- personalized metrics --------------------------

def dice_per_rater_mean(preds_single: torch.Tensor, raters: torch.Tensor, valid_raters: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean Dice across valid raters. Returns (B,)
    """
    P = (_to_probs(preds_single) > 0.5).float()
    if P.dim() == 3:
        P = P.unsqueeze(1)       # (B,1,H,W)
    elif P.dim() == 4 and P.size(1) != 1:
        P = P[:, :1]

    R = _ensure_BR1HW(raters, "dice_per_rater_mean/raters")  # (B,R,1,H,W)
    B, Rn = R.size(0), R.size(1)
    if valid_raters is None:
        valid_raters = torch.ones((B, Rn), dtype=torch.bool, device=P.device)

    P_flat = P.view(B, 1, -1)[:, 0, :]  # (B,N)

    dices = []
    for r in range(Rn):
        G = (R[:, r] > 0.5).float()     # (B,1,H,W)
        G_flat = G.view(B, 1, -1)[:, 0, :]
        inter = (P_flat * G_flat).sum(dim=1)
        p_sum = P_flat.sum(dim=1)
        g_sum = G_flat.sum(dim=1)
        d = (2.0 * inter + 1e-7) / (p_sum + g_sum + 1e-7)
        empty_pair = (p_sum == 0) & (g_sum == 0)
        d = torch.where(empty_pair, torch.ones_like(d), d)
        mask = valid_raters[:, r].float()
        d = d * mask
        dices.append(d)

    dices = torch.stack(dices, dim=1)  # (B,R)
    denom = valid_raters.float().sum(dim=1).clamp_min(1.0)
    return (dices.sum(dim=1) / denom)


def dice_follow(preds_single: torch.Tensor, raters: torch.Tensor, follow_idx: torch.Tensor, valid_raters: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Dice vs. the followed rater (index per sample). Returns (B,)
    """
    P = (_to_probs(preds_single) > 0.5).float()
    if P.dim() == 3:
        P = P.unsqueeze(1)
    elif P.dim() == 4 and P.size(1) != 1:
        P = P[:, :1]

    R = _ensure_BR1HW(raters, "dice_follow/raters")
    B, Rn = R.size(0), R.size(1)
    if valid_raters is None:
        valid_raters = torch.ones((B, Rn), dtype=torch.bool, device=P.device)

    follow_idx = follow_idx.clone().to(device=P.device, dtype=torch.long)
    for b in range(B):
        fb = int(follow_idx[b].item())
        if not (0 <= fb < Rn) or not bool(valid_raters[b, fb]):
            valids = torch.nonzero(valid_raters[b], as_tuple=False).flatten()
            follow_idx[b] = valids[0] if valids.numel() > 0 else 0

    idx_b = torch.arange(B, device=P.device)
    G = (R[idx_b, follow_idx] > 0.5).float().unsqueeze(1)  # (B,1,H,W)

    P_flat = P.view(B, 1, -1)[:, 0, :]
    G_flat = G.view(B, 1, -1)[:, 0, :]
    inter = (P_flat * G_flat).sum(dim=1)
    p_sum = P_flat.sum(dim=1)
    g_sum = G_flat.sum(dim=1)
    d = (2.0 * inter + 1e-7) / (p_sum + g_sum + 1e-7)
    empty_pair = (p_sum == 0) & (g_sum == 0)
    d = torch.where(empty_pair, torch.ones_like(d), d)
    return d


# -------------------------- public API --------------------------

@torch.no_grad()
def compute_metrics(
    preds_single: torch.Tensor,              # (B,1,H,W) logits/probs/binary
    raters: torch.Tensor,                    # (B,R,1,H,W) or (B,R,H,W)
    valid_raters: Optional[torch.Tensor] = None,   # (B,R)
    *,
    pred_set: Optional[torch.Tensor] = None,       # (B,M,1,H,W) or (B,M,H,W)
    follow_idx: Optional[torch.Tensor] = None,     # (B,)
    dice_match_norm: str = "gt",
    dice_max_symmetric: bool = False,
    run: Optional[int] = None,                     # include 'run' in output dict if provided
) -> Dict[str, float]:
    """
    Compute MBH-Seg (set-level) + personalized metrics. Return per-batch mean as Python floats.
    Safe for AESPA (multi-rater) and SPA (single-mask R=1).
    """
    device = preds_single.device
    B = preds_single.size(0)

    # default pred_set = single element set
    if pred_set is None:
        pred_set = preds_single.unsqueeze(1)  # (B,1,1,H,W) via later ensure

    # Ensure shapes
    Pset = _ensure_BM1HW(pred_set, "compute_metrics/pred_set")             # (B,M,1,H,W)
    Rset = _ensure_BR1HW(raters,   "compute_metrics/raters")               # (B,R,1,H,W)
    if valid_raters is None:
        valid_raters = torch.ones((B, Rset.size(1)), dtype=torch.bool, device=device)

    # Set-level
    m_dice_max   = dice_max(Pset, Rset, symmetric=dice_max_symmetric).mean().item()
    m_dice_match = dice_match(Pset, Rset, normalize_by=dice_match_norm).mean().item()
    m_ged        = ged(Pset, Rset).mean().item()

    # Personalized (single prediction)
    m_dice_soft  = dice_soft(preds_single, Rset, valid_raters).mean().item()
    m_personal   = dice_per_rater_mean(preds_single, Rset, valid_raters).mean().item()

    if follow_idx is not None:
        m_follow  = dice_follow(preds_single, Rset, follow_idx, valid_raters).mean().item()
        # neutral baseline (mean(valid_raters) @ 0.5)
        w = valid_raters.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)           # (B,R,1,1,1)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        neutral_soft = (Rset.float() * w).sum(dim=1, keepdim=True) / denom           # (B,1,1,H,W)
        neutral = (neutral_soft.squeeze(2) > 0.5).float()
        P_bin = (_to_probs(preds_single) > 0.5).float()
        if P_bin.dim() == 3:
            P_bin = P_bin.unsqueeze(1)
        P_flat = P_bin.view(B, 1, -1)[:, 0, :]
        N_flat = neutral.view(B, 1, -1)[:, 0, :]
        inter = (P_flat * N_flat).sum(dim=1)
        p_sum = P_flat.sum(dim=1)
        n_sum = N_flat.sum(dim=1)
        d_neu = (2.0 * inter + 1e-7) / (p_sum + n_sum + 1e-7)
        empty_pair = (p_sum == 0) & (n_sum == 0)
        d_neu = torch.where(empty_pair, torch.ones_like(d_neu), d_neu).mean().item()
        m_delta = m_follow - d_neu
    else:
        m_follow = 0.0
        m_delta  = 0.0

    out = {
        "GED": m_ged,
        "Dice_max": m_dice_max,
        "Dice_match": m_dice_match,
        "Dice_soft": m_dice_soft,
        "Dice_personal_mean": m_personal,
        "Dice_follow": m_follow,
        "Delta_follow_vs_neutral": m_delta,
    }
    if run is not None:
        out = {"run": int(run), **out}
    return out
