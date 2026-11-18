"""
Program-based FiLM model for knockout fold changes across conditions.

Data:   data/model_inputs.npz (from scripts/load_data.py)
Model:  u_p   = e_p A                                          knockout recipe (K)
        h     = tanh(gamma_c * u_p + beta_c) - tanh(beta_c)    program activity
        Delta = (E B) h                                        predicted fold change (G)
Loss:   weighted squared error, weights 1/SE^2, plus ridge on A and B

Convention: rows are samples, so A and B are both stored as (D, K).

Usage:
  python scripts/program_model.py --mode fit --variant film
  python scripts/program_model.py --mode cv  --variant film --k 3
  python scripts/program_model.py --mode cv  --variant none        # no-context comparison
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_inputs(path, se_floor=0.05, exclude=()):
    d = np.load(path, allow_pickle=True)
    perts = [str(p) for p in d["perts"]]
    keep = [i for i, p in enumerate(perts) if p not in exclude]

    Y = torch.tensor(d["Y"][keep], dtype=torch.float32)        # (P, C, G)
    S = torch.tensor(d["S"][keep], dtype=torch.float32)        # (P, C, G)
    return {
        "Y": Y,
        "S": S,
        "w": 1.0 / S.clamp(min=se_floor) ** 2,                 # (P, C, G)
        "E": torch.tensor(d["E"], dtype=torch.float32),         # (G, D)
        "Ep": torch.tensor(d["Ep"][keep], dtype=torch.float32), # (P, D)
        "perts": [perts[i] for i in keep],
        "conds": [str(c) for c in d["conds"]],
        "genes": [str(g) for g in d["genes"]],
    }


# --------------------------------------------------------------------------
# 2-3. Model
# --------------------------------------------------------------------------
class ProgramModel(torch.nn.Module):
    """variant: 'none' (no context), 'scale' (gamma only), 'film' (gamma + beta)."""

    def __init__(self, D, K, C, variant="film", learn_A=True):
        super().__init__()
        self.K, self.variant = K, variant

        A0 = torch.zeros(D, K)
        A0[torch.arange(K), torch.arange(K)] = 1.0              # start at fixed-PCA recipe
        self.A = torch.nn.Parameter(A0, requires_grad=learn_A)  # (D, K)
        self.B = torch.nn.Parameter(0.01 * torch.randn(D, K))   # (D, K)

        # Learned dials for non-reference conditions only (Rest is fixed)
        self.log_gamma = torch.nn.Parameter(
            torch.zeros(C - 1, K), requires_grad=variant in ("scale", "film"))
        self.beta = torch.nn.Parameter(
            torch.zeros(C - 1, K), requires_grad=variant == "film")

    def dials(self):
        ref = torch.zeros(1, self.K)                             # Rest: gamma = 1, beta = 0
        gamma = torch.exp(torch.cat([ref, self.log_gamma]))     # (C, K)
        beta = torch.cat([ref, self.beta])                      # (C, K)
        return gamma, beta

    def forward(self, E, Ep):
        u = Ep @ self.A                                         # (P, D) @ (D, K) -> (P, K)
        gamma, beta = self.dials()
        pre = gamma[None] * u[:, None, :] + beta[None]          # (P, C, K)
        h = torch.tanh(pre) - torch.tanh(beta)[None]            # minus control
        W = E @ self.B                                          # (G, D) @ (D, K) -> (G, K)
        return torch.einsum("pck,gk->pcg", h, W)                # (P, C, G)


# --------------------------------------------------------------------------
# 4. Loss
# --------------------------------------------------------------------------
def loss_fn(model, data, mask, lam):
    Delta = model(data["E"], data["Ep"])
    m = mask[:, :, None].float() * data["w"]                     # zero weight on held-out pairs
    fit = (m * (data["Y"] - Delta) ** 2).sum() / m.sum()
    reg = lam * (model.A.pow(2).sum() + model.B.pow(2).sum())
    return fit + reg


# --------------------------------------------------------------------------
# 5. Training
# --------------------------------------------------------------------------
def train(data, mask, args, verbose=False):
    P, C, _ = data["Y"].shape
    D = data["E"].shape[1]
    best_loss, best_model = float("inf"), None

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = ProgramModel(D, args.k, C, args.variant, learn_A=not args.freeze_A)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=args.lr)

        for step in range(args.steps):
            opt.zero_grad()
            loss = loss_fn(model, data, mask, args.lam)
            loss.backward()
            opt.step()
            if verbose and seed == 0 and step % 500 == 0:
                print(f"  seed 0  step {step:5d}  loss {loss.item():.4f}")

        with torch.no_grad():
            final = loss_fn(model, data, mask, args.lam).item()
        if final < best_loss:                                    # choose by TRAINING loss only
            best_loss, best_model = final, model

    return best_model, best_loss


# --------------------------------------------------------------------------
# 6. Metrics + evaluation
# --------------------------------------------------------------------------
def score(pred, obs, w):
    return {
        "spearman": spearmanr(pred, obs).correlation,
        "pearson": pearsonr(pred, obs)[0],
        "wrmse": float(np.sqrt((w * (pred - obs) ** 2).sum() / w.sum())),
    }


def leave_one_pair_out(data, args):
    P, C, _ = data["Y"].shape
    rows = []
    for p in range(P):
        for c in range(C):
            mask = torch.ones(P, C, dtype=torch.bool)
            mask[p, c] = False
            model, _ = train(data, mask, args)
            with torch.no_grad():
                pred = model(data["E"], data["Ep"])[p, c].numpy()
            obs = data["Y"][p, c].numpy()
            w = data["w"][p, c].numpy()
            rows.append({"pert": data["perts"][p], "cond": data["conds"][c],
                         **score(pred, obs, w)})
            print(f"  held out {rows[-1]['pert']:7s} {rows[-1]['cond']:8s}  "
                  f"spearman {rows[-1]['spearman']:.3f}")
    return pd.DataFrame(rows)


def make_folds(P, C, split):
    """Partition all (p, c) pairs into folds of 3 held-out pairs each.

    knockout: fold p holds out every condition of knockdown p (unseen knockdown).
    rotate:   pair (p, c) goes to fold (p - c) mod P, so each fold holds out one
              pair per condition from distinct knockdowns (new condition for a
              seen knockdown; every condition keeps training pairs).
    """
    if split == "knockout":
        return [[(p, c) for c in range(C)] for p in range(P)]
    folds = [[] for _ in range(P)]
    for p in range(P):
        for c in range(C):
            folds[(p - c) % P].append((p, c))
    return folds


def k_fold(data, args):
    P, C, _ = data["Y"].shape
    rows = []
    for f, held in enumerate(make_folds(P, C, args.split)):
        mask = torch.ones(P, C, dtype=torch.bool)
        for p, c in held:
            mask[p, c] = False
        model, _ = train(data, mask, args)
        with torch.no_grad():
            Delta = model(data["E"], data["Ep"])
        for p, c in held:
            obs, w = data["Y"][p, c].numpy(), data["w"][p, c].numpy()
            # Baselines use only the training pairs of this fold
            mean_resp = data["Y"][mask[:, c], c].mean(0).numpy()  # same condition
            preds = {f"model_{args.variant}": Delta[p, c].numpy(),
                     "baseline_mean": mean_resp,
                     "baseline_zero": np.zeros_like(obs)}
            for method, pred in preds.items():
                s = score(pred, obs, w)
                if method == "baseline_zero":                    # correlation undefined
                    s["spearman"] = s["pearson"] = np.nan
                rows.append({"fold": f, "pert": data["perts"][p],
                             "cond": data["conds"][c], "method": method, **s})
        print(f"  fold {f}: held out " + ", ".join(
            f"{data['perts'][p]}/{data['conds'][c]}" for p, c in held))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 8. Fit on everything + interpret
# --------------------------------------------------------------------------
def fit_and_report(data, args):
    P, C, _ = data["Y"].shape
    model, loss = train(data, torch.ones(P, C, dtype=torch.bool), args, verbose=True)
    print(f"\nFinal training loss: {loss:.4f}")

    with torch.no_grad():
        Delta = model(data["E"], data["Ep"]).numpy()
        gamma, beta = model.dials()
        u = (data["Ep"] @ model.A).numpy()
        W = (data["E"] @ model.B).numpy()

    print("\nTraining fit per pair (spearman, columns = conditions):")
    for p in range(P):
        s = [score(Delta[p, c], data["Y"][p, c].numpy(), data["w"][p, c].numpy())["spearman"]
             for c in range(C)]
        print(f"  {data['perts'][p]:7s} " + "  ".join(f"{x:.3f}" for x in s))

    print("\nScale gamma (rows = conditions, cols = programs):")
    print(pd.DataFrame(gamma.numpy(), index=data["conds"]).round(3).to_string(header=False))
    print("\nShift beta (rows = conditions, cols = programs):")
    print(pd.DataFrame(beta.numpy(), index=data["conds"]).round(3).to_string(header=False))

    print("\nKnockout recipes u_p:")
    print(pd.DataFrame(u, index=data["perts"]).round(3).to_string())

    print("\nTop genes per program:")
    for k in range(args.k):
        top = np.argsort(-np.abs(W[:, k]))[:10]
        print(f"  program {k}: " + ", ".join(
            f"{data['genes'][g]}({W[g, k]:+.2f})" for g in top))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="data/model_inputs.npz")
    ap.add_argument("--mode", choices=["fit", "cv", "kfold"], default="fit")
    ap.add_argument("--split", choices=["rotate", "knockout"], default="rotate",
                    help="kfold: which 3 pairs each fold holds out")
    ap.add_argument("--out", default=None, help="kfold: CSV path for per-pair scores")
    ap.add_argument("--variant", choices=["none", "scale", "film"], default="film")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--freeze-A", action="store_true", help="use fixed PCA recipes")
    ap.add_argument("--exclude", nargs="*", default=[], help="knockouts to drop, e.g. KMT2E")
    args = ap.parse_args()

    data = load_inputs(args.inputs, exclude=args.exclude)
    P, C, G = data["Y"].shape
    print(f"Data: {P} knockouts x {C} conditions x {G} genes, D = {data['E'].shape[1]}")
    print(f"Model: variant={args.variant}  K={args.k}  lambda={args.lam}  "
          f"A {'frozen' if args.freeze_A else 'learned'}")

    if args.mode == "fit":
        fit_and_report(data, args)
    else:
        if args.mode == "kfold":
            df = k_fold(data, args)
            if args.out:
                df.assign(split=args.split, variant=args.variant, k=args.k,
                          lam=args.lam).to_csv(args.out, index=False)
                print(f"\nSaved per-pair scores to {args.out}")
            model_df = df[df["method"] == f"model_{args.variant}"]
            print("\nHeld-out spearman, model (rows = knockdowns, cols = conditions):")
            print(model_df.pivot(index="pert", columns="cond", values="spearman")
                  .reindex(index=data["perts"], columns=data["conds"]).round(3).to_string())
            print("\nMean over held-out pairs, by method:")
            print(df.groupby("method")[["spearman", "pearson", "wrmse"]].mean()
                  .round(3).to_string())
            return
        df = leave_one_pair_out(data, args)
        print("\nMean over held-out pairs:")
        print(df[["spearman", "pearson", "wrmse"]].mean().round(3).to_string())
        print("\nBy condition:")
        print(df.groupby("cond")[["spearman", "wrmse"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
