"""
Simulation study for "Passive Runtime Integrity Verification for Edge AI Systems
via Multi-Layer Telemetry Correlation".

Self-contained numpy-only implementation. No sklearn / scipy required.

Outputs:
  results/results.json
  ../figures/{roc_grid, score_distributions, attack_signatures}.{png,pdf}
"""

from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(OUT_DIR, "..", "figures"))
RES_DIR = os.path.join(OUT_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


CHANNELS = [
    "cpu_freq_ghz",
    "gpu_util_pct",
    "soc_temp_c",
    "infer_latency_ms",
    "pkg_power_w",
    "mem_bw_pct",
    "ctx_switch_kHz",
    "net_bytes_per_s",
]


@dataclass
class Profile:
    name: str
    mean: np.ndarray
    std: np.ndarray
    coupling: np.ndarray = field(default_factory=lambda: np.zeros((len(CHANNELS), 2)))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        d = len(CHANNELS)
        z_indep = rng.standard_normal(size=(n, d))
        k = self.coupling.shape[1]
        z_shared = rng.standard_normal(size=(n, k))
        x = z_indep + z_shared @ self.coupling.T
        x = x * self.std + self.mean
        x[:, 0] = np.clip(x[:, 0], 0.1, 2.4)
        x[:, 1] = np.clip(x[:, 1], 0.0, 100.0)
        x[:, 2] = np.clip(x[:, 2], 25.0, 95.0)
        x[:, 3] = np.clip(x[:, 3], 1.0, 500.0)
        x[:, 4] = np.clip(x[:, 4], 2.0, 40.0)
        x[:, 5] = np.clip(x[:, 5], 0.0, 100.0)
        x[:, 6] = np.clip(x[:, 6], 0.1, 200.0)
        x[:, 7] = np.clip(x[:, 7], 0.0, 5e7)
        return x


def _coupling(d=8, k=2, scale=0.4, rng=None):
    rng = rng or RNG
    return scale * rng.standard_normal(size=(d, k))


BENIGN_PROFILES = [
    Profile(
        name="surveillance_idle",
        mean=np.array([0.6, 8.0, 42.0, 18.0, 5.5, 12.0, 1.8, 6e3]),
        std=np.array([0.05, 3.5, 1.5, 2.0, 0.6, 3.0, 0.3, 1.5e3]),
        coupling=_coupling(scale=0.25),
    ),
    Profile(
        name="active_inference",
        mean=np.array([1.5, 62.0, 56.0, 32.0, 15.5, 48.0, 18.0, 1.4e6]),
        std=np.array([0.10, 8.0, 2.5, 3.5, 1.6, 7.0, 2.5, 4e5]),
        coupling=_coupling(scale=0.45),
    ),
    Profile(
        name="peak_traffic",
        mean=np.array([1.85, 84.0, 64.0, 41.0, 22.0, 70.0, 28.0, 4.2e6]),
        std=np.array([0.08, 5.0, 2.0, 4.0, 1.4, 6.0, 3.0, 9e5]),
        coupling=_coupling(scale=0.55),
    ),
    Profile(
        name="standby",
        mean=np.array([0.30, 1.5, 38.0, 4.0, 3.2, 4.0, 0.9, 1.5e3]),
        std=np.array([0.03, 1.0, 1.0, 1.5, 0.3, 1.5, 0.2, 4e2]),
        coupling=_coupling(scale=0.15),
    ),
]


# ---------- attacks ----------
def attack_cryptomining(x, alpha):
    x = x.copy()
    x[:, 0] += alpha * 0.18
    x[:, 1] += alpha * 25.0
    x[:, 2] += alpha * 4.5
    x[:, 4] += alpha * 3.5
    x[:, 5] += alpha * 18.0
    x[:, 6] += alpha * 3.0
    return x


def attack_hidden_inference(x, alpha):
    x = x.copy()
    x[:, 1] += alpha * 14.0
    x[:, 3] += alpha * 8.0
    x[:, 4] += alpha * 1.6
    x[:, 5] += alpha * 9.0
    x[:, 6] += alpha * 1.4
    return x


def attack_beacon_malware(x, alpha):
    x = x.copy()
    n = x.shape[0]
    periodic = np.zeros(n)
    periodic[::60] = 1.0
    x[:, 7] += alpha * 4.5e4 * np.clip(periodic + 0.05 * RNG.standard_normal(n), 0, None)
    x[:, 6] += alpha * 0.35
    x[:, 4] += alpha * 0.20
    return x


def attack_firmware_boot(x, alpha):
    x = x.copy()
    x[:, 6] += alpha * 4.0
    x[:, 2] += alpha * 1.8
    x[:, 4] += alpha * 0.6
    x[:, 0] -= alpha * 0.05
    return x


ATTACKS = {
    "cryptomining": attack_cryptomining,
    "hidden_inference": attack_hidden_inference,
    "beacon_malware": attack_beacon_malware,
    "firmware_boot": attack_firmware_boot,
}


# ---------- detectors ----------
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + 1e-9
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_


class IsolationForestLite:
    def __init__(self, n_trees=80, sub=256, seed=0):
        self.n_trees = n_trees
        self.sub = sub
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def _grow(self, x, depth, max_depth):
        n, d = x.shape
        if n <= 1 or depth >= max_depth:
            return ("leaf", n)
        f = int(self.rng.integers(0, d))
        lo, hi = x[:, f].min(), x[:, f].max()
        if hi - lo < 1e-12:
            return ("leaf", n)
        t = self.rng.uniform(lo, hi)
        left = x[x[:, f] < t]
        right = x[x[:, f] >= t]
        return ("node", f, t,
                self._grow(left, depth + 1, max_depth),
                self._grow(right, depth + 1, max_depth))

    def fit(self, x):
        n = x.shape[0]
        sub = min(self.sub, n)
        max_depth = int(math.ceil(math.log2(max(sub, 2))))
        self.trees = []
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, size=sub, replace=False)
            self.trees.append(self._grow(x[idx], 0, max_depth))
        return self

    @staticmethod
    def _path_len(node, x_row, depth=0):
        if node[0] == "leaf":
            n = node[1]
            if n <= 1:
                return depth
            H = math.log(n - 1) + 0.5772156649 if n > 1 else 0.0
            return depth + 2 * H - 2 * (n - 1) / n
        f, t = node[1], node[2]
        return IsolationForestLite._path_len(
            node[3] if x_row[f] < t else node[4], x_row, depth + 1)

    def score(self, x):
        sub = self.sub
        H = math.log(sub - 1) + 0.5772156649 if sub > 1 else 1.0
        c = 2 * H - 2 * (sub - 1) / sub
        s = np.zeros(x.shape[0])
        for i, row in enumerate(x):
            pls = [self._path_len(t, row) for t in self.trees]
            s[i] = 2 ** (-np.mean(pls) / c)
        return s


class AELite:
    """Linear AE (closed-form PCA)."""

    def __init__(self, k=3):
        self.k = k
        self.mean_ = None
        self.components_ = None

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        self.components_ = vt[: self.k]
        return self

    def score(self, x):
        xc = x - self.mean_
        proj = xc @ self.components_.T
        recon = proj @ self.components_
        return np.linalg.norm(xc - recon, axis=1)


class MultiLayerCorrelation:
    """Proposed: mode-aware MLC. For each benign profile fit (mu, sd, V_k).
       Score = min over profiles of [mean z-score + lambda * subspace residual]."""

    def __init__(self, lam=0.8, k=2):
        self.lam = lam
        self.k = k
        self.profiles_ = []

    def fit_profiles(self, profile_data):
        self.profiles_ = []
        for x in profile_data:
            mu = x.mean(axis=0)
            sd = x.std(axis=0) + 1e-9
            xn = (x - mu) / sd
            _, _, vt = np.linalg.svd(xn, full_matrices=False)
            comp = vt[: self.k]
            self.profiles_.append({"mu": mu, "sd": sd, "comp": comp})
        return self

    def _score_p(self, x, p):
        xn = (x - p["mu"]) / p["sd"]
        z = np.mean(np.abs(xn), axis=1)
        proj = xn @ p["comp"].T
        recon = proj @ p["comp"]
        codev = np.linalg.norm(xn - recon, axis=1)
        return z + self.lam * codev

    def score(self, x):
        s = np.stack([self._score_p(x, p) for p in self.profiles_], axis=1)
        return s.min(axis=1)


# ---------- metrics ----------
def roc_curve(y, s):
    order = np.argsort(-s)
    y_s = y[order]
    P = y.sum()
    N = len(y) - P
    tps = np.cumsum(y_s == 1)
    fps = np.cumsum(y_s == 0)
    return fps / max(N, 1), tps / max(P, 1), s[order]


def auc_score(fpr, tpr):
    return float(np.trapz(tpr, fpr))


def best_f1(y, s):
    order = np.argsort(-s)
    y_s = y[order]
    P = y.sum()
    tps = np.cumsum(y_s == 1).astype(float)
    fps = np.cumsum(y_s == 0).astype(float)
    pp = np.arange(1, len(y) + 1, dtype=float)
    precision = tps / pp
    recall = tps / max(P, 1)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(np.max(f1))


def fpr_at_tpr(y, s, target=0.95):
    fpr, tpr, _ = roc_curve(y, s)
    m = tpr >= target
    if not m.any():
        return 1.0
    return float(fpr[m][0])


# ---------- experiment ----------
def build_benign_dataset(n_per_profile=2000):
    parts = []
    for p in BENIGN_PROFILES:
        parts.append(p.sample(n_per_profile, RNG))
    return np.vstack(parts)


def build_attack_dataset(attack_fn, n=2400, alphas=(0.4, 0.7, 1.0)):
    n_pp = n // len(BENIGN_PROFILES)
    base = build_benign_dataset(n_pp)
    alpha = RNG.choice(alphas, size=base.shape[0])
    return attack_fn(base, alpha)


def run():
    print("[run] starting", flush=True)
    n_per = 2500
    per_profile = [p.sample(n_per, RNG) for p in BENIGN_PROFILES]
    benign_train = np.vstack(per_profile)
    benign_test = build_benign_dataset(n_per_profile=600)

    scaler = StandardScaler().fit(benign_train)
    btr = scaler.transform(benign_train)

    print("[run] fitting IF", flush=True)
    iso = IsolationForestLite(n_trees=80, sub=256, seed=1).fit(btr)
    ae = AELite(k=3).fit(btr)
    mlc = MultiLayerCorrelation(lam=0.8, k=2).fit_profiles(per_profile)

    def score_iso(x_raw): return iso.score(scaler.transform(x_raw))
    def score_ae(x_raw): return ae.score(scaler.transform(x_raw))
    def score_mlc(x_raw): return mlc.score(x_raw)

    detectors = {
        "IF-LITE": score_iso,
        "AE-LITE": score_ae,
        "MLC (proposed)": score_mlc,
    }

    results = {}
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (atk_name, atk_fn) in zip(axes, ATTACKS.items()):
        print(f"[run] attack: {atk_name}", flush=True)
        atk_data = build_attack_dataset(atk_fn, n=2400)
        x_all = np.vstack([benign_test, atk_data])
        y_all = np.concatenate([np.zeros(benign_test.shape[0]),
                                 np.ones(atk_data.shape[0])])
        results[atk_name] = {}
        for name, sfn in detectors.items():
            s = sfn(x_all)
            fpr, tpr, _ = roc_curve(y_all, s)
            auc = auc_score(fpr, tpr)
            f1 = best_f1(y_all, s)
            f95 = fpr_at_tpr(y_all, s, 0.95)
            results[atk_name][name] = {"auc": auc, "f1": f1, "fpr_at_95tpr": f95}
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.6)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Attack: {atk_name.replace('_', ' ')}", fontsize=11)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("ROC curves vs four simulated attack classes", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roc_grid.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "roc_grid.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- MLC score distributions ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    benign_scores = mlc.score(benign_test)
    for ax, (atk_name, atk_fn) in zip(axes, ATTACKS.items()):
        atk_data = build_attack_dataset(atk_fn, n=2400)
        atk_scores = mlc.score(atk_data)
        ax.hist(benign_scores, bins=50, alpha=0.55, label="Benign", color="#3878c5")
        ax.hist(atk_scores, bins=50, alpha=0.55, label=atk_name, color="#c0392b")
        ax.set_title(f"MLC: benign vs {atk_name.replace('_',' ')}", fontsize=11)
        ax.set_xlabel("Integrity-deviation score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "score_distributions.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "score_distributions.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- per-channel signature bars ----
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.18
    xs = np.arange(len(CHANNELS))
    bm = benign_train.mean(axis=0)
    bs = benign_train.std(axis=0) + 1e-9
    for i, (atk_name, atk_fn) in enumerate(ATTACKS.items()):
        atk = build_attack_dataset(atk_fn, n=2000, alphas=(0.7,))
        dz = (atk.mean(axis=0) - bm) / bs
        ax.bar(xs + (i - 1.5) * width, dz, width=width, label=atk_name)
    ax.set_xticks(xs)
    ax.set_xticklabels(CHANNELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Channel z-shift vs benign baseline")
    ax.set_title("Per-channel telemetry signature of each simulated attack")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "attack_signatures.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "attack_signatures.pdf"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RES_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("[run] done", flush=True)
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
