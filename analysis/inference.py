# Shared inference for 05 to 07 and 09: OLS with CR1 cluster-robust standard errors, the
# wild cluster bootstrap, and the pool clusters (cases with the same knife or glove pool).
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

# Webb (2023) six-point weights for few clusters.
WEBB_POINTS = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                        np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# OLS with CR1 cluster-robust standard errors (Cameron and Miller 2015), t(G - 1).
class OLS:
    def __init__(self, y: np.ndarray, X: np.ndarray, cluster: np.ndarray,
                 names: list[str]):
        self.names = names
        self.y, self.X, self.cluster = y, X, cluster
        self.n, self.k = X.shape
        self.xtx_inv = np.linalg.pinv(X.T @ X)
        self.beta = self.xtx_inv @ (X.T @ y)
        self.resid = y - X @ self.beta
        self.r2 = 1 - (self.resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
        groups = np.unique(cluster)
        meat = np.zeros((self.k, self.k))
        for g in groups:
            m = cluster == g
            u = X[m].T @ self.resid[m]
            meat += np.outer(u, u)
        g = len(groups)
        c = (g / max(g - 1, 1)) * ((self.n - 1) / max(self.n - self.k, 1))
        cov = c * self.xtx_inv @ meat @ self.xtx_inv
        self.se = np.sqrt(np.maximum(np.diag(cov), 0))
        self.t = self.beta / self.se
        self.n_clusters = g
        self.df = max(g - 1, 1)
        self.p = 2 * stats.t.sf(np.abs(self.t), self.df)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.beta, "se": self.se, "t": self.t,
                             "p": self.p}, index=self.names)


def weight_scheme(n_clusters: int) -> str:
    return "webb" if n_clusters < C.WEBB_MAX_CLUSTERS else "rademacher"


# Wild cluster bootstrap p-value with the null imposed (restricted residuals),
# symmetric in |t|; Webb weights below WEBB_MAX_CLUSTERS clusters, else Rademacher.
def wild_bootstrap(y: np.ndarray, X: np.ndarray, cluster: np.ndarray,
                   j: int) -> float:
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    keep = [i for i in range(X.shape[1]) if i != j]
    Xr = X[:, keep]
    fitted_r = Xr @ (np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y))
    resid_r = y - fitted_r
    names = [str(i) for i in range(X.shape[1])]
    t_obs = abs(OLS(y, X, cluster, names).t[j])
    # Tolerance, so that exact ties count on every machine.
    t_obs *= 1 - 1e-9
    groups, idx = np.unique(cluster, return_inverse=True)
    webb = weight_scheme(len(groups)) == "webb"
    count = 0
    for _ in range(C.BOOTSTRAP_REPS):
        if webb:
            w = rng.choice(WEBB_POINTS, size=len(groups))
        else:
            w = rng.choice([-1.0, 1.0], size=len(groups))
        t_star = abs(OLS(fitted_r + resid_r * w[idx], X, cluster, names).t[j])
        count += t_star >= t_obs
    return (count + 1) / (C.BOOTSTRAP_REPS + 1)


# p-value behind the significance stars: CR1 t-test from WEBB_MAX_CLUSTERS clusters on,
# below that the wild bootstrap.
def star_p(p_cr1: float, p_wild: float, n_clusters: int) -> float:
    return p_wild if n_clusters < C.WEBB_MAX_CLUSTERS else p_cr1


# Design matrix: constant, regressors, fixed-effect dummies.
def design(df: pd.DataFrame, cols: list[str], fe: list[str] | None = None
           ) -> tuple[np.ndarray, list[str]]:
    parts = [np.ones((len(df), 1))]
    names = ["const"]
    for c in cols:
        parts.append(df[[c]].to_numpy(dtype=float))
        names.append(c)
    for f in fe or []:
        d = pd.get_dummies(df[f].astype(str), prefix=f, drop_first=True,
                           dtype=float)
        if d.shape[1]:
            parts.append(d.to_numpy())
            names.extend(d.columns)
    return np.column_stack(parts), names


# Pool = cases with the same set of knife or glove finishes (case_pools.csv).
@lru_cache(maxsize=1)
def _pool_tables() -> tuple[dict, dict]:
    pools = pd.read_csv(C.CASE_POOLS, encoding="utf-8-sig")
    sets = pools[~pools["stattrak"]].groupby("case")["skin"].apply(frozenset)
    members: dict[frozenset, list[str]] = {}
    for case, s in sets.items():
        members.setdefault(s, []).append(case)
    case_pool = {}
    for cases in members.values():
        name = " / ".join(sorted(cases))
        for case in cases:
            case_pool[case] = name
    item_pools = (pools.assign(pool=pools["case"].map(case_pool))
                  .groupby("market_hash_name")["pool"].apply(frozenset).to_dict())
    return case_pool, item_pools


def pool_of_case(case: str) -> str:
    return _pool_tables()[0][case]


# Cluster per item: its pool; outputs that lie in several pools merge those pools
# (connected component), so that dependent items always share a cluster.
def pool_clusters(names: pd.Index, cases: pd.Series) -> pd.Series:
    case_pool, item_pools = _pool_tables()
    per_item = {}
    for name in names:
        if name in item_pools:
            per_item[name] = set(item_pools[name])
        else:
            per_item[name] = {case_pool[cases[name]]}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ps in per_item.values():
        ps = sorted(ps)
        for p in ps[1:]:
            parent[find(p)] = find(ps[0])
    return pd.Series({n: find(sorted(ps)[0]) for n, ps in per_item.items()},
                     name="pool").reindex(names)
