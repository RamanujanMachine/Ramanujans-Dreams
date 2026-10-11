"""
Dwell-time measurement for the strict-vs-closed recession-cone decision.

Question: if we relax the PT walker's accept region from the STRICT interior
(``B z < 0``, current production) to the CLOSED cone (exact integer ``A v <= 0``,
the candidate), does the cold harvester get *stuck* on the lower-dimensional
non-forced faces (directions parallel to a live facet)?

Method: a faithful pure-Python port of ``_pt_mcmc_walk`` (same proposal mixture,
flatland cage, tempered Metropolis on gravity+repulsion, replica swaps, PID,
burn-in, adaptive harvest radius) with a ``cone_mode`` toggle:
  * "strict"  -> reject unless  B z < -tol           (current production, float B)
  * "closed"  -> reject unless  A_prime (Z z) <= 0    (candidate, EXACT integer)
The conditioned Z / B / seed are taken from a real ParallelTemperingSampler so the
geometry is identical to production.

Instrumentation (cold chain = replica 0 only):
  * face_hits  : accepted states lying on >=1 NON-forced facet (exact A_i v == 0)
  * dwell runs : consecutive cold-chain steps spent on a face before returning interior
We report both modes; strict is the baseline (should ~never touch a face).
"""
import time
import numpy as np

from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler as PT


def gcd_abs(v):
    g = 0
    for x in v:
        x = abs(int(x))
        while x:
            g, x = x, g % x
    return g


def walk(Z, B_float, A_prime, forced_mask, z0, *, cone_mode,
         beta_ladder=(1.0, 0.1, 0.01, 0.0), quota=80,
         initial_lambda=0.5, gamma=1.0, target_yield_ratio=0.01, learning_rate=0.5,
         min_gravity_floor=0.05, monitor_window=500, repulsion_subset=50,
         max_useful_norm=1000.0, flatland_box=10000, burn_in_steps=2000,
         initial_harvest_limit=50.0, harvest_expansion_factor=1.5,
         measure_steps=15000, tol=1e-6, rng_seed=12345):
    """Instrumented port of _pt_mcmc_walk. Returns (harvest_list, accept_rate, stats).

    NOTE: runs a FIXED budget (burn_in + measure_steps) and never exits on quota, so
    the dwell statistics cover the full thermalised walk (incl. adaptive-radius growth),
    not just the pre-quota transient.  ``quota`` only caps the banking array size.
    """
    rng = np.random.default_rng(rng_seed)
    d_flat = Z.shape[1]
    d_orig = Z.shape[0]
    m = B_float.shape[0]
    n_rep = len(beta_ladder)
    beta_ladder = np.asarray(beta_ladder, float)
    nonforced_rows = A_prime[~forced_mask]               # exact integer, original space

    max_steps = burn_in_steps + measure_steps
    swap_interval = max(20, max_steps // 100)

    v0 = Z @ z0
    seed_norm = float(np.linalg.norm(v0))
    z_curr = np.tile(z0.astype(np.int64), (n_rep, 1))
    v_curr = np.tile((Z @ z0).astype(float), (n_rep, 1))
    norm_curr = np.full(n_rep, seed_norm)
    stride = np.full(n_rep, 10, dtype=np.int64)

    harvest = []
    harvest_unit = np.zeros((quota, d_orig))
    hc = 0
    lam = initial_lambda
    window_yield = 0
    burn = 0
    cur_limit = min(initial_harvest_limit, max_useful_norm)
    total_prop = total_acc = 0

    # cold-chain dwell instrumentation
    cold_steps = 0          # accepted cold-chain moves after burn-in
    cold_on_face = 0
    dwell_runs = []         # list of consecutive-on-face run lengths
    run_len = 0

    def in_cone(z_prop, v_int):
        if cone_mode == "strict":
            return bool(np.all(B_float @ z_prop < -tol))
        else:  # closed: exact integer A v <= 0
            return bool(np.all(nonforced_rows @ v_int <= 0))  # forced rows == 0 by construction

    step = 0
    while step < max_steps:
        if norm_curr[0] <= max_useful_norm:
            burn += 1
        for i in range(n_rep):
            total_prop += 1
            z_prop = z_curr[i].copy()
            r = rng.random()
            is_scale = False
            if r < 0.60:
                a = rng.integers(d_flat); z_prop[a] += 1 if rng.random() < 0.5 else -1
            elif r < 0.85:
                a = rng.integers(d_flat); b = rng.integers(d_flat)
                z_prop[a] += 1 if rng.random() < 0.5 else -1
                z_prop[b] += 1 if rng.random() < 0.5 else -1
            elif r < 0.95:
                dim = rng.integers(d_flat); z_prop[dim] += int(stride[i]) * (1 if rng.random() < 0.5 else -1)
                is_scale = True
            else:
                z_prop += rng.integers(-2, 3, size=d_flat)

            if np.max(np.abs(z_prop)) > flatland_box:
                if is_scale and stride[i] > 2: stride[i] -= 1
                continue

            v_prop = Z @ z_prop
            v_int = np.round(v_prop).astype(np.int64)
            if not in_cone(z_prop, v_int):
                if is_scale and stride[i] > 2: stride[i] -= 1
                continue
            norm_prop = float(np.linalg.norm(v_prop))
            if norm_prop < 1e-12:
                if is_scale and stride[i] > 2: stride[i] -= 1
                continue

            s_prop = s_cur = 0.0
            if hc > 0:
                n_sub = min(repulsion_subset, hc)
                idx = rng.integers(0, hc, size=n_sub)
                U = harvest_unit[idx]
                dp = (U @ v_prop) / norm_prop
                s_prop = float(dp.max()) if dp.size else 0.0
                if norm_curr[i] > 1e-12:
                    dc = (U @ v_curr[i]) / norm_curr[i]
                    s_cur = float(dc.max()) if dc.size else 0.0
                s_prop = max(s_prop, 0.0); s_cur = max(s_cur, 0.0)

            beta_i = beta_ladder[i]
            diff = beta_i * (lam * norm_curr[i] + gamma * s_cur) - beta_i * (lam * norm_prop + gamma * s_prop)
            accept = diff >= 0.0 or rng.random() < np.exp(min(diff, 0.0))

            if is_scale:
                if accept and stride[i] < 50: stride[i] += 1
                elif not accept and stride[i] > 2: stride[i] -= 1

            if accept:
                total_acc += 1
                z_curr[i] = z_prop; v_curr[i] = v_prop; norm_curr[i] = norm_prop

                if i == 0 and burn > burn_in_steps:
                    cold_steps += 1
                    on_face = bool(np.any(nonforced_rows @ v_int == 0)) if nonforced_rows.shape[0] else False
                    if on_face:
                        cold_on_face += 1; run_len += 1
                    else:
                        if run_len: dwell_runs.append(run_len)
                        run_len = 0

                    if norm_prop <= cur_limit and hc < quota:
                        g = gcd_abs(v_int)
                        if g == 1 and (hc == 0 or not np.array_equal(harvest[-1], v_int)):
                            harvest.append(v_int.copy())
                            harvest_unit[hc] = v_prop / norm_prop
                            hc += 1; window_yield += 1

        if (step + 1) % swap_interval == 0:
            for i in range(n_rep - 1):
                j = i + 1
                a_swap = (beta_ladder[i] - beta_ladder[j]) * (lam * norm_curr[i] - lam * norm_curr[j])
                if a_swap >= 0.0 or rng.random() < np.exp(min(a_swap, 0.0)):
                    z_curr[[i, j]] = z_curr[[j, i]]; v_curr[[i, j]] = v_curr[[j, i]]
                    norm_curr[[i, j]] = norm_curr[[j, i]]

        if (step + 1) % monitor_window == 0:
            if norm_curr[0] > max_useful_norm or burn <= burn_in_steps:
                lam = initial_lambda
            else:
                if window_yield == 0 and cur_limit < max_useful_norm:
                    cur_limit = min(max_useful_norm, cur_limit * harvest_expansion_factor)
                ratio = (window_yield / monitor_window + 1e-5) / target_yield_ratio
                lam = lam * np.exp(learning_rate * np.log(ratio))
            lam = min(max(lam, min_gravity_floor), initial_lambda)
            window_yield = 0
        step += 1

    if run_len:
        dwell_runs.append(run_len)
    accept_rate = total_acc / total_prop if total_prop else 0.0
    stats = dict(cold_steps=cold_steps, cold_on_face=cold_on_face,
                 dwell_runs=dwell_runs, steps=step)
    return harvest, accept_rate, stats


def split_forced(A_prime):
    """Return (forced_mask, d_flat) using the conditioner's own implicit-equality LP."""
    cond = HyperSpaceConditioner(np.asarray(A_prime, float), max_beta=25)
    E, Bo = cond._extract_constraints()
    Z = cond._compute_integer_basis(E)
    forced = np.zeros(len(A_prime), dtype=bool)
    for k, row in enumerate(A_prime):
        if any(np.array_equal(row, e.astype(A_prime.dtype)) for e in E.astype(np.int64)):
            forced[k] = True
    return forced, Z.shape[1]


def run(name, A_prime, quota=100000):
    A_prime = np.asarray(A_prime, dtype=np.int64)
    forced, d_flat = split_forced(A_prime)
    sampler = PT(A_prime, rng_seed=12345)
    z0 = sampler._compute_chebyshev_center()
    print("=" * 90)
    print(f"[{name}]  d_orig={A_prime.shape[1]} -> d_flat={d_flat} | "
          f"forced={int(forced.sum())} nonforced={int((~forced).sum())}")
    for mode in ("strict", "closed"):
        t = time.time()
        harvest, acc, st = walk(sampler.Z, sampler.B, A_prime, forced, z0,
                                cone_mode=mode, quota=quota, rng_seed=12345)
        runs = st["dwell_runs"]
        cs = st["cold_steps"]
        frac = (st["cold_on_face"] / cs) if cs else 0.0
        mean_dwell = (np.mean(runs) if runs else 0.0)
        max_dwell = (max(runs) if runs else 0)
        print(f"  [{mode:6}] banked={len(harvest):4d}  accept={acc*100:5.1f}%  "
              f"cold_steps={cs:5d}  on_face={frac*100:5.1f}%  "
              f"face_visits={len(runs):4d}  mean_dwell={mean_dwell:5.2f}  max_dwell={max_dwell:4d}  "
              f"({time.time()-t:.1f}s)")


if __name__ == "__main__":
    from tests.testing_tool import TestHarness  # for _generate_matrix archetypes
    gen = TestHarness._generate_matrix
    for scen in ("10D_Fat_Baseline", "15D_Needle", "15D_Pancake"):
        run(scen, gen(scen, seeding=True))

    # Degenerate corridor: hidden forced walls (v0=v1=0) + live facets in R^6.
    corridor = np.array([
        [1, 1, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0],  # force v0=v1=0
        [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1],     # live facets
    ])
    run("6D_Corridor (hidden walls + live facets)", corridor)
