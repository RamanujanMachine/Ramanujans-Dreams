"""Diagnostic harness for the DiscreteMCMCSampler.

Companion to ``tests/testing_tool.py`` (the raycast harness).  It reuses the same
synthetic shard archetypes (fat baseline / needle / pancake), adds a real-CMF
degenerate matrix, and renders an **8-pane (4x2)** diagnostic dashboard for the
discrete walk:

Row 1 (mechanics): unit-direction PCA, nearest-neighbour angle histogram,
boundary-slack histogram, L2-norm-vs-quota timeline.
Row 2 (uniformity): radial-volume CDF, angular-uniformity CDF, trajectory-length
histogram, ranked-L2-norm line.

The CDFs and the ranked-L2 line expose skew that medians/histograms hide.  PCA is
computed on **unit directions** ``v/||v||`` (not raw coordinates): raw-coordinate PCA
is dominated by radial norm spread and misreports a multi-dimensional angular cloud as
a 1D corridor.  See ``context/sampling_trajectories/SAMPLING_MATH.md`` Section 12.12.
"""

import csv
import time

import matplotlib.pyplot as plt

from dreamer.utils.rand import np
from dreamer.extraction.samplers.discrete_raycaster import DiscreteMCMCSampler
from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler
from dreamer.extraction.samplers.raw_space_raycaster import RawSpaceMCMCSampler
from tests.testing_tool import TestHarness


def _unit_pca_spectrum(rays):
    """Explained-variance ratio of the harvested **unit directions**.

    :param rays: ``(n, d)`` harvested integer vectors.
    :return: ``(variance_ratio, effective_dim_90)`` — the per-component variance ratio
        of ``v/||v||`` and the number of components needed to reach 90% variance.
    """
    lengths = np.linalg.norm(rays, axis=1)
    units = rays / lengths[:, None]
    centered = units - units.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    var = sv ** 2
    var_ratio = var / var.sum() if var.sum() > 0 else var
    eff_dim = int(np.searchsorted(np.cumsum(var_ratio), 0.90) + 1)
    return var_ratio, eff_dim


class DiscreteTestHarness(TestHarness):
    """Evaluation framework for the discrete lattice sampler.

    Inherits the synthetic ``A_prime`` generators from :class:`TestHarness`, adds a
    real-CMF degenerate matrix, and renders the 8-pane discrete-walk dashboard.
    """

    #: A real CMF-style arrangement with duplicate / overlapping hyperplanes (D=7).
    #: Exercises the engine on degenerate, non-simple geometry rather than the
    #: synthetic fat/needle/pancake archetypes.
    REAL_WORLD_DEGENERATE = np.array([
        [0, -1, 0, 0, 0, 0, 1],
        [-1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, -1, 0],
        [0, -1, 0, 0, 0, 0, 1],
        [-1, 0, 0, 0, 0, 0, 1],
        [0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, -1],
        [0, 0, 1, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, -1],
        [0, 0, 1, 0, -1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, -1],
        [0, 0, 1, 0, 0, -1, 0],
    ])

    def __init__(self, engine_class=DiscreteMCMCSampler):
        """Bind the harness to a discrete-lattice sampler engine.

        :param engine_class: the sampler class to evaluate; defaults to
            :class:`DiscreteMCMCSampler`.  Pass :class:`ParallelTemperingSampler`
            (or any ``Sampler`` exposing ``d_flat`` / ``A_prime`` / ``last_accept_rate``)
            to run that engine through the same 8-pane dashboard for comparison.
        """
        super().__init__(engine_class=engine_class)

    def _generate_matrix(self, scenario_type, seeding=False):
        """Return the constraint matrix for a scenario, adding the real-world case.

        :param scenario_type: archetype key, or ``"7D_Real_World_Degenerate"``.
        :param seeding: forwarded to the base synthetic generator.
        :return: ``(rows, d_orig)`` constraint matrix.
        """
        if scenario_type == "7D_Real_World_Degenerate":
            return self.REAL_WORLD_DEGENERATE.copy()
        return super()._generate_matrix(scenario_type, seeding)

    def render_dashboard(self, scenario_name, save_path=None):
        """Render the 8-pane (4x2) discrete-sampler diagnostic dashboard for one scenario.

        Row 1 (mechanics): unit-direction PCA, NN-angle histogram, boundary-slack
        histogram, L2-norm-vs-quota timeline.  Row 2 (uniformity): radial-volume CDF,
        angular-uniformity CDF, trajectory-length histogram, ranked-L2-norm line.  The
        CDFs and the ranked plot expose skew that medians/histograms can hide.

        :param scenario_name: key of a stored run in ``self.results``.
        :param save_path: if given, save the figure to this path instead of showing it.
        """
        data = self.results.get(scenario_name)
        if not data or "error" in data or len(data["rays"]) == 0:
            print(f"No valid data to plot for {scenario_name}.")
            return

        rays = np.asarray(data["rays"], dtype=np.float64)
        lengths = np.linalg.norm(rays, axis=1)
        u_rays = rays / lengths[:, np.newaxis]
        cos_sim = np.clip(u_rays @ u_rays.T, -1.0, 1.0)
        np.fill_diagonal(cos_sim, -1.0)
        min_angles = np.degrees(np.arccos(np.max(cos_sim, axis=1)))
        engine = data.get("engine")
        accept = getattr(engine, "last_accept_rate", 0.0)

        fig, axs = plt.subplots(2, 4, figsize=(26, 12))
        fig.suptitle(
            f"{scenario_name}  —  yield={len(rays)}, accept={accept * 100:.2f}%", fontsize=14
        )

        # --- Row 1: mechanics ---
        # 1. Unit-direction PCA spectrum.
        var_ratio, eff_dim = _unit_pca_spectrum(rays)
        axs[0, 0].bar(np.arange(1, len(var_ratio) + 1), var_ratio, color="#00FFFF", edgecolor="black")
        axs[0, 0].set_title(f"1. Unit-direction PCA (eff dim@90% = {eff_dim}/{rays.shape[1]})")
        axs[0, 0].set_xlabel("principal component")
        axs[0, 0].set_ylabel("variance ratio")

        # 2. Nearest-neighbour angle histogram.
        axs[0, 1].hist(min_angles, bins=50, color="#FF00FF", edgecolor="black")
        axs[0, 1].set_title("2. Nearest-neighbour angle")
        axs[0, 1].set_xlabel("min angle to neighbour (deg)")

        # 3. Boundary-slack histogram.
        if engine is not None and engine.A_prime.shape[0] > 0:
            slacks = -(engine.A_prime @ rays.T).T.flatten()
            axs[0, 2].hist(slacks, bins=50, color="#00FF00", edgecolor="black")
            axs[0, 2].axvline(0.0, color="red", linestyle="--", label="boundary")
            axs[0, 2].legend()
        axs[0, 2].set_title("3. Boundary slack ( >0 = inside )")
        axs[0, 2].set_xlabel("slack")

        # 4. L2 norm vs. quota timeline (harvest order).
        axs[0, 3].plot(np.arange(1, len(lengths) + 1), lengths, color="#FFAA00", linewidth=1.2)
        axs[0, 3].set_title("4. L2 norm vs. harvest index")
        axs[0, 3].set_xlabel("harvest index (quota fill)")
        axs[0, 3].set_ylabel("L2 norm")

        # --- Row 2: uniformity ---
        # 5. Radial volume CDF (empirical CDF of L2 norms).
        sorted_norms = np.sort(lengths)
        cdf = np.arange(1, len(sorted_norms) + 1) / len(sorted_norms)
        axs[1, 0].plot(sorted_norms, cdf, color="#5DADE2", linewidth=2)
        axs[1, 0].set_title("5. Radial volume CDF")
        axs[1, 0].set_xlabel("L2 norm")
        axs[1, 0].set_ylabel("cumulative fraction")

        # 6. Angular uniformity CDF (CDF of NN angles; convex+smooth = uniform).
        sorted_ang = np.sort(min_angles)
        acdf = np.arange(1, len(sorted_ang) + 1) / len(sorted_ang)
        axs[1, 1].plot(sorted_ang, acdf, color="#AF7AC5", linewidth=2)
        axs[1, 1].set_title("6. Angular uniformity CDF")
        axs[1, 1].set_xlabel("NN angle (deg)")
        axs[1, 1].set_ylabel("cumulative fraction")

        # 7. Shell-density bar chart — absolute count of harvested vectors per norm band.
        #    Shows directly how densely the inner shells (<50, <200) are mined.
        shell_edges = [0, 50, 100, 200, 500, np.inf]
        shell_labels = ["0-50", "50-100", "100-200", "200-500", "500+"]
        shell_counts = [
            int(np.sum((lengths >= shell_edges[k]) & (lengths < shell_edges[k + 1])))
            for k in range(len(shell_labels))
        ]
        bars = axs[1, 2].bar(shell_labels, shell_counts, color="#48C9B0", edgecolor="black")
        axs[1, 2].bar_label(bars, labels=[str(c) for c in shell_counts], padding=2)
        axs[1, 2].set_title("7. Shell density (count per L2-norm band)")
        axs[1, 2].set_xlabel("L2-norm band")
        axs[1, 2].set_ylabel("count")

        # 8. Ranked trajectories (sorted L2 norm; smooth = good, staircase = shell-trapped).
        axs[1, 3].plot(sorted_norms, color="#00FF00", linewidth=2)
        axs[1, 3].set_title("8. Ranked trajectories (sorted L2)")
        axs[1, 3].set_xlabel("rank")
        axs[1, 3].set_ylabel("L2 norm")

        for ax in axs.flat:
            ax.grid(color="grey", linestyle="--", alpha=0.3)

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        if save_path is not None:
            fig.savefig(save_path, dpi=110)
            plt.close(fig)
            print(f"Saved dashboard: {save_path}")
        else:
            plt.show()

    def run_one(self, scenario_name, target_quota=10000, seeding=True):
        """Run a single scenario and stash the engine alongside the harvested rays.

        :param scenario_name: archetype key understood by ``_generate_matrix``.
        :param target_quota: exact number of primitive vectors to request.
        :param seeding: whether to seed the synthetic-matrix RNG for reproducibility.
        """
        A_prime = self._generate_matrix(scenario_name, seeding)
        engine = self.EngineClass(A_prime)
        start = time.time()
        # exact=True so the diagnostic suite always targets the literal quota and never
        # the cone-volume-scaled count (the production path scales; the suite must not).
        rays = engine.harvest(target_quota, exact=True)
        elapsed = time.time() - start
        self.results[scenario_name] = {
            "rays": rays,
            "time": elapsed,
            "yield": len(rays),
            "d_flat": engine.d_flat,
            "engine": engine,
        }
        print(f"[{scenario_name}] yielded {len(rays)} vectors in {elapsed:.2f}s (d_flat={engine.d_flat})")
        return self.results[scenario_name]

    def summary_row(self, scenario_name):
        """Compute a flat dict of diagnostic metrics for one stored run.

        :param scenario_name: key of a stored run in ``self.results``.
        :return: dict of summary metrics (empty-ish row if the run yielded too few points).
        """
        data = self.results.get(scenario_name, {})
        rays = np.asarray(data.get("rays", []), dtype=np.float64)
        engine = data.get("engine")
        row = {
            "scenario": scenario_name,
            "d_orig": getattr(engine, "d_orig", ""),
            "d_flat": data.get("d_flat", ""),
            "yield": data.get("yield", 0),
            "time_s": round(data.get("time", 0.0), 3),
            "accept_rate": round(float(getattr(engine, "last_accept_rate", 0.0)), 5),
        }
        if rays.shape[0] < 3:
            return row

        lengths = np.linalg.norm(rays, axis=1)
        var_ratio, eff_dim = _unit_pca_spectrum(rays)
        u = rays / lengths[:, None]
        cos_sim = np.clip(u @ u.T, -1.0, 1.0)
        np.fill_diagonal(cos_sim, -1.0)
        nn = np.degrees(np.arccos(cos_sim.max(axis=1)))
        slacks = -(engine.A_prime @ rays.T) if engine is not None else np.array([[0.0]])
        # Gram log-volume: slogdet(V^T V) — large/finite => vectors span their space;
        # -inf (sign 0) => rank-deficient (clustered on a lower-dim plane).
        gram_sign, gram_logdet = np.linalg.slogdet(rays.T @ rays)
        gram_val = float(gram_logdet) if gram_sign > 0 else float("-inf")

        row.update({
            "unit_pca_pc1": round(float(var_ratio[0]), 4),
            "unit_pca_effdim90": eff_dim,
            "gram_logdet": round(gram_val, 3) if np.isfinite(gram_val) else "-inf",
            "norm_min": round(float(lengths.min()), 2),
            "norm_mean": round(float(np.mean(lengths)), 2),
            "norm_median": round(float(np.median(lengths)), 2),
            "norm_max": round(float(lengths.max()), 2),
            "slack_min": round(float(slacks.min()), 4),
            "slack_mean": round(float(np.mean(slacks)), 4),
            "slack_median": round(float(np.median(slacks)), 4),
            "nn_angle_min_deg": round(float(nn.min()), 2),
            "nn_angle_mean_deg": round(float(np.mean(nn)), 2),
            "nn_angle_median_deg": round(float(np.median(nn)), 2),
        })
        return row

    def export_csv(self, path, scenario_names=None):
        """Write a one-row-per-scenario diagnostic summary CSV (shareable).

        :param path: output CSV file path.
        :param scenario_names: scenarios to include; defaults to all stored runs.
        """
        names = scenario_names if scenario_names is not None else list(self.results.keys())
        rows = [self.summary_row(n) for n in names]
        fields = [
            "scenario", "d_orig", "d_flat", "yield", "time_s", "accept_rate",
            "unit_pca_pc1", "unit_pca_effdim90", "gram_logdet",
            "norm_min", "norm_mean", "norm_median", "norm_max",
            "slack_min", "slack_mean", "slack_median",
            "nn_angle_min_deg", "nn_angle_mean_deg", "nn_angle_median_deg",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        print(f"Wrote diagnostic summary CSV: {path}")


#: Engines selectable by short name from ``run_*`` helpers and the CLI.
ENGINES = {
    "discrete": DiscreteMCMCSampler,
    "pt": ParallelTemperingSampler,
    "raw": RawSpaceMCMCSampler,
}


def _resolve_engine(engine):
    """Resolve an engine short-name or class to a sampler class.

    :param engine: an :data:`ENGINES` key (e.g. ``"pt"``) or a sampler class.
    :return: the sampler class to instantiate.
    """
    return ENGINES[engine] if isinstance(engine, str) else engine


def run_diagnostic_dashboard(target_quota=10_000, scenario_name="15D_Pancake", engine="discrete"):
    """Run one scenario through a chosen sampler and render its 8-pane dashboard.

    :param target_quota: number of primitive vectors to request.
    :param scenario_name: archetype key (e.g. ``"10D_Fat_Baseline"``).
    :param engine: ``"discrete"`` / ``"pt"`` (or a sampler class) to evaluate.
    """
    cls = _resolve_engine(engine)
    print("=" * 60)
    print(f" DIAGNOSTIC HARNESS — {cls.__name__}")
    print("=" * 60)
    harness = DiscreteTestHarness(engine_class=cls)
    data = harness.run_one(scenario_name, target_quota)
    if "error" not in data and len(data["rays"]) > 0:
        harness.render_dashboard(scenario_name)


def run_gauntlet_csv(out_path="discrete_sampler_diagnostics.csv", target_quota=2_000,
                     engine="discrete"):
    """Run all archetypes, render dashboards, and write a shareable summary CSV.

    :param out_path: path for the summary CSV.
    :param target_quota: quota requested per scenario.
    :param engine: ``"discrete"`` / ``"pt"`` (or a sampler class) to evaluate.
    """
    cls = _resolve_engine(engine)
    harness = DiscreteTestHarness(engine_class=cls)
    scenarios = ["10D_Fat_Baseline", "15D_Needle", "15D_Pancake", "7D_Real_World_Degenerate"]
    for name in scenarios:
        data = harness.run_one(name, target_quota)
        if "error" not in data and len(data["rays"]) > 0:
            harness.render_dashboard(name)
    harness.export_csv(out_path, scenarios)


def compare_engines(target_quota=2_000, scenario="15D_Needle"):
    """Run both engines on one scenario and print a side-by-side comparison.

    :param target_quota: quota requested per engine.
    :param scenario: archetype key to compare on (the hard 15D needle by default).
    """
    for label, cls in (("single-chain", DiscreteMCMCSampler),
                       ("parallel-tempering", ParallelTemperingSampler)):
        harness = DiscreteTestHarness(engine_class=cls)
        data = harness.run_one(scenario, target_quota)
        eng = data["engine"]
        print(f"  [{label}] yield={data['yield']} accept={eng.last_accept_rate * 100:.2f}%")


if __name__ == "__main__":
    import sys

    # Usage: python -m tests.discrete_testing_tool [discrete|pt]
    # engine_name = sys.argv[1] if len(sys.argv) > 1 else "discrete"
    engine_name = 'discrete'
    out = f"{engine_name}_sampler_diagnostics.csv"
    run_gauntlet_csv(out_path=out, engine=engine_name)
