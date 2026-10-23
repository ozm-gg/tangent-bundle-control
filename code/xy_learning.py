"""Train autonomous position and tangent-bundle policies on the XY cost.

The optimizer sees energy and terminal residuals only, never reference samples.
The reference is used exclusively for independent post-training validation.
"""

from pathlib import Path
import argparse
import json
import time
import platform
import scipy
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from planar_steering import (
    Q0,
    V0,
    G,
    W,
    KAPPA,
    OPTIMUM,
    BARRIER,
    CROSSING_TIMES,
    reference,
)

ROOT = Path(__file__).resolve().parents[1]


class Network:
    def __init__(self, kind, width=8):
        self.kind, self.width = kind, width
        self.dim = 2 if kind == "position" else 4
        self.size = width * self.dim + width + 2 * width + (2 if kind == "lift" else 0)

    def unpack(self, p):
        p = np.atleast_2d(p)
        b = len(p)
        h = self.width
        d = self.dim
        if self.kind == "position":
            p = np.concatenate((p, np.zeros((b, 2), dtype=p.dtype)), axis=1)
        return (
            p[:, : h * d].reshape(b, h, d),
            p[:, h * d : h * d + h],
            p[:, h * d + h : -2].reshape(b, 2, h),
            p[:, -2:],
        )

    def initial(self, seed):
        rng = np.random.default_rng(seed)
        return np.r_[
            rng.normal(0, 0.08, self.width * self.dim),
            rng.normal(0, 0.08, self.width),
            rng.normal(0, 0.5, 2 * self.width),
            np.zeros(2 if self.kind == "lift" else 0),
        ]

    def field(self, weights, x):
        a, b, c, d = weights
        z = np.tanh(np.einsum("bhd,bd->bh", a, x) + b)
        return np.einsum("boh,bh->bo", c, z) + d, z

    def rhs(self, weights, y):
        if self.kind == "lift":
            acc, _ = self.field(weights, y)
            return np.concatenate((y[:, 2:], acc), axis=1), acc
        raw, z = self.field(weights, y)
        base, _ = self.field(weights, np.broadcast_to(Q0, y.shape))
        vel = V0 + raw - base
        a, b, c, d = weights
        acc = np.einsum("boh,bh->bo", c, (1 - z * z) * np.einsum("bhd,bd->bh", a, vel))
        return vel, acc

    def residual(self, p, steps):
        p = np.atleast_2d(p)
        weights = self.unpack(p)
        dtype = p.dtype
        y = (
            np.broadcast_to(
                np.r_[Q0, V0] if self.kind == "lift" else Q0,
                (len(p), 4 if self.kind == "lift" else 2),
            )
            .astype(dtype)
            .copy()
        )
        h = 1 / steps
        aa = []
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(steps + 1):
                k1, acc = self.rhs(weights, y)
                aa.append(acc)
                if i == steps:
                    break
                k2, _ = self.rhs(weights, y + h * k1 / 2)
                k3, _ = self.rhs(weights, y + h * k2 / 2)
                k4, _ = self.rhs(weights, y + h * k3)
                y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            velocity = y[:, 2:] if self.kind == "lift" else self.rhs(weights, y)[0]
            q = y[:, :2]
            factors = np.ones(steps + 1)
            factors[1:-1:2] = 4
            factors[2:-1:2] = 2
            energy = np.stack(aa, axis=1) * np.sqrt(factors[None, :, None] * h / 3)
            residual = np.concatenate(
                (
                    energy.reshape(len(p), -1),
                    np.sqrt(KAPPA) * (q - G),
                    np.sqrt(KAPPA) * (velocity - W),
                ),
                axis=1,
            )
        return np.nan_to_num(residual, nan=1e100, posinf=1e100, neginf=-1e100)

    def jacobian(self, p, steps):
        perturb = np.broadcast_to(p, (len(p), len(p))).astype(complex).copy()
        perturb[np.arange(len(p)), np.arange(len(p))] += 1e-25j
        return self.residual(perturb, steps).imag.T / 1e-25

    def verify(self, p, rtol=2e-11, atol=2e-12):
        weights = self.unpack(p)
        dim = 4 if self.kind == "lift" else 2

        def rhs(t, y):
            field, acc = self.rhs(weights, y[:dim][None, :])
            acc = acc[0]
            error = acc - reference(t)[2]
            return np.r_[field[0], 0.5 * acc @ acc, error @ error]

        result = solve_ivp(
            rhs,
            (0, 1),
            np.r_[Q0, V0, 0.0, 0.0] if dim == 4 else np.r_[Q0, 0.0, 0.0],
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=1 / 128,
            t_eval=np.linspace(0, 1, 501),
        )
        if not result.success:
            raise RuntimeError(result.message)
        states = result.y[:dim].T
        field, acc = self.rhs(
            tuple(np.broadcast_to(a, (len(states),) + a.shape[1:]) for a in weights),
            states,
        )
        q = states[:, :2]
        v = states[:, 2:] if dim == 4 else field
        qr, vr, _ = reference(1.0)
        gap = 0.5 * result.y[dim + 1, -1] + KAPPA / 2 * (
            np.sum((q[-1] - qr) ** 2) + np.sum((v[-1] - vr) ** 2)
        )
        cost = result.y[dim, -1] + KAPPA / 2 * (
            np.sum((q[-1] - G) ** 2) + np.sum((v[-1] - W) ** 2)
        )
        return dict(
            cost=float(cost),
            gap=float(gap),
            identity_error=float(abs(cost - OPTIMUM - gap)),
            nfev=result.nfev,
            max_weight=float(abs(p).max()),
            below_barrier=bool(gap < BARRIER),
        ), dict(t=result.t, q=q, v=v, u=acc)


def train(kind, seed, steps, budget, width, output, refine_budget=300):
    model = Network(kind, width)
    initial = model.initial(seed)
    history = []
    checkpoints = []
    checkpoint_steps = steps
    calls = 0
    best_cost = float("inf")
    best_p = initial.copy()
    start = time.monotonic()

    def fun(p):
        nonlocal calls, best_cost, best_p
        r = model.residual(p, steps)[0]
        calls += 1
        current_cost = float(np.sum(r * r) / 2)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        if steps == checkpoint_steps and (calls == 1 or calls % 50 == 0):
            checkpoints.append((calls, best_p.copy()))
        history.append(
            dict(
                evaluation=calls,
                steps=steps,
                seconds=time.monotonic() - start,
                cost=current_cost,
            )
        )
        return r

    result = least_squares(
        fun,
        initial,
        jac=lambda p: model.jacobian(p, steps),
        max_nfev=budget,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-10,
        x_scale="jac",
        tr_solver="exact",
        bounds=(-20.0, 20.0),
    )
    phases = [
        dict(
            steps=steps,
            budget=budget,
            nfev=result.nfev,
            njev=result.njev,
            status=int(result.status),
            message=result.message,
        )
    ]
    if refine_budget:
        steps *= 2
        best_cost = float("inf")
        best_p = result.x.copy()
        result = least_squares(
            fun,
            result.x,
            jac=lambda p: model.jacobian(p, steps),
            max_nfev=refine_budget,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-10,
            x_scale="jac",
            tr_solver="exact",
            bounds=(-20.0, 20.0),
        )
        phases.append(
            dict(
                steps=steps,
                budget=refine_budget,
                nfev=result.nfev,
                njev=result.njev,
                status=int(result.status),
                message=result.message,
            )
        )
    elapsed = time.monotonic() - start
    metrics, path = model.verify(result.x)
    tight, _ = model.verify(result.x, rtol=2e-12, atol=2e-13)
    fine = model.residual(result.x, steps * 2)[0]
    # One central finite-difference directional check independent of complex step.
    direction = np.random.default_rng(991).normal(size=model.size)
    direction /= np.linalg.norm(direction)
    fd = (
        model.residual(result.x + 1e-6 * direction, steps)[0]
        - model.residual(result.x - 1e-6 * direction, steps)[0]
    ) / 2e-6
    analytic = np.sum(model.jacobian(result.x, steps) * direction[None, :], axis=1)
    metrics.update(
        phases=phases,
        parameter_count=model.size,
        training_vs_dop853=float(abs(result.cost - metrics["cost"])),
        all_finite=bool(
            np.isfinite(result.x).all()
            and all(np.isfinite(a).all() for a in path.values())
        ),
        initial_velocity_error=float(np.linalg.norm(path["v"][0] - V0)),
        max_position_error=float(
            np.max(np.linalg.norm(path["q"] - reference(path["t"])[0], axis=1))
        ),
        max_velocity_error=float(
            np.max(np.linalg.norm(path["v"] - reference(path["t"])[1], axis=1))
        ),
        kind=kind,
        seed=seed,
        width=width,
        steps=steps,
        budget=budget + refine_budget,
        nfev=calls,
        njev=sum(a["njev"] for a in phases),
        status=int(result.status),
        message=result.message,
        optimality=float(result.optimality),
        seconds=elapsed,
        training_cost=float(result.cost),
        fine_grid_cost=float(np.sum(fine * fine) / 2),
        grid_discrepancy=float(abs(result.cost - np.sum(fine * fine) / 2)),
        integration_refinement=float(abs(metrics["gap"] - tight["gap"])),
        directional_jacobian_relative_error=float(
            np.linalg.norm(fd - analytic) / max(1, np.linalg.norm(analytic))
        ),
    )
    dest = output / "results" / "xy_learning"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / f"{kind}_{seed}.npz", weights=result.x, initial=initial, **path
    )
    (dest / f"{kind}_{seed}.json").write_text(
        json.dumps(dict(metrics=metrics, history=history), indent=2) + "\n"
    )
    checkpoints.append((calls, result.x.copy()))
    verified_history = []
    for count, weights in checkpoints:
        row, _ = model.verify(weights)
        verified_history.append(
            dict(evaluation=count, gap=row["gap"], cost=row["cost"])
        )
    (dest / f"{kind}_{seed}.json").write_text(
        json.dumps(
            dict(metrics=metrics, history=history, verified_history=verified_history),
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        dest / f"{kind}_{seed}_checkpoints.npz",
        evaluations=np.array([x[0] for x in checkpoints]),
        weights=np.stack([x[1] for x in checkpoints]),
    )
    print(json.dumps(metrics), flush=True)
    return metrics


def summarize(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = output / "results" / "xy_learning"
    records = [
        json.loads((destination / f"{kind}_{seed}.json").read_text())
        for kind in ("position", "lift")
        for seed in (11, 22, 33)
    ]
    rows = [item["metrics"] for item in records]
    record = json.loads((destination / "summary.json").read_text())
    record.update(
        barrier=BARRIER,
        optimum=OPTIMUM,
        task=dict(
            initial_position=Q0.tolist(),
            initial_velocity=V0.tolist(),
            target_position=G.tolist(),
            target_velocity=W.tolist(),
            kappa=KAPPA,
            crossing_times=CROSSING_TIMES.tolist(),
            optimal_terminal_position=reference(1.0)[0].tolist(),
            optimal_terminal_velocity=reference(1.0)[1].tolist(),
        ),
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
        ),
        protocol=dict(
            width=rows[0]["width"],
            seeds=[11, 22, 33],
            parameter_box=[-20, 20],
            initial_W1_and_b1_standard_deviation=0.08,
            initial_W2_standard_deviation=0.5,
            initial_lift_output_bias=0,
            precision="float64",
            integration="RK4",
            quadrature="composite Simpson",
            stages=[
                dict(steps=a["steps"], max_nfev=a["budget"]) for a in rows[0]["phases"]
            ],
            optimizer="scipy least_squares, trf, exact dense trust-region solve",
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-10,
            jacobian="complex step 1e-25",
            verification=(
                "DOP853: rtol 2e-11, atol 2e-12, max_step 1/128; repeat at rtol 2e-12,"
                " atol 2e-13"
            ),
            checkpoint_selection=(
                "Initial and every 50 evaluations in first phase: best discrete"
                " objective seen so far; final returned model after second phase. All"
                " selected checkpoints independently reintegrated."
            ),
        ),
    )
    (destination / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    checks = []
    for row in rows:
        check = dict(
            all_finite=row["all_finite"],
            initial_velocity_exact=row["initial_velocity_error"] == 0.0,
            cost_identity=row["identity_error"] < 1e-8,
            independent_refinement=row["integration_refinement"] < 1e-8,
            parameter_box=row["max_weight"] <= 20.0 + 1e-10,
            directional_derivative=row["directional_jacobian_relative_error"] < 1e-5,
            training_vs_independent=row["training_vs_dop853"]
            < max(1e-7, 1e-3 * row["gap"]),
        )
        model = Network(row["kind"], row["width"])
        with np.load(destination / f"{row['kind']}_{row['seed']}.npz") as saved:
            fresh, path = model.verify(saved["weights"])
            check["saved_weights_reintegrate"] = abs(fresh["gap"] - row["gap"]) < 1e-12
            check["saved_paths_match"] = all(
                np.allclose(path[key], saved[key], rtol=1e-9, atol=1e-10)
                for key in path
            )
            check["seed_initialization_matches"] = np.array_equal(
                saved["initial"], model.initial(row["seed"])
            )
        checks.append(dict(kind=row["kind"], seed=row["seed"], checks=check))
    validation = output / "validation"
    validation.mkdir(exist_ok=True)
    report = dict(
        passed=all(all(r["checks"].values()) for r in checks),
        checks=checks,
        scope=(
            "Numerical consistency checks; not a convergence or statistical"
            " generalization guarantee."
        ),
        all_lift_runs_below_barrier=all(
            r["gap"] < BARRIER for r in rows if r["kind"] == "lift"
        ),
        all_position_runs_above_barrier=all(
            r["gap"] >= BARRIER for r in rows if r["kind"] == "position"
        ),
    )
    (validation / "xy_learning.json").write_text(json.dumps(report, indent=2) + "\n")
    colors = dict(position="#c56820", lift="#2774a8")
    labels = dict(position="Position only", lift="Tangent lift")
    plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight"})
    fig, (full, cost) = plt.subplots(1, 2, figsize=(7.4, 3.8), layout="constrained")
    for item in records:
        row = item["metrics"]
        kind = row["kind"]
        seed = row["seed"]
        label = labels[kind] if seed == 11 else None
        with np.load(destination / f"{kind}_{seed}.npz") as path:
            full.plot(
                path["q"][:, 0],
                path["q"][:, 1],
                color=colors[kind],
                alpha=0.65,
                lw=1.3,
                label=label,
            )
        history = item["verified_history"]
        cost.semilogy(
            [a["evaluation"] for a in history],
            [a["gap"] for a in history],
            color=colors[kind],
            alpha=0.65,
            lw=1.2,
            label=label,
        )
    t = np.linspace(0, 1, 1001)
    q, v, _ = reference(t)
    full.plot(q[:, 0], q[:, 1], "k--", lw=1.3, label="Analytical optimum", zorder=5)
    full.plot(*Q0, "ks", ms=4)
    full.plot(*q[-1], "k^", ms=4)
    full.annotate("$q_0=g$", Q0, xytext=(-14, -14), textcoords="offset points")
    full.annotate("$q_*(1)$", q[-1], xytext=(5, 4), textcoords="offset points")
    full.set(xlabel="$q_1$", ylabel="$q_2$", title="(a) Position trajectories")
    full.margins(0.15)
    full.set_aspect("equal", adjustable="box")
    full.legend(loc="lower left", fontsize=8, frameon=False)
    cost.axhline(BARRIER, color="#a52b3b", ls="--", lw=1, label="Proved barrier")
    cost.legend(loc="upper right", fontsize=8, frameon=False)
    cost.grid(alpha=0.15, which="major")
    cost.set(
        xlabel="Objective evaluations",
        ylabel="$J_X-J_{X,*}$",
        title="(b) Training histories",
    )
    target = output / "figures"
    target.mkdir(exist_ok=True)
    fig.savefig(target / "xy_learning.pdf")
    fig.savefig(target / "xy_learning.png", dpi=180)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--kind", choices=["position", "lift", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--refine-budget", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    rows = []
    if args.steps < 2 or args.steps % 2:
        parser.error("--steps must be a positive even integer (Simpson quadrature)")
    if args.summarize_only:
        print(json.dumps(summarize(args.output), indent=2))
        return
    for kind in ["position", "lift"] if args.kind == "both" else [args.kind]:
        for seed in args.seeds:
            rows.append(
                train(
                    kind,
                    seed,
                    args.steps,
                    args.budget,
                    args.width,
                    args.output,
                    args.refine_budget,
                )
            )
    dest = args.output / "results" / "xy_learning"
    (dest / "summary.json").write_text(
        json.dumps(
            dict(
                training_objective=(
                    "Original energy plus terminal penalties; reference not used in"
                    " training."
                ),
                optimizer=(
                    "scipy least_squares, exact dense trust-region subproblem;"
                    " complex-step Jacobian"
                ),
                optimum=OPTIMUM,
                barrier=BARRIER,
                runs=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    if args.kind == "both" and args.seeds == [11, 22, 33]:
        summarize(args.output)


if __name__ == "__main__":
    main()
