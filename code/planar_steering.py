"""Analytic soft-terminal planar return task used by the manuscript."""

import numpy as np

KAPPA = 50.0
Q0 = np.array([0.0, 0.0])
V0 = np.array([1.0, 0.0])
G = np.array([0.0, 0.0])
W = np.array([0.0, 1.0])
QUADRATIC, CUBIC = np.linalg.solve(
    np.array([[2 * KAPPA + 2, 3 * KAPPA + 6], [KAPPA, KAPPA - 6]]),
    np.stack([KAPPA * (W - V0), KAPPA * (G - Q0 - V0)]),
)


def reference(t):
    t = np.asarray(t)[..., None]
    return (
        Q0 + V0 * t + QUADRATIC * t * t + CUBIC * t**3,
        V0 + 2 * QUADRATIC * t + 3 * CUBIC * t * t,
        2 * QUADRATIC + 6 * CUBIC * t,
    )


OPTIMUM = 2725 / 828
BARRIER = 1 / 20000
CROSSING_TIMES = np.array([0.5 - np.sqrt(13) / 10, 0.5 + np.sqrt(13) / 10])


def affine_acceleration(v):
    v = np.asarray(v)
    tau = (25 - 25 * v[..., 0] + 26 * v[..., 1]) / 50
    return 2 * QUADRATIC + 6 * CUBIC * tau[..., None]


def verify():
    """Check proof arithmetic exactly and reintegrate the affine realization."""
    from fractions import Fraction as F
    from scipy.integrate import solve_ivp

    b = (F(-350, 207), F(-275, 414))
    a = (F(325, 414), F(625, 828))
    v = (F(1), F(0))
    g = (F(0), F(0))
    w = (F(0), F(1))
    K = F(50)
    L = ((F(-1), F(26, 25)), (F(-22, 25), F(56, 25)))
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))
    q1 = tuple(v[i] + b[i] + a[i] for i in range(2))
    v1 = tuple(v[i] + 2 * b[i] + 3 * a[i] for i in range(2))
    u1 = tuple(2 * b[i] + 6 * a[i] for i in range(2))
    det = b[0] * a[1] - b[1] * a[0]
    p = F(1)
    r = F(22, 25)
    disc = 4 * r - 3 * p * p
    product = p * p - r
    e = F(1, 25)
    rho = F(1, 100)
    checks = {
        "terminal_velocity_condition": all(
            u1[i] + K * (v1[i] - w[i]) == 0 for i in range(2)
        ),
        "terminal_position_condition": all(
            -6 * a[i] + K * (q1[i] - g[i]) == 0 for i in range(2)
        ),
        "exact_cost": (
            2 * dot(b, b)
            + 6 * dot(b, a)
            + 6 * dot(a, a)
            + K / 2 * (sum((q1[i] - g[i]) ** 2 + (v1[i] - w[i]) ** 2 for i in range(2)))
            == F(2725, 828)
        ),
        "matrix_inverse": (
            [[dot(row, col) for col in [b, a]] for row in L] == [[1, 0], [0, 1]]
        ),
        "first_row_norm": dot(L[0], L[0]) < 9,
        "second_row_norm": dot(L[1], L[1]) < 9,
        "crossing_equations": all(v[i] + b[i] * p + a[i] * r == 0 for i in range(2)),
        "crossing_point": (
            tuple(-product * (b[i] + a[i] * p) for i in range(2))
            == (F(5, 46), F(-1, 92))
        ),
        "transverse_determinant_squared": det**2 * disc**3 == F(65**2 * 13, 828**2),
        "clock_identity": (
            dot((-25, 26), v) == -25
            and dot((-25, 26), b) == 25
            and dot((-25, 26), a) == 0
        ),
        "rectangle_time_separation": 4 * (r - e) - 3 * (p + e) ** 2 > F(9, 100),
        "rectangle_positive_times": r + e < (p - e) ** 2,
        "rectangle_times_below_one": p + e < 2 and r + e < (p - e) ** 2 - (p - e) + 1,
        "face_signs_persist": 3 * rho < e,
        "different_velocities_persist": 6 * rho < F(3, 5),
        "cost_barrier": rho * rho / 2 == F(1, 20000),
    }
    # At K=24 the two times merge; every finite K>24 has an interior crossing.
    cases = [
        (F(1), False),
        (F(24), False),
        (F(25), True),
        (F(50), True),
        (F(1000), True),
    ]
    for penalty, expected in cases:
        disc = 1 - 24 / penalty
        checks[f"interior_crossing_K{penalty}"] = bool(0 < disc < 1) == expected
    assert all(checks.values()), checks

    def rhs(t, y):
        acc = affine_acceleration(y[2:4])
        return np.r_[y[2:4], acc, 0.5 * acc @ acc]

    times = np.linspace(0, 1, 501)
    sol = solve_ivp(
        rhs,
        (0, 1),
        np.r_[Q0, V0, 0.0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
        max_step=1 / 64,
        t_eval=times,
    )
    qr, vr, _ = reference(times)
    error = max(np.max(abs(sol.y[:2].T - qr)), np.max(abs(sol.y[2:4].T - vr)))
    cost = sol.y[4, -1] + KAPPA / 2 * (
        np.sum((sol.y[:2, -1] - G) ** 2) + np.sum((sol.y[2:4, -1] - W) ** 2)
    )
    numeric = dict(
        integration_success=bool(sol.success),
        reference_match=bool(error < 1e-10),
        optimal_cost=bool(abs(cost - OPTIMUM) < 1e-10),
    )
    assert all(numeric.values()), numeric
    return dict(
        passed=True,
        exact_checks=checks,
        numerical_checks=numeric,
        max_affine_state_error=float(error),
        affine_cost_error=float(abs(cost - OPTIMUM)),
        barrier=BARRIER,
        scope=(
            "Exact proof arithmetic and affine-reference integration, not a proof of"
            " optimizer convergence. Boundary cases test interior crossing only."
        ),
    )


if __name__ == "__main__":
    import json
    from pathlib import Path

    report = verify()
    destination = (
        Path(__file__).resolve().parents[1] / "validation/planar_steering.json"
    )
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
