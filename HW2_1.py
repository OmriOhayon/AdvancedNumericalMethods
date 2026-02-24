
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Numerical Methods HW2
# 2D heat equation on a square plate:
#     dT/dt = alpha * (d2T/dx2 + d2T/dy2) + q
# Solve with Crank–Nicolson implemented via ADI + Thomas algorithm (tridiagonal).
#
# Also includes a full Von Neumann stability analysis for the proposed 1D scheme in Q2.
# =========================


def thomas_solve(a, b, c, d):
    """Solve a tridiagonal system Ax=d via Thomas algorithm.

    The matrix A has:
        sub-diagonal a (len n-1),
        diagonal      b (len n),
        super-diag    c (len n-1).

    Args:
        a (np.ndarray): Sub-diagonal, shape (n-1,).
        b (np.ndarray): Diagonal, shape (n,).
        c (np.ndarray): Super-diagonal, shape (n-1,).
        d (np.ndarray): RHS, shape (n,).

    Returns:
        np.ndarray: Solution x, shape (n,).

    Notes:
        - No pivoting. This is fine here because our matrices are diagonally dominant.
    """
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()

    n = b.size
    if d.size != n:
        raise ValueError("d must have same length as b")
    if a.size != n - 1 or c.size != n - 1:
        raise ValueError("a and c must have length n-1")

    # Forward elimination
    for i in range(1, n):
        if b[i - 1] == 0.0:
            raise ZeroDivisionError("Zero pivot encountered in Thomas solve")
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    # Back substitution
    x = np.zeros(n, dtype=float)
    if b[-1] == 0.0:
        raise ZeroDivisionError("Zero pivot encountered in Thomas solve")
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        if b[i] == 0.0:
            raise ZeroDivisionError("Zero pivot encountered in Thomas solve")
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def apply_dirichlet_bc(T, bc):
    """Apply Dirichlet boundary conditions in-place where specified.

    Args:
        T (np.ndarray): Temperature field (Ny, Nx).
        bc (dict): Boundary condition dict (see make_bc()).
    """
    if bc["left"]["type"] == "dirichlet":
        T[:, 0] = bc["left"]["value"]
    if bc["right"]["type"] == "dirichlet":
        T[:, -1] = bc["right"]["value"]
    if bc["bottom"]["type"] == "dirichlet":
        T[0, :] = bc["bottom"]["value"]
    if bc["top"]["type"] == "dirichlet":
        T[-1, :] = bc["top"]["value"]


def delta_xx(T, bc):
    """Compute the x-second-difference operator (no /dx^2 factor).

    Handles:
      - interior: T[i+1]-2T[i]+T[i-1]
      - Neumann0 at boundary: uses a ghost node with 2nd order approximation:
            dT/dx = 0 at right => T_ghost = T_{Nx-2}
            => delta_xx at boundary node i=Nx-1 equals 2*(T_{Nx-2}-T_{Nx-1})

    Args:
        T (np.ndarray): (Ny, Nx) temperature.
        bc (dict): boundary condition dict.

    Returns:
        np.ndarray: (Ny, Nx) delta_xx(T).
    """
    Ny, Nx = T.shape
    dxx = np.zeros_like(T, dtype=float)

    # Interior (i=1..Nx-2)
    dxx[:, 1:-1] = T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]

    # Left boundary i=0
    if bc["left"]["type"] == "neumann0":
        # Ghost: T[-1] mirrored: T_ghost = T[1]
        dxx[:, 0] = 2.0 * (T[:, 1] - T[:, 0])
    else:
        # Dirichlet nodes are treated as fixed; we keep dxx=0 there for operator use.
        dxx[:, 0] = 0.0

    # Right boundary i=Nx-1
    if bc["right"]["type"] == "neumann0":
        dxx[:, -1] = 2.0 * (T[:, -2] - T[:, -1])
    else:
        dxx[:, -1] = 0.0

    return dxx


def delta_yy(T, bc):
    """Compute the y-second-difference operator (no /dy^2 factor).

    Uses the same Neumann0 ghost-node trick as delta_xx, but in y.

    Args:
        T (np.ndarray): (Ny, Nx) temperature.
        bc (dict): boundary condition dict.

    Returns:
        np.ndarray: (Ny, Nx) delta_yy(T).
    """
    Ny, Nx = T.shape
    dyy = np.zeros_like(T, dtype=float)

    # Interior (j=1..Ny-2)
    dyy[1:-1, :] = T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]

    # Bottom boundary j=0
    if bc["bottom"]["type"] == "neumann0":
        dyy[0, :] = 2.0 * (T[1, :] - T[0, :])
    else:
        dyy[0, :] = 0.0

    # Top boundary j=Ny-1
    if bc["top"]["type"] == "neumann0":
        dyy[-1, :] = 2.0 * (T[-2, :] - T[-1, :])
    else:
        dyy[-1, :] = 0.0

    return dyy


def make_bc(left=("dirichlet", 60.0),
            right=("dirichlet", 20.0),
            bottom=("dirichlet", 60.0),
            top=("dirichlet", 20.0)):
    """Helper to build a boundary-condition dictionary.

    Boundary types:
      - ("dirichlet", value)
      - ("neumann0", None)  # insulated boundary, dT/dn = 0

    Returns:
        dict: bc structure used by the solver.
    """
    def _one(spec):
        typ = spec[0].lower()
        if typ not in ("dirichlet", "neumann0"):
            raise ValueError(f"Unknown BC type: {spec[0]}")
        return {"type": typ, "value": None if typ == "neumann0" else float(spec[1])}

    return {
        "left": _one(left),
        "right": _one(right),
        "bottom": _one(bottom),
        "top": _one(top),
    }


def adi_cn_step(Tn, alpha, q, dt, dx, dy, bc):
    """Advance one time step using CN via Peaceman–Rachford ADI.

    Split CN into two implicit half-steps:
      (I - rx/2 * Dxx) T*     = (I + ry/2 * Dyy) Tn + dt/2 * q
      (I - ry/2 * Dyy) Tn1    = (I + rx/2 * Dxx) T* + dt/2 * q

    where:
      rx = alpha*dt/dx^2, ry = alpha*dt/dy^2,
      Dxx, Dyy are the second-difference operators (including BC handling).

    Args:
        Tn (np.ndarray): Current field (Ny, Nx).
        alpha (float): Thermal diffusivity.
        q (float): Volumetric source term (treated constant here).
        dt (float): Time step.
        dx (float): Grid step in x.
        dy (float): Grid step in y.
        bc (dict): Boundary conditions.

    Returns:
        np.ndarray: Updated field T^{n+1} (Ny, Nx).
    """
    Tn = np.asarray(Tn, dtype=float)
    Ny, Nx = Tn.shape

    rx = alpha * dt / (dx * dx)
    ry = alpha * dt / (dy * dy)

    # --- Half step 1: implicit in x ---
    rhs1 = Tn + 0.5 * ry * delta_yy(Tn, bc) + 0.5 * dt * q

    Tstar = rhs1.copy()
    apply_dirichlet_bc(Tstar, bc)

    # Solve tridiagonal along x for each row j
    for j in range(Ny):
        # Determine unknown i-range for this row
        i0 = 0
        i1 = Nx - 1
        if bc["left"]["type"] == "dirichlet":
            i0 = 1
        if bc["right"]["type"] == "dirichlet":
            i1 = Nx - 2

        if i1 < i0:
            continue  # nothing to solve

        n = i1 - i0 + 1
        a = np.zeros(n - 1, dtype=float)
        b = np.zeros(n, dtype=float)
        c = np.zeros(n - 1, dtype=float)
        d = np.zeros(n, dtype=float)

        for k, i in enumerate(range(i0, i1 + 1)):
            # Build (I - rx/2 Dxx) row for node (j,i)
            if 0 < i < Nx - 1:
                a_k = -0.5 * rx
                b_k = 1.0 + rx
                c_k = -0.5 * rx
            elif i == 0:
                # left boundary node: only if Neumann0 (Dirichlet would be excluded)
                a_k = 0.0
                b_k = 1.0 + rx
                c_k = -rx  # from 2*(T1-T0)
            else:  # i == Nx-1
                a_k = -rx
                b_k = 1.0 + rx
                c_k = 0.0

            b[k] = b_k
            d[k] = rhs1[j, i]

            # Handle left neighbor coefficient
            if k > 0:
                a[k - 1] = a_k
            else:
                # k==0: if there is a neighbor outside unknown set, move to RHS
                if i - 1 >= 0 and (i - 1) < i0:
                    # neighbor is a Dirichlet boundary
                    d[k] -= a_k * Tstar[j, i - 1]

            # Handle right neighbor coefficient
            if k < n - 1:
                c[k] = c_k
            else:
                if i + 1 <= Nx - 1 and (i + 1) > i1:
                    d[k] -= c_k * Tstar[j, i + 1]

        sol = thomas_solve(a, b, c, d)
        Tstar[j, i0:i1 + 1] = sol

    apply_dirichlet_bc(Tstar, bc)

    # --- Half step 2: implicit in y ---
    rhs2 = Tstar + 0.5 * rx * delta_xx(Tstar, bc) + 0.5 * dt * q

    Tn1 = rhs2.copy()
    apply_dirichlet_bc(Tn1, bc)

    # Solve tridiagonal along y for each column i
    for i in range(Nx):
        j0 = 0
        j1 = Ny - 1
        if bc["bottom"]["type"] == "dirichlet":
            j0 = 1
        if bc["top"]["type"] == "dirichlet":
            j1 = Ny - 2

        if j1 < j0:
            continue

        n = j1 - j0 + 1
        a = np.zeros(n - 1, dtype=float)
        b = np.zeros(n, dtype=float)
        c = np.zeros(n - 1, dtype=float)
        d = np.zeros(n, dtype=float)

        for k, j in enumerate(range(j0, j1 + 1)):
            if 0 < j < Ny - 1:
                a_k = -0.5 * ry
                b_k = 1.0 + ry
                c_k = -0.5 * ry
            elif j == 0:
                a_k = 0.0
                b_k = 1.0 + ry
                c_k = -ry
            else:  # j == Ny-1
                a_k = -ry
                b_k = 1.0 + ry
                c_k = 0.0

            b[k] = b_k
            d[k] = rhs2[j, i]

            if k > 0:
                a[k - 1] = a_k
            else:
                if j - 1 >= 0 and (j - 1) < j0:
                    d[k] -= a_k * Tn1[j - 1, i]

            if k < n - 1:
                c[k] = c_k
            else:
                if j + 1 <= Ny - 1 and (j + 1) > j1:
                    d[k] -= c_k * Tn1[j + 1, i]

        sol = thomas_solve(a, b, c, d)
        Tn1[j0:j1 + 1, i] = sol

    apply_dirichlet_bc(Tn1, bc)
    return Tn1


def run_plate(alpha=4e-6, L=0.3, q=0.0,
              Nx=30, Ny=30, dt=1.0,
              t_final=5000.0,
              T_init=60.0,
              bc=None,
              steady_tol=1e-6,
              steady_window=20,
              snapshot_times=None,
              verbose=True,
              progress_updates=20):
    """Run the transient 2D plate simulation.

    Args:
        alpha (float): Thermal diffusivity.
        L (float): Plate side length (meters). The doc shows L=0.3m.
        q (float): Volumetric source term (doc shows q=0 in one place; keep parametric).
        Nx (int): Grid points in x.
        Ny (int): Grid points in y.
        dt (float): Time step (seconds).
        t_final (float): Max simulation time.
        T_init (float): Initial temperature (doc includes T(t=0,x,y)=60).
        bc (dict): Boundary conditions. If None, uses default hot-left/bottom and cold-right/top.
        steady_tol (float): Steady criterion on max |T^{n+1}-T^n|.
        steady_window (int): Require criterion for this many consecutive steps to avoid false triggers.
        snapshot_times (list[float]): Times at which to store 2D snapshots.

    Returns:
        dict: results with fields:
            x, y, times, center_T, snapshots, T_final, steady_time
    """
    if bc is None:
        bc = make_bc()

    if snapshot_times is None:
        snapshot_times = [0.0, 0.1 * t_final, 0.3 * t_final, 0.6 * t_final, t_final]

    x = np.linspace(0.0, L, Nx)
    y = np.linspace(0.0, L, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    T = np.full((Ny, Nx), float(T_init), dtype=float)
    apply_dirichlet_bc(T, bc)

    times = [0.0]
    center_T = [T[Ny // 2, Nx // 2]]
    snapshots = {0.0: T.copy()}

    snap_set = set([float(t) for t in snapshot_times])
    steady_hits = 0
    steady_time = None

    n_steps = int(np.ceil(t_final / dt))
    if progress_updates is None or progress_updates <= 0:
        progress_stride = n_steps + 1
    else:
        progress_stride = max(1, n_steps // int(progress_updates))
    if verbose:
        print(f"[run_plate] start: Nx={Nx}, Ny={Ny}, dt={dt:g}s, t_final={t_final:g}s, steps={n_steps}")

    for n in range(1, n_steps + 1):
        t = n * dt
        Tn1 = adi_cn_step(T, alpha=alpha, q=q, dt=dt, dx=dx, dy=dy, bc=bc)

        diff_inf = np.max(np.abs(Tn1 - T))
        if diff_inf < steady_tol:
            steady_hits += 1
            if steady_hits >= steady_window and steady_time is None:
                steady_time = t
                if verbose:
                    print(f"[run_plate] steady criterion met at t={t:.6g}s (|ΔT|_inf={diff_inf:.3e})")
        else:
            steady_hits = 0

        T = Tn1

        times.append(t)
        center_T.append(T[Ny // 2, Nx // 2])

        # Snapshots: store at the closest step to requested times
        for ts in list(snap_set):
            if abs(t - ts) <= 0.5 * dt:
                snapshots[ts] = T.copy()
                snap_set.remove(ts)
                if verbose:
                    print(f"[run_plate] snapshot captured at t={ts:g}s")

        if verbose and (n == 1 or n == n_steps or (n % progress_stride) == 0):
            pct = 100.0 * n / n_steps
            remaining = len(snap_set)
            print(f"[run_plate] {pct:6.2f}%  step {n}/{n_steps}  t={t:.6g}s  |ΔT|_inf={diff_inf:.3e}  remaining_snaps={remaining}")

        if steady_time is not None and t > max(snapshot_times):
            # If we've already reached steady and got requested snapshots, stop early.
            if len(snap_set) == 0:
                break
            if verbose:
                print(f"[run_plate] early stop at t={t:.6g}s (steady + snapshots done)")

    return {
        "x": x,
        "y": y,
        "times": np.asarray(times),
        "center_T": np.asarray(center_T),
        "snapshots": snapshots,
        "T_final": T,
        "steady_time": steady_time,
        "params": {
            "alpha": alpha,
            "L": L,
            "q": q,
            "Nx": Nx,
            "Ny": Ny,
            "dt": dt,
            "t_final": t_final,
            "T_init": T_init,
            "steady_tol": steady_tol,
            "steady_window": steady_window,
            "bc": bc,
        }
    }


def plot_snapshots(result, title_prefix=""):
    """Plot 2D temperature snapshots using imshow."""
    x = result["x"]
    y = result["y"]
    snaps = result["snapshots"]
    t_list = sorted(snaps.keys())

    for t in t_list:
        T = snaps[t]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(
            T,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="equal",
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{title_prefix}T(x,y) at t={t:.3g} s")
        fig.colorbar(im, ax=ax, label="Temperature [°C]")
        fig.tight_layout()


def plot_center_temperature(result, title="Center temperature vs time"):
    """Plot temperature at the plate center over time."""
    t = result["times"]
    Tc = result["center_T"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(t, Tc, linewidth=2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T_center [°C]")
    ax.set_title(title)
    ax.grid(True, alpha=0.6)

    steady_time = result.get("steady_time", None)
    if steady_time is not None:
        ax.axvline(steady_time, linestyle="--", linewidth=2, label=f"steady @ {steady_time:.3g}s")
        ax.legend(loc="best")
    fig.tight_layout()


def parameter_study():
    print("\n=== Parameter study (dt/grid sensitivity) ===")
    """Minimal param study: dt and grid resolution effects on center temperature.

    This is not a full factorial sweep; it's meant to give you clean plots and numbers
    you can discuss in the writeup.
    """
    alpha = 4e-6
    L = 0.3
    q = 0.0

    bc_dirichlet = make_bc(
        left=("dirichlet", 60.0),
        bottom=("dirichlet", 60.0),
        right=("dirichlet", 20.0),
        top=("dirichlet", 20.0),
    )

    # Fixed "reference-ish" run
    ref = run_plate(alpha=alpha, L=L, q=q, Nx=30, Ny=30, dt=0.5, t_final=6000.0,
                    T_init=60.0, bc=bc_dirichlet, steady_tol=1e-6, steady_window=20, verbose=False)

    # dt sweep
    dt_list = [2.0, 1.0, 0.5, 0.25]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for dt in dt_list:
        res = run_plate(alpha=alpha, L=L, q=q, Nx=30, Ny=30, dt=dt, t_final=6000.0,
                        T_init=60.0, bc=bc_dirichlet, steady_tol=1e-6, steady_window=20, verbose=False)
        ax.plot(res["times"], res["center_T"], linewidth=2, label=f"dt={dt:g}s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T_center [°C]")
    ax.set_title("dt sensitivity (Nx=Ny=81)")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    # Grid sweep
    grid_list = [10, 20, 30, 40, 50]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for N in grid_list:
        res = run_plate(alpha=alpha, L=L, q=q, Nx=N, Ny=N, dt=0.5, t_final=6000.0,
                        T_init=60.0, bc=bc_dirichlet, steady_tol=1e-6, steady_window=20, verbose=False)
        ax.plot(res["times"], res["center_T"], linewidth=2, label=f"N={N}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T_center [°C]")
    ax.set_title("Grid sensitivity (dt=0.5s)")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    return ref


def stability_analysis_q2():
    """Von Neumann stability analysis for Q2 scheme.

    PDE on 0<=x<=1:
        du/dt = alpha * d2u/dx2 , alpha>0

    Proposed scheme (as appears in the HW):
        (u_m^{n+1} - u_m^{n-1}) / dt = (alpha/h^2) * (u_{m+1}^n - 2u_m^n + u_{m-1}^n)

    Analysis:
        Assume mode u_m^n = G^n * exp(i*k*m*h).
        Then:
            (G - G^{-1})/dt = (alpha/h^2) * (e^{ikh} -2 + e^{-ikh})
                           = (alpha/h^2) * (-4 sin^2(kh/2))

        Let mu = 4*alpha*dt/h^2 * sin^2(kh/2) >= 0.
        Then:
            G - G^{-1} = -mu
            => G^2 + mu*G - 1 = 0
            => G = (-mu ± sqrt(mu^2 + 4)) / 2

        Since mu>=0:
          - The roots are real and have product G1*G2 = -1
          - Therefore |G1| * |G2| = 1, and one root must have |G| > 1 (unless mu=0 where |G|=1).
        This implies the scheme is unstable for any nontrivial spatial frequency (mu>0).
    """
    pass


def main():
    # --------------------------
    # Q1(a): Dirichlet on all edges (as implied by the figure labels T=60C and T=20C),
    #        with initial condition T(t=0,x,y)=60C and alpha=4e-6, L=0.3m.
    #        The doc also shows q=0 in one place, so default q=0 here.
    # --------------------------
    bc_a = make_bc(
        left=("dirichlet", 60.0),
        bottom=("dirichlet", 60.0),
        right=("dirichlet", 20.0),
        top=("dirichlet", 20.0),
    )

    res_a = run_plate(
        alpha=4e-6, L=0.3, q=0.0,
        Nx=30, Ny=30,
        dt=0.5,
        t_final=6000.0,
        T_init=60.0,
        bc=bc_a,
        steady_tol=1e-6,
        steady_window=30,
        snapshot_times=[0.0, 500.0, 1500.0, 3000.0, 6000.0],
        verbose=True,
        progress_updates=25,
    )

    plot_snapshots(res_a, title_prefix="Q1(a) ")
    plot_center_temperature(res_a, title="Q1(a) Center temperature vs time (Dirichlet top+right)")

    # --------------------------
    # Q1(b): Replace the boundary conditions on the *right* and *top* edges with insulation:
    #        dT/dx = 0 on x=L,  dT/dy = 0 on y=L.
    # --------------------------
    bc_b = make_bc(
        left=("dirichlet", 60.0),
        bottom=("dirichlet", 60.0),
        right=("neumann0", None),
        top=("neumann0", None),
    )

    res_b = run_plate(
        alpha=4e-6, L=0.3, q=0.0,
        Nx=30, Ny=30,
        dt=0.5,
        t_final=6000.0,
        T_init=60.0,
        bc=bc_b,
        steady_tol=1e-6,
        steady_window=30,
        snapshot_times=[0.0, 500.0, 1500.0, 3000.0, 6000.0],
        verbose=True,
        progress_updates=25,
    )

    plot_snapshots(res_b, title_prefix="Q1(b) ")
    plot_center_temperature(res_b, title="Q1(b) Center temperature vs time (Insulated top+right)")

    # Comparison plot: center temperature
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(res_a["times"], res_a["center_T"], linewidth=2, label="Dirichlet top+right")
    ax.plot(res_b["times"], res_b["center_T"], linewidth=2, label="Insulated top+right")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T_center [°C]")
    ax.set_title("Center temperature comparison")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    # Param study figures
    parameter_study()

    # Q2: show the stability derivation in console (compact, but explicit)
    print("\n=== Q2 Stability (Von Neumann) ===")
    print("Scheme: (u^{n+1}_m - u^{n-1}_m)/dt = (alpha/h^2)(u^n_{m+1}-2u^n_m+u^n_{m-1})")
    print("Assume u^n_m = G^n exp(i k m h).")
    print("(G - G^{-1})/dt = (alpha/h^2)(e^{ikh}-2+e^{-ikh}) = -(4 alpha/h^2) sin^2(kh/2).")
    print("Let mu = 4 alpha dt / h^2 * sin^2(kh/2) >= 0.")
    print("Then G - G^{-1} = -mu  =>  G^2 + mu G - 1 = 0.")
    print("Roots: G = (-mu ± sqrt(mu^2 + 4))/2, with product G1*G2 = -1.")
    print("For any mu>0 (nontrivial mode), roots are real and one has |G|>1 -> unstable.")

    plt.show()


if __name__ == "__main__":
    main()
