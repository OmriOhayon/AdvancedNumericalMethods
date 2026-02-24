import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

# Parameters (put Ra/Pe, fw, etc. here)
fw = 0.0    # example wall mass flux parameter
lam = 1.0   # example mixed-convection parameter

def rhs(eta, y):
    f, fp, th, thp = y
    # y = [f, f', theta, theta']
    # y' = [f', f'', theta', theta'']
    # ----- Replace these with your equations (8) and (9) -----
    # Example *structure* ONLY:
    F_val  = -((0.5 - 2)/3 * 1)  # f'' = F_val(f, fp, th, thp, eta, params)
    G_val  = ...  # th'' = G_val(f, fp, th, thp, eta, params)
    # ---------------------------------------------------------

    return [fp, F_val, th, thp]

eta_max = 10.0

def residuals(ab):
    a, b = ab  # a = f'(0), b = theta'(0)
    y0 = [fw, a, 1.0, b]   # f(0)=fw, theta(0)=1

    sol = solve_ivp(rhs, [0, eta_max], y0, dense_output=False, max_step=0.1)

    f_end, fp_end, th_end, thp_end = sol.y[:, -1]

    R1 = fp_end      # want f'(eta_max) -> 0
    R2 = th_end      # want theta(eta_max) -> 0
    return [R1, R2]

def secant_method(func, x0, x1, tol=1e-6, max_iter=100):
    """Secant method for root finding."""
    for _ in range(max_iter):
        fx0, fx1 = func(x0), func(x1)
        if abs(fx1) < tol:
            return x1
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x_new
    return x1

def newton_raphson(func, jac, x0, tol=1e-6, max_iter=100):
    """Newton-Raphson method for root finding."""
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        if np.linalg.norm(fx) < tol:
            return x
        jfx = jac(x)
        x = x - np.linalg.solve(jfx, fx)
    return x

def jacobian_fd(func, x, eps=1e-8):
    """Finite difference Jacobian approximation."""
    fx = func(x)
    n = len(x)
    m = len(fx)
    jac = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        jac[:, j] = (func(x_plus) - fx) / eps
    return jac

# initial guesses for a and b
ab_guess = [0.1, -0.1]

sol_root = root(residuals, ab_guess)
a_star, b_star = sol_root.x
print("f'(0) =", a_star, "theta'(0) =", b_star)

# Now re-integrate once with the converged values to get the full profiles
y0_star = [fw, a_star, 1.0, b_star]
sol_profiles = solve_ivp(rhs, [0, eta_max], y0_star, dense_output=True)
eta = sol_profiles.t
f, fp, th, thp = sol_profiles.y