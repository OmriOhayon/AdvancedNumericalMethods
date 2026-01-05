import numpy as np
import matplotlib.pyplot as plt

def calc_ddf(lamda, eta, dtheta, theta):
    """Calculate the second derivative of f (equation 8 from the paper).
    
    Args:
        lamda (float): Lambda parameter.
        eta (float): Similarity variable.
        dtheta (float): First derivative of theta.
        theta (float): Temperature function.
    
    Returns:
        float: Second derivative of f (f'').
    """
    ddf = -((lamda - 2)/3 * eta * dtheta + lamda * theta)
    return ddf

def calc_ddtheta(lamda, df, theta, dtheta, f):
    """Calculate the second derivative of theta (equation 9 from the paper).
    
    Args:
        lamda (float): Lambda parameter.
        df (float): First derivative of f.
        theta (float): Temperature function.
        dtheta (float): First derivative of theta.
        f (float): Stream function.
    
    Returns:
        float: Second derivative of theta (theta'').
    """
    ddtheta = lamda * df * theta - (lamda + 1)/3 * f * dtheta
    return ddtheta

def derivatives(eta, y):
    """Compute the system of first-order ODEs for the shooting method.
    
    Args:
        eta (float): Independent variable (similarity variable).
        y (np.ndarray): State vector [f, f', theta, theta'].
    
    Returns:
        np.ndarray: Derivatives [f', f'', theta', theta''].
    """
    # y = [f, f', theta, theta']
    f = y[0]
    df = y[1]
    theta = y[2]
    dtheta = y[3]
    
    # משוואה 8 מהמאמר
    ddf = calc_ddf(LAMBDA, eta, dtheta, theta)
    
    # משוואה 9 מהמאמר
    ddtheta = calc_ddtheta(LAMBDA, df, theta, dtheta, f)
    
    return np.array([df, ddf, dtheta, ddtheta])

def RK4_solver(fun, eta, y, h):
    """Fourth-order Runge-Kutta solver for ODE systems.
    
    Args:
        fun (callable): Function that computes derivatives dy/deta = fun(eta, y).
        eta (float): Current value of independent variable.
        y (np.ndarray): Current state vector.
        h (float): Step size.
    
    Returns:
        np.ndarray: Updated state vector at eta + h.
    """
    k1 = fun(eta, y)
    k2 = fun(eta + 0.5*h, y + 0.5*h*k1)
    k3 = fun(eta + 0.5*h, y + 0.5*h*k2)
    k4 = fun(eta + h, y + h*k3)

    y_new = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return y_new

def integrate_system(fw, df_guess, dtheta_guess):
    """Integrate the ODE system using RK4 method.
    
    Args:
        fw (float): Initial value of f at eta=0.
        df_guess (float): Guessed value of f' at eta=0.
        dtheta_guess (float): Guessed value of theta' at eta=0.
    
    Returns:
        tuple: (eta_array, results_array) where:
            - eta_array: Array of eta values.
            - results_array: Array of shape (N, 4) with [f, f', theta, theta'] at each eta.
    """
    y0 = np.array([fw, df_guess, 1.0, dtheta_guess]) # תנאי התחלה: f(0)=fw, theta(0)=1
    eta = 0.0
    results = [y0.copy()] # לאחסון התוצאות
    etas = [eta] # לאחסון ערכי eta

    for i in range(STEPS):
        y0 = RK4_solver(derivatives, eta, y0, H)
        eta += H
        results.append(y0.copy())
        etas.append(eta)
        # if y0[2] < 1e-8:  # עצירה מוקדמת אם theta קטן מאוד
        #     return np.array(etas), np.array(results)
        
    return np.array(etas), np.array(results)

def newton_raphson_solver(x1, x2Df, x2Theta, y1, y2Df, y2Theta, delta, initGuess, errors):
    """Newton-Raphson solver for updating initial guess in shooting method.
    
    Args:
        x1 (float): Baseline f' at boundary.
        x2Df (float): f' at boundary with perturbed df_guess.
        x2Theta (float): theta at boundary with perturbed df_guess.
        y1 (float): Baseline theta at boundary.
        y2Df (float): f' at boundary with perturbed dtheta_guess.
        y2Theta (float): theta at boundary with perturbed dtheta_guess.
        delta (float): Perturbation size.
        initGuess (np.ndarray): Current guess [df_guess, dtheta_guess].
        errors (np.ndarray): Error vector [f'_error, theta_error].
    
    Returns:
        np.ndarray: Updated guess [df_guess, dtheta_guess].
    """
    J = np.zeros((2, 2)) # הגדרת יעקוביאן לניוטון רפסון
    # J[i,j] = ∂(error_i)/∂(guess_j)
    # error vector = [f'_error, theta_error]
    # guess vector = [df_guess, dtheta_guess]
    J[0, 0] = (x2Df - x1) / delta      # ∂(f'_error)/∂(df_guess)
    J[0, 1] = (y2Df - x1) / delta      # ∂(f'_error)/∂(dtheta_guess)
    J[1, 0] = (x2Theta - y1) / delta   # ∂(theta_error)/∂(df_guess)
    J[1, 1] = (y2Theta - y1) / delta   # ∂(theta_error)/∂(dtheta_guess)
    damper = 0.1
    try:
        update = damper * np.linalg.solve(J, errors)
        guess = initGuess - update
        return guess
    except np.linalg.LinAlgError:
        print("Singular Matrix encountered. Try different initial guesses.")
        return initGuess

def derivatives_with_sens(eta, y_aug):
    """
    Augmented system:
      y    = [f, df, theta, dtheta]
      s1   = dy/d(df0)      (4 vars)
      s2   = dy/d(dtheta0)  (4 vars)
    y_aug = [y(4), s1(4), s2(4)] => 12 elements
    """
    f, df, theta, dtheta = y_aug[0:4]
    s1 = y_aug[4:8]
    s2 = y_aug[8:12]

    lam = LAMBDA  # uses your global

    # original RHS (same as your derivatives())
    ddf = -(((lam - 2.0) / 3.0) * eta * dtheta + lam * theta)
    ddtheta = lam * df * theta - ((lam + 1.0) / 3.0) * f * dtheta
    dy = np.array([df, ddf, dtheta, ddtheta], dtype=float)

    # Jacobian A = ∂(dy)/∂(y)
    A = np.zeros((4, 4), dtype=float)
    A[0, 1] = 1.0
    A[1, 2] = -lam
    A[1, 3] = -((lam - 2.0) / 3.0) * eta
    A[2, 3] = 1.0
    A[3, 0] = -((lam + 1.0) / 3.0) * dtheta
    A[3, 1] = lam * theta
    A[3, 2] = lam * df
    A[3, 3] = -((lam + 1.0) / 3.0) * f

    ds1 = A @ s1
    ds2 = A @ s2

    out = np.zeros_like(y_aug, dtype=float)
    out[0:4] = dy
    out[4:8] = ds1
    out[8:12] = ds2
    return out

def integrate_system_with_sens(fw, df_guess, dtheta_guess):
    """
    Same as integrate_system(), but also integrates sensitivities so you get
    Jacobian wrt initial guesses in ONE integration.
    Returns:
      etas: (N,)
      Y:    (N,4)  [f, df, theta, dtheta]
      S:    (N,8)  [s1(4), s2(4)] each row
    """
    # base ICs
    y0 = np.array([fw, df_guess, 1.0, dtheta_guess], dtype=float)

    # sensitivities ICs:
    # s1 = dy/d(df0): [0, 1, 0, 0]
    # s2 = dy/d(dtheta0): [0, 0, 0, 1]
    s1_0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    s2_0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    y_aug = np.concatenate([y0, s1_0, s2_0])

    eta = 0.0
    etas = [eta]
    Y = [y0.copy()]
    S = [np.concatenate([s1_0, s2_0])]

    for _ in range(STEPS):
        y_aug = RK4_solver(derivatives_with_sens, eta, y_aug, H)
        eta += H

        y = y_aug[0:4]
        s1 = y_aug[4:8]
        s2 = y_aug[8:12]

        etas.append(eta)
        Y.append(y.copy())
        S.append(np.concatenate([s1, s2]))

        # hard fail early if it blows up
        if not np.isfinite(y_aug).all():
            break

    return np.array(etas), np.array(Y), np.array(S)

def newton_update_from_J(initGuess, errors, J, damper=0.2):
    """
    initGuess: [df0, dtheta0]
    errors:    [df_end - target_fp, theta_end - target_theta]
    J: 2x2 Jacobian of errors wrt [df0, dtheta0]
    """
    # basic conditioning check
    if not np.isfinite(J).all() or not np.isfinite(errors).all():
        return initGuess

    try:
        step = np.linalg.solve(J, errors)
    except np.linalg.LinAlgError:
        return initGuess

    return initGuess - damper * step

def shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-3, H_input=0.1):
    """Solve boundary value problem using shooting method with Newton-Raphson iteration.
    
    Args:
        lambdaArray (np.ndarray): Array of lambda values to solve for.
        fw (np.ndarray): Array of wall velocity values.
        oneOverC (np.ndarray): Array of 1/C values (suction/injection parameter).
        initGuessArray (np.ndarray): Initial guess table with shape (N, 6).
        TARGET_THETA (float): Target value for theta at infinity (typically 0).
        EPS (float, optional): Convergence tolerance. Defaults to 1e-3.
        H_input (float, optional): Step size for integration. Defaults to 0.1.
    
    Returns:
        tuple: (resArray, paramsArray, iterationsArray) where:
            - resArray: List of tuples (eta_array, solution_array) for each case.
            - paramsArray: List of parameter dictionaries for each case.
            - iterationsArray: List of iteration counts for each case.
    """
    global ETA_INF, STEPS, H, LAMBDA
    resArray = []
    paramsArray = []  # Store parameters for each run
    iterationsArray = []  # Store iteration counts for each run
    # Full arrays matching the data table structure
    full_lambda = [0.5, 2.0]
    full_fw = [-1.0, 0.0, 1.0]
    full_oneOverC = [1.0, 2.0, 5.0, 8.0]
    
    for lIdx, lamda in enumerate(lambdaArray):
        for fwIdx, fw_val in enumerate(fw):
            for cIdx, c in enumerate(oneOverC):
                # if lamda == 2.0 and fw_val == -1.0 and (c == 5.0 or c == 8.0):
                #     print(f"Skipping known divergent case: Lambda={lamda}, fw={fw_val}, C={c}")
                #     continue
                # Calculate correct index into data table
                lambda_idx_full = full_lambda.index(lamda)
                fw_idx_full = full_fw.index(fw_val)
                c_idx_full = full_oneOverC.index(c)
                ii = lambda_idx_full * 12 + fw_idx_full * 4 + c_idx_full
                
                if lamda == 2 and fw_val == -1.0:
                    if c == 5.0:
                        ETA_INF = initGuessArray[ii, 5] * 3
                    elif c == 8.0:
                        ETA_INF = initGuessArray[ii, 5] * 3.17
                    elif c == 2.0:
                        ETA_INF = initGuessArray[ii, 5] * 2.5
                    else:
                        ETA_INF = initGuessArray[ii, 5] * 3
                else:
                    ETA_INF = initGuessArray[ii, 5] * 3
                # ETA_INF = 6
                H = H_input
                STEPS = int(ETA_INF / H)
                LAMBDA = lamda
                C_PARAM_FINAL = c**(-2/3)
                C_PARAM = c**(1/3)
                initDThetaGuess = initGuessArray[ii, 3]
                initDfGuess = initGuessArray[ii, 4]
                # Boundary condition: f(0) = fw (no scaling needed for shooting method)
                fwToUse = fw_val * C_PARAM
                # res = integrate_system(fwToUse, initDfGuess, initDThetaGuess)
                # resArray.append(res)
                etas, Y, S = integrate_system_with_sens(fwToUse, initDfGuess, initDThetaGuess)
                res = (etas, Y)              # keeps your existing plotting/packing
                resArray.append(res)
                paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})

                boundryValues = res[1][-1, :]
                fFinal = boundryValues[0]
                dfFinal = boundryValues[1]
                thetaFinal = boundryValues[2]
                dthetaFinal = boundryValues[3]
                dfFinal = Y[-1, 1]
                thetaFinal = Y[-1, 2]

                fTagError = dfFinal - C_PARAM_FINAL
                thetaError = thetaFinal - TARGET_THETA
                errors = np.array([fTagError, thetaError], dtype=float)
                errNorm = max(abs(fTagError), abs(thetaError))
                
                if errNorm <= EPS:
                    print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}. Error Norm={errNorm}")
                    iterationsArray.append(1)  # Initial integration counts as 1
                    continue  # Skip to next iteration, already converged
                
                # Initialize divergence check - will be updated in first loop iteration
                divergence_detected = False
                iteration_count = 1  # First integration already done above
                
                # --- Newton loop ---
                while errNorm > EPS:
                    # re-integrate with current guess (ONE run)
                    etas, Y, S = integrate_system_with_sens(fwToUse, initDfGuess, initDThetaGuess)

                    # update stored result (same as you do)
                    res = (etas, Y)
                    resArray[-1] = res

                    dfFinal = Y[-1, 1]
                    thetaFinal = Y[-1, 2]
                    errors = np.array([dfFinal - C_PARAM_FINAL, thetaFinal - TARGET_THETA], dtype=float)
                    errNorm = max(abs(errors[0]), abs(errors[1]))

                    if errNorm <= EPS:
                        print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}. Final Error Norm={errNorm}, Iterations={iteration_count}")
                        break

                    # sensitivities at boundary
                    s1_end = S[-1, 0:4]  # dy/d(df0)
                    s2_end = S[-1, 4:8]  # dy/d(dtheta0)

                    # Jacobian of residuals w.r.t [df0, dtheta0]
                    J = np.array([
                        [s1_end[1], s2_end[1]],  # d(df_end)/d(df0), d(df_end)/d(dtheta0)
                        [s1_end[2], s2_end[2]],  # d(theta_end)/d(df0), d(theta_end)/d(dtheta0)
                    ], dtype=float)

                    # update guess
                    guess = np.array([initDfGuess, initDThetaGuess], dtype=float)
                    guess_new = newton_update_from_J(guess, errors, J, damper=0.2)

                    initDfGuess, initDThetaGuess = guess_new
                    iteration_count += 1  # Count the Newton update

                    # optional: bail if NaN
                    if not np.isfinite([initDfGuess, initDThetaGuess]).all():
                        print(f"Diverged (NaN guess) for Lambda={lamda}, fw={fw_val}, C={c}")
                        break
                
                iterationsArray.append(iteration_count)

    return resArray, paramsArray, iterationsArray

def cumulative_trapezoid_integration(y, h):
    """Compute cumulative trapezoidal integration from 0 to each point.
    
    Args:
        y (np.ndarray): Array of function values to integrate.
        h (float): Step size (uniform spacing).
    
    Returns:
        np.ndarray: Cumulative integral values, with out[0] = 0.
    """
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * h)
    return out

def central_difference_derivative(y, h):
    """Compute first derivative using central differences with 2nd-order one-sided boundaries.
    
    Args:
        y (np.ndarray): Array of function values.
        h (float): Step size (uniform spacing).
    
    Returns:
        np.ndarray: First derivative dy/dx at each point.
    """
    dy = np.zeros_like(y)
    # Central difference for interior points
    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * h)
    # 2nd-order one-sided at boundaries
    dy[0] = (-3*y[0] + 4*y[1] - y[2]) / (2.0 * h)
    dy[-1] = (3*y[-1] - 4*y[-2] + y[-3]) / (2.0 * h)
    return dy

def integrate_f_from_theta(eta, theta, lamda, C_PARAM_FINAL, fwToUse, H):
    """Integrate f from equation (8) using current theta distribution.
    
    Computes f'' from equation (8): f'' = -((λ-2)/3 * η * θ' + λ * θ)
    Then integrates twice to get f' and f.
    
    Args:
        eta (np.ndarray): Array of eta (similarity variable) values.
        theta (np.ndarray): Current temperature distribution.
        lamda (float): Lambda parameter.
        C_PARAM_FINAL (float): Target f'(∞) boundary condition (C^(-2/3)).
        fwToUse (float): Initial value f(0) = fw * C^(1/3).
        H (float): Step size.
    
    Returns:
        tuple: (f, fp, fpp) where:
            - f: Stream function values.
            - fp: First derivative f' values.
            - fpp: Second derivative f'' values.
    """
    # Compute theta derivative
    dtheta = central_difference_derivative(theta, H)
    
    # Equation (8): f'' = -((λ-2)/3 * η * θ' + λ * θ)
    fpp = -(((lamda - 2.0) / 3.0) * eta * dtheta + lamda * theta)
    
    # Enforce far-field slope: fp(0) = fp_inf - ∫_0^∞ f'' dη
    I = np.trapz(fpp, x=eta)
    fp0 = C_PARAM_FINAL - I
    
    # Integrate f'' to get f'
    fp = fp0 + cumulative_trapezoid_integration(fpp, H)
    
    # Integrate f' to get f
    f = fwToUse + cumulative_trapezoid_integration(fp, H)
    
    return f, fp, fpp

def gauss_seidel_theta_sweep(theta, f, fp, lamda, H, STEPS, omega=1.5):
    """Perform one Gauss-Seidel sweep for theta using equation (9).
    
    Solves the discretized equation (9): θ'' + p*θ' + q*θ = 0
    where p = (λ+1)/3 * f and q = -λ * f'
    
    Uses central differences for θ'' and θ', leading to:
    (1/h² - p/(2h)) * θ[i-1] + (-2/h² + q) * θ[i] + (1/h² + p/(2h)) * θ[i+1] = 0
    
    Args:
        theta (np.ndarray): Current theta distribution (modified in place).
        f (np.ndarray): Stream function values.
        fp (np.ndarray): First derivative f' values.
        lamda (float): Lambda parameter.
        H (float): Step size.
        STEPS (int): Number of grid points.
        omega (float, optional): Relaxation factor for SOR. Defaults to 0.8.
    
    Returns:
        np.ndarray: Updated theta distribution.
    """
    inv_h2 = 1.0 / (H * H)
    inv_2h = 1.0 / (2.0 * H)
    
    # Boundary conditions
    theta[0] = 1.0
    theta[-1] = 0.0
    
    # Gauss-Seidel sweep for interior points
    for i in range(1, STEPS - 1):
        # Coefficients from equation (9)
        p = ((lamda + 1.0) / 3.0) * f[i]
        q = -lamda * fp[i]
        
        coeff_minus = inv_h2 - p * inv_2h
        coeff_plus = inv_h2 + p * inv_2h
        denom = -2.0 * inv_h2 + q
        
        # Safety check for near-zero denominator
        if abs(denom) < 1e-300:
            denom = np.copysign(1e-300, denom)
        
        # Compute new value
        val = -(coeff_minus * theta[i - 1] + coeff_plus * theta[i + 1]) / denom
        
        # Apply relaxation (SOR)
        theta[i] = (1.0 - omega) * theta[i] + omega * val
    
    # Re-enforce boundary conditions
    theta[0] = 1.0
    theta[-1] = 0.0
    
    return theta

def finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5, H_input=0.1, omega_input=1.5):
    """Solve boundary value problem using finite difference method with iterative solving.
    
    Uses a block iteration scheme where f is integrated from equation (8) and theta
    is solved using Gauss-Seidel iteration from equation (9).
    
    Args:
        lambdaArray (np.ndarray): Array of lambda values to solve for.
        fw (np.ndarray): Array of wall velocity values.
        oneOverC (np.ndarray): Array of 1/C values (suction/injection parameter).
        initGuessArray (np.ndarray): Initial guess table with shape (N, 6).
        TARGET_THETA (float): Target value for theta at infinity (typically 0).
        EPS (float, optional): Convergence tolerance. Defaults to 1e-5.
        H_input (float, optional): Step size for grid. Defaults to 0.1.
        omega_input (float, optional): Relaxation parameter for Gauss-Seidel (SOR). Defaults to 0.8.
    
    Returns:
        tuple: (resArray, paramsArray, iterationsArray) where:
            - resArray: List of tuples (eta_array, solution_array) for each case.
            - paramsArray: List of parameter dictionaries for each case.
            - iterationsArray: List of iteration counts for each case.
    """
    global ETA_INF, STEPS, H, LAMBDA, MAX_ITER
    MAX_ITER = 30000
    resArray = []
    paramsArray = []  # Store parameters for each run
    iterationsArray = []  # Store iteration counts for each run
    # Full arrays matching the data table structure
    full_lambda = [0.5, 2.0]
    full_fw = [-1.0, 0.0, 1.0]
    full_oneOverC = [1.0, 2.0, 5.0, 8.0]

    for lamda in lambdaArray:
        for fw_val in fw:
            for c in oneOverC:
                # Calculate correct index into data table
                lambda_idx_full = full_lambda.index(lamda)
                fw_idx_full = full_fw.index(fw_val)
                c_idx_full = full_oneOverC.index(c)
                ii = lambda_idx_full * 12 + fw_idx_full * 4 + c_idx_full
                
                # Match the shooting method's eta_inf selection for the same case
                if lamda == 2 and fw_val == -1.0:
                    if c == 5.0:
                        ETA_INF = initGuessArray[ii, 5] * 3
                    elif c == 8.0:
                        ETA_INF = initGuessArray[ii, 5] * 3.17
                    elif c == 2.0:
                        ETA_INF = initGuessArray[ii, 5] * 2.5
                    else:
                        ETA_INF = initGuessArray[ii, 5] * 3
                else:
                    ETA_INF = initGuessArray[ii, 5] * 3
                # ---- FIX: consistent grid ----
                H = H_input
                N = int(np.ceil(ETA_INF / H))
                if N < 6:
                    N = 6
                    ETA_INF = N * H
                eta = np.linspace(0.0, ETA_INF, N + 1)
                H = eta[1] - eta[0]
                STEPS = N + 1
                LAMBDA = lamda

                # oneOverC = 1/C
                C_PARAM_FINAL = c**(-2/3)     # = C^(2/3) : f'(inf)
                C_PARAM = c**(1/3)            # = C^(-1/3)
                fwToUse = fw_val * C_PARAM    # f(0)

                # init guesses
                theta = np.exp(-eta)
                theta[0] = 1.0
                theta[-1] = 0.0

                # f init consistent with far-field slope
                f = fwToUse + C_PARAM_FINAL * eta
                fp = np.full_like(eta, C_PARAM_FINAL)

                omega_theta = omega_input  # relaxation inside theta GS sweep
                omega_block = 1.0  # optional block relaxation (theta <- mix(theta_old, theta_new))

                converged = False
                for k in range(MAX_ITER):
                    theta_old = theta.copy()
                    f_old = f.copy()

                    # ===== Block 1: integrate f from eq (8) using current theta =====
                    f, fp, fpp = integrate_f_from_theta(eta, theta, LAMBDA, C_PARAM_FINAL, fwToUse, H)

                    # ===== Block 2: GS sweep for theta from eq (9) using updated f, fp =====
                    theta = gauss_seidel_theta_sweep(theta, f, fp, LAMBDA, H, STEPS, omega=omega_theta)

                    # optional block relaxation on theta (helps if oscillatory)
                    if omega_block < 1.0:
                        theta = (1.0 - omega_block) * theta_old + omega_block * theta
                        theta[0] = 1.0
                        theta[-1] = 0.0

                    err_theta = np.max(np.abs(theta - theta_old))
                    err_f = np.max(np.abs(f - f_old))

                    if k % 1000 == 0:
                        fp_inf_est = (3*f[-1] - 4*f[-2] + f[-3]) / (2.0 * H)
                        print(f"Iter {k}: Theta Error={err_theta:.2e}, f'(inf)={fp_inf_est:.6f}, STEPS={STEPS}")

                    if max(err_theta, err_f) < EPS:
                        print(f"  Converged in {k} iterations.")
                        dtheta_out = central_difference_derivative(theta, H)
                        resArray.append([eta, np.asarray([f, fp, theta, dtheta_out]).T])
                        paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})
                        iterationsArray.append(k)
                        converged = True
                        break

                if not converged:
                    print("Did not converge.")
                    dtheta_out = central_difference_derivative(theta, H)
                    resArray.append([eta, np.asarray([f, fp, theta, dtheta_out]).T])
                    paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})
                    iterationsArray.append(MAX_ITER)

    return resArray, paramsArray, iterationsArray


data = [ # ערכי ההתחלה מהמאמר
    # --- Lambda = 0.5 ---
    # fw = -1
    [0.5, -1.0, 1.0, -0.8862, 1.8862, 3.2154],
    [0.5, -1.0, 2.0, -1.0450, 2.5547, 2.9088],
    [0.5, -1.0, 5.0, -1.3575, 4.1212, 2.4379],
    [0.5, -1.0, 8.0, -1.5724, 5.3921, 1.8704],
    # fw = 0-
    [0.5, 0.0, 1.0, -1.1020, 1.7474, 2.8250],
    [0.5, 0.0, 2.0, -1.2495, 2.3479, 2.6049],
    [0.5, 0.0, 5.0, -1.5503, 3.7996, 2.2380],
    [0.5, 0.0, 8.0, -1.7610, 4.9990, 2.0340],
    # fw = 1-
    [0.5, 1.0, 1.0, -1.3745, 1.6264, 2.4624],
    [0.5, 1.0, 2.0, -1.5041, 2.1591, 2.3115],
    [0.5, 1.0, 5.0, -1.7825, 3.4927, 2.0347],
    [0.5, 1.0, 8.0, -1.9836, 4.6181, 1.8688],

    # --- Lambda = 2.0 ---
    # fw = -1
    [2.0, -1.0, 1.0, -1.6309, 2.1235, 2.2138],
    [2.0, -1.0, 2.0, -1.9494, 2.9781, 2.0284],
    [2.0, -1.0, 5.0, -2.5716, 4.9815, 1.7304],
    [2.0, -1.0, 8.0, -2.9971, 6.6069, 1.5729],
    # fw = 0, 3.2154
    [2.0, 0.0, 1.0, -2.0044, 1.9159, 1.8532],
    [2.0, 0.0, 2.0, -2.2889, 2.6630, 1.7334],
    [2.0, 0.0, 5.0, -2.8820, 4.4981, 1.5188],
    [2.0, 0.0, 8.0, -3.2927, 6.0105, 1.3954],
    # fw = 1-
    [2.0, 1.0, 1.0, -2.5182, 1.7391, 1.5371],
    [2.0, 1.0, 2.0, -2.7574, 2.3852, 1.4656],
    [2.0, 1.0, 5.0, -3.2827, 4.0268, 1.3230],
    [2.0, 1.0, 8.0, -3.6683, 5.4273, 1.2327]
]

initGuessArray = np.array(data)

lambdaArray = np.asarray([0.5, 2])
fw = np.asarray([-1, 0, 1])
oneOverC = np.asarray([1, 2, 5, 8])

resArray = []
TARGET_THETA = 0.0

lambdaLabels = {0.5: '0.5', 2.0: '2.0'}
cLabels = {1: '1', 2: '2', 5: '5', 8: '8'}
labels = [r'$f(\eta)$', r"$f'(\eta)$", r'$\theta(\eta)$', r"$\theta'(\eta)$"]

# Run both methods
print("Running Shooting Method...")
resArray_shooting, paramsArray_shooting, iterArray_shooting = shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5, H_input=0.1)
print("\nRunning Finite Difference Method...")
resArray_fd, paramsArray_fd, iterArray_fd = finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5, H_input=0.1)

# Flexible plotting function
def plot_comparison(variable_idx, filter_lambda=None, filter_fw=None, filter_oneOverC=None, 
                   plot_shooting=True, plot_fd=True):
    """Plot and compare results from shooting and finite difference methods with filtering options.
    
    Args:
        variable_idx (int): Index of variable to plot (0=f, 1=f', 2=theta, 3=theta').
        filter_lambda (float, optional): Filter results by lambda value (0.5 or 2.0). Defaults to None.
        filter_fw (float, optional): Filter results by fw value (-1, 0, or 1). Defaults to None.
        filter_oneOverC (float, optional): Filter results by 1/C value (1, 2, 5, or 8). Defaults to None.
        plot_shooting (bool, optional): Whether to plot shooting method results. Defaults to True.
        plot_fd (bool, optional): Whether to plot finite difference results. Defaults to True.
    
    Returns:
        None: Displays matplotlib figure.
    
    Examples:
        >>> plot_comparison(2)  # Plot all theta for all parameters
        >>> plot_comparison(3, filter_oneOverC=2)  # Plot theta' only for 1/C=2
        >>> plot_comparison(2, filter_fw=1, plot_fd=False)  # Plot theta from shooting only, fw=1
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    
    # Plot shooting method results
    if plot_shooting:
        for i, (run, params) in enumerate(zip(resArray_shooting, paramsArray_shooting)):
            # Apply filters
            if filter_lambda is not None and params['lambda'] != filter_lambda:
                continue
            if filter_fw is not None and params['fw'] != filter_fw:
                continue
            if filter_oneOverC is not None and params['oneOverC'] != filter_oneOverC:
                continue
            
            eta_vals = run[0]
            f_vals = run[1][:, variable_idx]
            legend_label = f"Shoot: λ={params['lambda']}, fw={params['fw']}, 1/C={params['oneOverC']}"
            ax.plot(eta_vals, f_vals, label=legend_label, linestyle='-')
    
    # Plot finite difference results
    if plot_fd:
        for i, (run, params) in enumerate(zip(resArray_fd, paramsArray_fd)):
            # Apply filters
            if filter_lambda is not None and params['lambda'] != filter_lambda:
                continue
            if filter_fw is not None and params['fw'] != filter_fw:
                continue
            if filter_oneOverC is not None and params['oneOverC'] != filter_oneOverC:
                continue
            
            eta_vals = run[0]
            f_vals = run[1][:, variable_idx]
            legend_label = f"FD: λ={params['lambda']}, fw={params['fw']}, 1/C={params['oneOverC']}"
            ax.plot(eta_vals, f_vals, label=legend_label, linestyle='--')
    
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(labels[variable_idx])
    # ax.set_title(labels[variable_idx])
    # Normal legend placement (inside axes) + tight layout to avoid clipping.
    ax.legend(loc='best', fontsize='small')
    # fig.tight_layout()

# # Example plots - uncomment the ones you want:

# # # Plot all theta (variable_idx=2) for all fw values
# plot_comparison(2, filter_lambda=None, filter_fw=None, filter_oneOverC=None)

# # Plot all f' (variable_idx=1) for all runs
# plot_comparison(1)

# # Plot theta' (variable_idx=3) for oneOverC=2
# plot_comparison(3, filter_oneOverC=None)

# # Plot theta for fw=1 only
# # plot_comparison(2, filter_fw=1)

# # Plot f for lambda=0.5, fw=0
# plot_comparison(0, filter_lambda=None, filter_fw=None)


def plot_iterations_vs_H(H_array, lamda_val, fw_val, oneOverC_val, 
                          plot_shooting=True, plot_fd=True, EPS=1e-5):
    """Plot number of iterations vs step size H for a specific scenario.
    
    Args:
        H_array (array-like): Array of H values to test.
        lamda_val (float): Lambda parameter value (0.5 or 2.0).
        fw_val (float): Wall velocity value (-1, 0, or 1).
        oneOverC_val (float): 1/C parameter value (1, 2, 5, or 8).
        plot_shooting (bool, optional): Whether to plot shooting method results. Defaults to True.
        plot_fd (bool, optional): Whether to plot finite difference results. Defaults to True.
    
    Returns:
        dict: Dictionary containing iteration counts for each method and H value.
    
    Examples:
        >>> plot_iterations_vs_H([0.05, 0.1, 0.2, 0.5], 0.5, 0, 2)
        >>> plot_iterations_vs_H([0.01, 0.05, 0.1], 2.0, -1, 1, plot_fd=False)
    """
    lambdaArr = np.asarray([lamda_val])
    fwArr = np.asarray([fw_val])
    oneOverCArr = np.asarray([oneOverC_val])
    
    shooting_iters = []
    fd_iters = []
    
    for H_val in H_array:
        print(f"\n--- Testing H = {H_val} ---")
        
        if plot_shooting:
            print(f"  Running Shooting Method with H={H_val}...")
            _, _, iter_shoot = shooting_method(lambdaArr, fwArr, oneOverCArr, 
                                                initGuessArray, TARGET_THETA, 
                                                EPS=EPS, H_input=H_val)
            shooting_iters.append(iter_shoot[0] if iter_shoot else 0)
        
        if plot_fd:
            print(f"  Running Finite Difference Method with H={H_val}...")
            _, _, iter_fd = finite_difference_method(lambdaArr, fwArr, oneOverCArr, 
                                                      initGuessArray, TARGET_THETA, 
                                                      EPS=EPS, H_input=H_val)
            fd_iters.append(iter_fd[0] if iter_fd else 0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_shooting and shooting_iters:
        ax.plot(H_array, shooting_iters, 'o-', label='Shooting Method', 
                markersize=8, linewidth=2)
    
    if plot_fd and fd_iters:
        ax.plot(H_array, fd_iters, 's--', label='Finite Difference Method', 
                markersize=8, linewidth=2)
    
    ax.set_xlabel('Step Size (H)', fontsize=12)
    ax.set_ylabel('Number of Iterations', fontsize=12)
    # ax.set_title(f'Iterations vs H for λ={lamda_val}, fw={fw_val}, 1/C={oneOverC_val}', fontsize=14)
    ax.grid(True, alpha=0.7)
    ax.legend(loc='best', fontsize=10)
    
    # # Use log scale if range is large
    # if plot_fd and fd_iters and max(fd_iters) > 10 * min([x for x in fd_iters if x > 0], default=1):
    #     ax.set_yscale('log')
    
    # plt.tight_layout()
    
    return {'H_array': H_array, 'shooting_iterations': shooting_iters, 'fd_iterations': fd_iters}

def plot_sensitivity_vs_EPS(EPS_array, variable_idx, lamda_val, fw_val, oneOverC_val,
                            method='shooting', H_val=0.1):
    """Plot sensitivity of output variable to convergence tolerance EPS.
    
    Args:
        EPS_array (array-like): Array of EPS (convergence tolerance) values to test.
        variable_idx (int): Index of variable to plot (0=f, 1=f', 2=theta, 3=theta').
        lamda_val (float): Lambda parameter value (0.5 or 2.0).
        fw_val (float): Wall velocity value (-1, 0, or 1).
        oneOverC_val (float): 1/C parameter value (1, 2, 5, or 8).
        method (str, optional): 'shooting' or 'fd' (finite difference). Defaults to 'shooting'.
        H_val (float, optional): Step size for integration. Defaults to 0.1.
    
    Returns:
        dict: Dictionary containing results for each EPS value.
    
    Examples:
        >>> plot_sensitivity_vs_EPS([1e-3, 1e-4, 1e-5, 1e-6], 2, 0.5, 0, 2)
        >>> plot_sensitivity_vs_EPS([1e-2, 1e-3, 1e-4], 1, 2.0, 1, 5, method='fd')
    """
    lambdaArr = np.asarray([lamda_val])
    fwArr = np.asarray([fw_val])
    oneOverCArr = np.asarray([oneOverC_val])
    
    results_per_eps = []
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color map for different EPS values
    colors = plt.cm.viridis(np.linspace(0, 1, len(EPS_array)))
    
    for idx, eps_val in enumerate(EPS_array):
        print(f"\n--- Testing EPS = {eps_val:.2e} ---")
        
        if method.lower() == 'shooting':
            res_array, params_array, iter_array = shooting_method(
                lambdaArr, fwArr, oneOverCArr, initGuessArray, TARGET_THETA,
                EPS=eps_val, H_input=H_val
            )
        else:  # finite difference
            res_array, params_array, iter_array = finite_difference_method(
                lambdaArr, fwArr, oneOverCArr, initGuessArray, TARGET_THETA,
                EPS=eps_val, H_input=H_val
            )
        
        if res_array:
            eta_vals = res_array[0][0]
            var_vals = res_array[0][1][:, variable_idx]
            iterations = iter_array[0] if iter_array else 0
            
            results_per_eps.append({
                'EPS': eps_val,
                'eta': eta_vals,
                'values': var_vals,
                'iterations': iterations
            })
            
            ax.plot(eta_vals, var_vals, color=colors[idx], 
                    label=f'EPS={eps_val:.1e} (iter={iterations})', 
                    linewidth=2, alpha=0.8)
    
    ax.set_xlabel(r'$\eta$', fontsize=12)
    ax.set_ylabel(labels[variable_idx], fontsize=12)
    method_name = 'Shooting' if method.lower() == 'shooting' else 'Finite Difference'
    # ax.set_title(f'Sensitivity Analysis: {labels[variable_idx]} vs ' + r'$\eta$' + f'\n{method_name} Method - λ={lamda_val}, fw={fw_val}, 1/C={oneOverC_val}, H={H_val}', 
    #              fontsize=12)
    ax.grid(True, alpha=0.7)
    ax.legend(loc='best', fontsize=9)
    
    # plt.tight_layout()
    
    # Create second plot: iterations vs EPS
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Plot iterations vs EPS
    eps_values = [r['EPS'] for r in results_per_eps]
    iter_values = [r['iterations'] for r in results_per_eps]
    
    # ax2.semilogx(eps_values, iter_values, 'o-', markersize=10, linewidth=2)
    ax2.plot(eps_values, iter_values, 'o-', markersize=10, linewidth=2)
    ax2.set_xlabel('EPS (Convergence Tolerance)', fontsize=12)
    ax2.set_ylabel('Number of Iterations', fontsize=12)
    # ax2.set_title('Iterations vs Convergence Tolerance', fontsize=12)
    ax2.grid(True, alpha=0.7)
    ax2.invert_xaxis()  # Show smaller EPS on right (tighter tolerance)
    
    # Create third plot: final value at eta_inf vs EPS
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Plot final value at eta_inf vs EPS (to check convergence)
    if len(results_per_eps) >= 2:
        final_values = [r['values'][-1] for r in results_per_eps]
        # ax3.semilogx(eps_values, final_values, 's-', markersize=10, linewidth=2, color='red')
        ax3.plot(eps_values, final_values, 's-', markersize=10, linewidth=2, color='red')
        ax3.set_xlabel('EPS (Convergence Tolerance)', fontsize=12)
        ax3.set_ylabel(f'{labels[variable_idx]} at ' + r'$\eta \rightarrow \infty$', fontsize=12)
        # ax3.set_title('Final Value vs Convergence Tolerance', fontsize=12)
        ax3.grid(True, alpha=0.7)
        ax3.invert_xaxis()
    
    # plt.tight_layout()
    
    return {'results': results_per_eps, 'variable': labels[variable_idx]}

def plot_relaxation_comparison(omega_array, lamda_val, fw_val, oneOverC_val, 
                                variable_idx=2, H_val=0.1, EPS_val=1e-5):
    """Plot comparison of different relaxation parameters (omega) for Gauss-Seidel in FD method.
    
    Args:
        omega_array (array-like): Array of omega (relaxation) values to test (e.g., [0.5, 0.8, 1.2]).
        lamda_val (float): Lambda parameter value (0.5 or 2.0).
        fw_val (float): Wall velocity value (-1, 0, or 1).
        oneOverC_val (float): 1/C parameter value (1, 2, 5, or 8).
        variable_idx (int, optional): Index of variable to plot (0=f, 1=f', 2=theta, 3=theta'). Defaults to 2.
        H_val (float, optional): Step size for grid. Defaults to 0.1.
        EPS_val (float, optional): Convergence tolerance. Defaults to 1e-5.
    
    Returns:
        dict: Dictionary containing iteration counts and results for each omega value.
    
    Examples:
        >>> plot_relaxation_comparison([0.5, 0.8, 1.2], 0.5, 0, 2)
        >>> plot_relaxation_comparison([0.6, 1.0, 1.4], 2.0, -1, 5, variable_idx=0)
    """
    lambdaArr = np.asarray([lamda_val])
    fwArr = np.asarray([fw_val])
    oneOverCArr = np.asarray([oneOverC_val])
    
    results_per_omega = []
    
    # Color map for different omega values
    colors = plt.cm.viridis(np.linspace(0, 1, len(omega_array)))
    
    for idx, omega_val in enumerate(omega_array):
        print(f"\n--- Testing omega = {omega_val} ---")
        
        res_array, params_array, iter_array = finite_difference_method(
            lambdaArr, fwArr, oneOverCArr, initGuessArray, TARGET_THETA,
            EPS=EPS_val, H_input=H_val, omega_input=omega_val
        )
        
        if res_array:
            eta_vals = res_array[0][0]
            var_vals = res_array[0][1][:, variable_idx]
            iterations = iter_array[0] if iter_array else 0
            converged = iterations < 30000  # MAX_ITER
            
            results_per_omega.append({
                'omega': omega_val,
                'eta': eta_vals,
                'values': var_vals,
                'iterations': iterations,
                'converged': converged
            })
    
    # Create figure 1: Solution profiles for each omega
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot 1: Solution profiles for each omega
    for idx, result in enumerate(results_per_omega):
        status = "" if result['converged'] else " (not conv.)"
        ax1.plot(result['eta'], result['values'], 
                     color=colors[idx], linewidth=2,
                     label=f"ω={result['omega']}, iters={result['iterations']}{status}")
    
    ax1.set_xlabel(r'$\eta$', fontsize=12)
    ax1.set_ylabel(labels[variable_idx], fontsize=12)
    # ax1.set_title(f'{labels[variable_idx]} vs ' + r'$\eta$' + r' for different $\omega$', fontsize=12)
    ax1.grid(True, alpha=0.7)
    ax1.legend(loc='best', fontsize=9)
    
    # Create figure 2: Iterations vs omega (bar chart)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    omega_vals = [r['omega'] for r in results_per_omega]
    iter_vals = [r['iterations'] for r in results_per_omega]
    bar_colors = ['green' if r['converged'] else 'red' for r in results_per_omega]
    
    bars = ax2.bar(range(len(omega_vals)), iter_vals, color=bar_colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(omega_vals)))
    ax2.set_xticklabels([f'{o}' for o in omega_vals])
    ax2.set_xlabel(r'Relaxation Parameter ($\omega$)', fontsize=12)
    ax2.set_ylabel('Number of Iterations', fontsize=12)
    ax2.set_title(r'Iterations to Convergence vs $\omega$', fontsize=12)
    ax2.grid(True, alpha=0.7, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, iter_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                 f'{val}', ha='center', va='bottom', fontsize=10)
    
    # Create figure 3: Iterations vs omega (line plot for trend)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    ax3.plot(omega_vals, iter_vals, 'o-', markersize=10, linewidth=2, color='blue')
    ax3.set_xlabel(r'Relaxation Parameter ($\omega$)', fontsize=12)
    ax3.set_ylabel('Number of Iterations', fontsize=12)
    ax3.set_title(r'Convergence Rate vs $\omega$', fontsize=12)
    ax3.grid(True, alpha=0.7)
    
    # Mark optimal omega
    converged_results = [r for r in results_per_omega if r['converged']]
    if converged_results:
        min_iter_result = min(converged_results, key=lambda x: x['iterations'])
        ax3.axvline(x=min_iter_result['omega'], color='green', linestyle='--', 
                    label=r"Optimal $\omega$=" + f"{min_iter_result['omega']}")
        ax3.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    return {'omega_array': omega_vals, 'iterations': iter_vals, 'results': results_per_omega}

# ===== Example usage of analysis functions =====
# Uncomment to run sensitivity and convergence analysis

# # Example 1: Plot iterations vs H for a specific scenario
# H_test_array = [0.001, 0.005, 0.01, 0.05, 0.1]
# plot_iterations_vs_H(H_test_array, lamda_val=2, fw_val=-1, oneOverC_val=2, EPS=1e-5)

# Example 2: Plot sensitivity to EPS for theta (shooting method)
EPS_test_array = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
plot_sensitivity_vs_EPS(EPS_test_array, variable_idx=2, lamda_val=2, fw_val=-1, 
                        oneOverC_val=5, method='shooting', H_val=0.001)

# Example 3: Plot sensitivity to EPS for f (finite difference method)
EPS_test_array = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
plot_sensitivity_vs_EPS(EPS_test_array, variable_idx=0, lamda_val=2.0, fw_val=-1, 
                        oneOverC_val=5, method='shooting', H_val=0.001)

# Example 2: Plot sensitivity to EPS for theta (shooting method)
EPS_test_array = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
plot_sensitivity_vs_EPS(EPS_test_array, variable_idx=2, lamda_val=2, fw_val=-1, 
                        oneOverC_val=5, method='fd', H_val=0.001)

# Example 3: Plot sensitivity to EPS for f (finite difference method)
EPS_test_array = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
plot_sensitivity_vs_EPS(EPS_test_array, variable_idx=0, lamda_val=2.0, fw_val=-1, 
                        oneOverC_val=5, method='fd', H_val=0.001)

# # Example 4: Plot relaxation parameter comparison for Gauss-Seidel
# omega_test_array = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
# plot_relaxation_comparison(omega_test_array, lamda_val=2, fw_val=-1, oneOverC_val=5, 
#                            variable_idx=2, H_val=0.01, EPS_val=1e-5)

plt.show()