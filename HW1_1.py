import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

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


def shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-3):
    """Solve boundary value problem using shooting method with Newton-Raphson iteration.
    
    Args:
        lambdaArray (np.ndarray): Array of lambda values to solve for.
        fw (np.ndarray): Array of wall velocity values.
        oneOverC (np.ndarray): Array of 1/C values (suction/injection parameter).
        initGuessArray (np.ndarray): Initial guess table with shape (N, 6).
        TARGET_THETA (float): Target value for theta at infinity (typically 0).
        EPS (float, optional): Convergence tolerance. Defaults to 1e-3.
    
    Returns:
        tuple: (resArray, paramsArray) where:
            - resArray: List of tuples (eta_array, solution_array) for each case.
            - paramsArray: List of parameter dictionaries for each case.
    """
    global ETA_INF, STEPS, H, LAMBDA
    resArray = []
    paramsArray = []  # Store parameters for each run
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
                H = 0.1
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
                errNorm = np.linalg.norm(errors)

                # errNorm = np.sqrt(fTagError**2 + thetaError**2)
                
                if errNorm <= EPS:
                    print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}. Error Norm={errNorm}")
                    continue  # Skip to next iteration, already converged
                
                # Initialize divergence check - will be updated in first loop iteration
                divergence_detected = False
                
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
                    errNorm = np.linalg.norm(errors)

                    if errNorm <= EPS:
                        print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}. Final Error Norm={errNorm}")
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

                    # optional: bail if NaN
                    if not np.isfinite([initDfGuess, initDThetaGuess]).all():
                        print(f"Diverged (NaN guess) for Lambda={lamda}, fw={fw_val}, C={c}")
                        break

    return resArray, paramsArray

def finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5):
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
    
    Returns:
        tuple: (resArray, paramsArray) where:
            - resArray: List of tuples (eta_array, solution_array) for each case.
            - paramsArray: List of parameter dictionaries for each case.
    """
    global ETA_INF, STEPS, H, LAMBDA, MAX_ITER
    MAX_ITER = 30000
    resArray = []
    paramsArray = []  # Store parameters for each run
    # Full arrays matching the data table structure
    full_lambda = [0.5, 2.0]
    full_fw = [-1.0, 0.0, 1.0]
    full_oneOverC = [1.0, 2.0, 5.0, 8.0]

    def cumtrapz(y, h):
        # cumulative trapezoid integral from 0..i
        out = np.zeros_like(y)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * h)
        return out

    def d1_center(y, h):
        dy = np.zeros_like(y)
        dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * h)
        # 2nd-order one-sided at boundaries
        dy[0]  = (-3*y[0] + 4*y[1] - y[2]) / (2.0*h)
        dy[-1] = ( 3*y[-1] - 4*y[-2] + y[-3]) / (2.0*h)
        return dy

    for lamda in lambdaArray:
        for fw_val in fw:
            for c in oneOverC:
                # Calculate correct index into data table
                lambda_idx_full = full_lambda.index(lamda)
                fw_idx_full = full_fw.index(fw_val)
                c_idx_full = full_oneOverC.index(c)
                ii = lambda_idx_full * 12 + fw_idx_full * 4 + c_idx_full
                
                # ETA_INF = initGuessArray[ii, 5] * 2
                ETA_INF = 7.0
                # ---- FIX: consistent grid ----
                H = 0.1
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

                omega_theta = 0.8  # relaxation inside theta GS sweep
                omega_block = 1.0  # optional block relaxation (theta <- mix(theta_old, theta_new))
                inv_h2 = 1.0 / (H * H)
                inv_2h = 1.0 / (2.0 * H)

                converged = False
                for k in range(MAX_ITER):
                    theta_old = theta.copy()
                    f_old = f.copy()

                    # ===== Block 1: integrate f from eq (8) using current theta =====
                    dtheta = d1_center(theta, H)

                    # eq (8): f'' = -((λ-2)/3 * η * θ' + λ θ)
                    fpp = -(((LAMBDA - 2.0) / 3.0) * eta * dtheta + LAMBDA * theta)

                    # enforce far-field slope: fp(0) = fp_inf - ∫_0^∞ f'' dη
                    I = np.trapz(fpp, x=eta)
                    fp0 = C_PARAM_FINAL - I

                    fp = fp0 + cumtrapz(fpp, H)
                    f = fwToUse + cumtrapz(fp, H)

                    # ===== Block 2: GS sweep for theta from eq (9) using updated f, fp =====
                    theta[0] = 1.0
                    theta[-1] = 0.0

                    for i in range(1, STEPS - 1):
                        # eq (9): theta'' + p theta' + q theta = 0
                        p = ((LAMBDA + 1.0) / 3.0) * f[i]
                        q = -LAMBDA * fp[i]

                        coeff_minus = inv_h2 - p * inv_2h
                        coeff_plus  = inv_h2 + p * inv_2h
                        denom = -2.0 * inv_h2 + q

                        # safety
                        if abs(denom) < 1e-300:
                            denom = np.copysign(1e-300, denom)

                        val = -(coeff_minus * theta[i - 1] + coeff_plus * theta[i + 1]) / denom
                        theta[i] = (1.0 - omega_theta) * theta[i] + omega_theta * val

                    theta[0] = 1.0
                    theta[-1] = 0.0

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
                        dtheta_out = d1_center(theta, H)
                        resArray.append([eta, np.asarray([f, fp, theta, dtheta_out]).T])
                        paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})
                        converged = True
                        break

                if not converged:
                    print("Did not converge.")
                    dtheta_out = d1_center(theta, H)
                    resArray.append([eta, np.asarray([f, fp, theta, dtheta_out]).T])
                    paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})

    return resArray, paramsArray

def gauss_seidel_method():
    """Placeholder for Gauss-Seidel method implementation.
    
    This function is currently not implemented.
    """
    pass

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

dataOptimized = [ # ערכי ההתחלה מהמאמר
    # --- Lambda = 0.5 ---
    # fw = -1
    [0.5, -1.0, 1.0, -0.8862, 1.8862, 3.2154],
    [0.5, -1.0, 2.0, -1.0450-0.1, 2.5547-0.1, 2.9088],
    [0.5, -1.0, 5.0, -1.3575, 4.1212-4, 2.4379],
    [0.5, -1.0, 8.0, -1.5724, 5.3921-5, 1.8704],
    # fw = 0
    [0.5, 0.0, 1.0, -1.1020, 1.7474, 2.8250],
    [0.5, 0.0, 2.0, -1.2495, 2.3479, 2.6049],
    [0.5, 0.0, 5.0, -1.5503-2.2, 3.7996-2.2, 2.2380],
    [0.5, 0.0, 8.0, -1.7610-2.2, 4.9990-2.2, 2.0340],
    # fw = 1
    [0.5, 1.0, 1.0, -1.3745, 1.6264, 2.4624],
    [0.5, 1.0, 2.0, -1.5041, 2.1591, 2.3115],
    [0.5, 1.0, 5.0, -1.7825-3, 3.4927-3, 2.0347],
    [0.5, 1.0, 8.0, -1.9836-3, 4.6181-3, 1.8688],

    # --- Lambda = 2.0 ---
    # fw = -1
    [2.0, -1.0, 1.0, -1.6309, 2.1235, 2.2138],
    [2.0, -1.0, 2.0, -1.9494, 2.9781, 2.0284],
    [2.0, -1.0, 5.0, -2.5716-0.35, 4.9815-4, 1.7304],
    [2.0, -1.0, 8.0, -2.9971-0.35, 6.6069-5, 1.5729],
    # fw = 0, 3.2154
    [2.0, 0.0, 1.0, -2.0044, 1.9159, 1.8532],
    [2.0, 0.0, 2.0, -2.2889, 2.6630, 1.7334],
    [2.0, 0.0, 5.0, -2.8820, 4.4981, 1.5188],
    [2.0, 0.0, 8.0, -3.2927, 6.0105, 1.3954],
    # fw = 1
    [2.0, 1.0, 1.0, -2.5182, 1.7391, 1.5371],
    [2.0, 1.0, 2.0, -2.7574, 2.3852, 1.4656],
    [2.0, 1.0, 5.0, -3.2827, 4.0268, 1.3230],
    [2.0, 1.0, 8.0, -3.6683, 5.4273, 1.2327]
]

initGuessArray = np.array(data)
# initGuessArray = np.array(dataOptimized)
# initGuessArray[:, 3:5] -= 0.1
# initGuessArray[:, 3] += 0.01
# initGuessArray[:, 4] -= 0.1

lambdaArray = np.asarray([0.5, 2])
# lambdaArray = np.asarray([2])
# lambdaArray = np.asarray([0.5])
fw = np.asarray([-1, 0, 1])
# fw = np.asarray([-1])
# fw = np.asarray([0])
# fw = np.asarray([1])
# fw = np.asarray([0, 1])
# oneOverC = np.asarray([1, 2, 5, 8])
# oneOverC = np.asarray([1, 2])
oneOverC = np.asarray([1, 2, 5, 8])
# oneOverC = np.asarray([1, 2, 5])
# oneOverC = np.asarray([1, 2, 8])
# oneOverC = np.asarray([2, 5, 8])
# oneOverC = np.asarray([1, 2])
# oneOverC = np.asarray([5, 8])
# oneOverC = np.asarray([8])

resArray = []

# ערכי מטרה באינסוף
# TARGET_F_TAG = C_PARAM**2
TARGET_THETA = 0.0

lambdaLabels = {0.5: '0.5', 2.0: '2.0'}
cLabels = {1: '1', 2: '2', 5: '5', 8: '8'}
labels = ['f(eta)', "f '(eta)", 'theta(eta)', "theta '(eta)"]

# Run both methods
print("Running Shooting Method...")
resArray_shooting, paramsArray_shooting = shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5)
print("\nRunning Finite Difference Method...")
resArray_fd, paramsArray_fd = finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-4)

# Flexible plotting function
def plot_comparison(variable_idx, filter_lambda=None, filter_fw=None, filter_oneOverC=None, 
                   plot_shooting=False, plot_fd=True):
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
    
    ax.set_xlabel('Eta')
    ax.set_ylabel(labels[variable_idx])
    ax.set_title(labels[variable_idx])
    # Normal legend placement (inside axes) + tight layout to avoid clipping.
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()

# Example plots - uncomment the ones you want:

# # Plot all theta (variable_idx=2) for all fw values
plot_comparison(2, filter_lambda=None, filter_fw=None, filter_oneOverC=None)

# Plot all f' (variable_idx=1) for all runs
plot_comparison(1)

# Plot theta' (variable_idx=3) for oneOverC=2
plot_comparison(3, filter_oneOverC=None)

# Plot theta for fw=1 only
# plot_comparison(2, filter_fw=1)

# Plot f for lambda=0.5, fw=0
plot_comparison(0, filter_lambda=None, filter_fw=None)

plt.show()
