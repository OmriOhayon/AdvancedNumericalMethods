import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def calc_ddf(lamda, eta, dtheta, theta):
    ddf = -((lamda - 2)/3 * eta * dtheta + lamda * theta)
    return ddf

def calc_ddtheta(lamda, df, theta, dtheta, f):
    ddtheta = lamda * df * theta - (lamda + 1)/3 * f * dtheta
    return ddtheta

def derivatives(eta, y):
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
    k1 = fun(eta, y)
    k2 = fun(eta + 0.5*h, y + 0.5*h*k1)
    k3 = fun(eta + 0.5*h, y + 0.5*h*k2)
    k4 = fun(eta + h, y + h*k3)

    y += (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return y 

def integrate_system(fw, df_guess, dtheta_guess):
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
    J = np.zeros((2, 2)) # הגדרת יעקוביאן לניוטון רפסון
    J[0, 0] = (x2Df - x1) / delta  # df_final לפי df_guess
    J[1, 0] = (x2Theta - y1) / delta  # theta_final לפי dtheta_guess

    J[0, 1] = (y2Df - x1) / delta  # df_final לפי dtheta_guess
    J[1, 1] = (y2Theta - y1) / delta
    
    try:
        update = np.linalg.solve(J, errors)
        guess = initGuess - update
        return guess
    except np.linalg.LinAlgError:
        print("Singular Matrix encountered. Try different initial guesses.")
        return initGuess

def shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-3):
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
                # Calculate correct index into data table
                lambda_idx_full = full_lambda.index(lamda)
                fw_idx_full = full_fw.index(fw_val)
                c_idx_full = full_oneOverC.index(c)
                ii = lambda_idx_full * 12 + fw_idx_full * 4 + c_idx_full
                
                ETA_INF = initGuessArray[ii, 5] * 1.3
                # ETA_INF = 6.0
                H = 0.01
                STEPS = int(ETA_INF / H)
                # STEPS = 200
                LAMBDA = lamda
                C_PARAM_FINAL = c**(-2/3)
                C_PARAM = c**(1/3)
                initDThetaGuess = -initGuessArray[ii, 3]
                initDfGuess = initGuessArray[ii, 4]
                fwTimesC = fw_val * C_PARAM
                # fwToUse = fwTimesC
                fwToUse = fw_val
                res = integrate_system(fwToUse, initDfGuess, initDThetaGuess)
                resArray.append(res)
                paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})

                boundryValues = res[1][-1, :]
                fFinal = boundryValues[0]
                dfFinal = boundryValues[1]
                thetaFinal = boundryValues[2]
                dthetaFinal = boundryValues[3]

                fTagError = dfFinal - C_PARAM_FINAL
                thetaError = thetaFinal - TARGET_THETA

                errNorm = np.sqrt(fTagError**2 + thetaError**2)
                
                if errNorm <= EPS:
                    print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}. Error Norm={errNorm}")
                    continue  # Skip to next iteration, already converged
                
                # Initialize divergence check - will be updated in first loop iteration
                divergence_detected = False
                
                while errNorm > EPS and not divergence_detected:
                    delta = 1e-1
                    resDf = integrate_system(fwToUse, initDfGuess + delta, initDThetaGuess)
                    resDtheta = integrate_system(fwToUse, initDfGuess, initDThetaGuess + delta)

                    boundryDf_Df = resDf[1][-1, 1]
                    boundryDf_Theta = resDf[1][-1, 2]
                    boundryDtheta_Df = resDtheta[1][-1, 1]
                    boundryDtheta_Theta = resDtheta[1][-1, 2]
                    
                    # Check for divergence
                    if (np.any(np.abs(resDf[1][-1, :]) > 10) or np.any(np.abs(resDtheta[1][-1, :]) > 10)):
                        print(f"Divergence detected for Lambda={lamda}, fw={fw_val}, C={c}")
                        divergence_detected = True
                        break

                    new_res = newton_raphson_solver(dfFinal, boundryDf_Df, boundryDf_Theta, thetaFinal, boundryDtheta_Df, boundryDtheta_Theta, delta, np.array([initDfGuess, initDThetaGuess]), np.array([fTagError, thetaError]))
                    
                    initDfGuess, initDThetaGuess = new_res
                    res = integrate_system(fwToUse, initDfGuess, initDThetaGuess)
                    resArray[-1] = res

                    boundryValues = res[1][-1, :]
                    fFinal = boundryValues[0]
                    dfFinal = boundryValues[1]
                    thetaFinal = boundryValues[2]
                    dthetaFinal = boundryValues[3]

                    fTagError = dfFinal - C_PARAM_FINAL
                    thetaError = thetaFinal - TARGET_THETA

                    errNorm = np.sqrt(fTagError**2 + thetaError**2)
                    if errNorm <= EPS:
                        print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}. Final Error Norm={errNorm}")
                    # print(f"Iterating for Lambda={lamda}, fw={fw_val}, 1/C={c}, Error Norm={errNorm}")

                # plt.figure()
                # plt.plot(res[0], res[1][:, :])  # Plotting eta vs f
                # plt.grid()
                # plt.xlabel('Eta')
                # plt.ylabel('f(eta)')
                # plt.title(f'Lambda={lamda}, C^1/3={C_PARAM_FINAL}, fw={fw_val}')
                # plt.legend(['f', "f'", 'theta', "theta'"])

    return resArray, paramsArray

# def finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5):
#     global ETA_INF, STEPS, H, LAMBDA, MAX_ITER
#     MAX_ITER = 10000
#     resArray = []
#     ii = 0

#     for lIdx, lamda in enumerate(lambdaArray):
#         for fwIdx, fw_val in enumerate(fw):
#             for cIdx, c in enumerate(oneOverC):

#                 ETA_INF = initGuessArray[ii, 5] * 1
#                 ii += 1

#                 # ---- FIX 1: consistent grid with H ----
#                 H = 0.01
#                 N = int(round(ETA_INF / H))
#                 if N < 4:
#                     # either bump ETA_INF or reduce H; pick one:
#                     N = 4
#                     ETA_INF = N * H                    # keep H constant, enlarge domain
                
#                 eta = np.linspace(0.0, ETA_INF, N + 1)
#                 H = eta[1] - eta[0]           # enforce consistency
#                 STEPS = N + 1                 # keep your global name

#                 LAMBDA = lamda

#                 # You treat input as oneOverC = 1/C
#                 C_PARAM_FINAL = c**(-2/3)     # = C^(2/3) (far-field f')
#                 C_PARAM = c**(1/3)            # = C^(-1/3) (mapping for f_w)
#                 fwToUse = fw_val * C_PARAM    # f(0)

#                 theta = np.exp(-eta)
#                 theta[0] = 1.0
#                 theta[-1] = 0.0

#                 # better initial guess for f: match far-field slope
#                 f = fwToUse + C_PARAM_FINAL * eta
#                 f[0] = fwToUse

#                 omega_theta = 0.8
#                 omega_f = 0.4   # 0.1 is unnecessarily tiny; keep it stable but not glacial

#                 inv_h2 = 1.0 / (H * H)
#                 inv_2h = 1.0 / (2.0 * H)

#                 def enforce_bc_f():
#                     f[0] = fwToUse
#                     # 2nd-order backward: (3 f_N - 4 f_{N-1} + f_{N-2})/(2H) = C^(2/3)
#                     f[-1] = (4.0 * f[-2] - f[-3] + 2.0 * H * C_PARAM_FINAL) / 3.0

#                 def fp_inf_from_f():
#                     # consistent with the BC stencil above
#                     return (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) * inv_2h

#                 enforce_bc_f()

#                 converged = False
#                 for k in range(MAX_ITER):
#                     theta_old = theta.copy()
#                     f_old = f.copy()

#                     # ---- GS sweep 1: update f using current theta ----
#                     enforce_bc_f()
#                     for i in range(1, STEPS - 1):
#                         # theta' central using current theta values
#                         dtheta_i = (theta[i + 1] - theta[i - 1]) * inv_2h

#                         # from eq (8): f'' = -((λ-2)/3 * η * θ' + λ θ)
#                         # rearranged update: f_i = 0.5*(f_{i-1}+f_{i+1} + H^2*((λ-2)/3*η*θ' + λ θ))
#                         src = ((LAMBDA - 2.0) / 3.0) * eta[i] * dtheta_i + LAMBDA * theta[i]
#                         val_f = 0.5 * (f[i - 1] + f[i + 1] + (H * H) * src)

#                         f[i] = (1.0 - omega_f) * f[i] + omega_f * val_f

#                     enforce_bc_f()

#                     # ---- GS sweep 2: update theta using updated f ----
#                     theta[0] = 1.0
#                     theta[-1] = 0.0
#                     for i in range(1, STEPS - 1):
#                         # f' central using *updated* f values
#                         df_i = (f[i + 1] - f[i - 1]) * inv_2h

#                         # eq (9) -> theta'' + p theta' + q theta = 0
#                         p = ((LAMBDA + 1.0) / 3.0) * f[i]
#                         q = -LAMBDA * df_i

#                         coeff_minus = inv_h2 - p * inv_2h
#                         coeff_plus  = inv_h2 + p * inv_2h
#                         denom = -2.0 * inv_h2 + q

#                         # avoid divide-by-zero / blow-ups
#                         if denom == 0.0:
#                             denom = np.copysign(1e-300, denom)

#                         val = -(coeff_minus * theta[i - 1] + coeff_plus * theta[i + 1]) / denom
#                         theta[i] = (1.0 - omega_theta) * theta[i] + omega_theta * val

#                     theta[0] = 1.0
#                     theta[-1] = 0.0

#                     err_theta = np.max(np.abs(theta - theta_old))
#                     err_f = np.max(np.abs(f - f_old))

#                     if k % 100 == 0:
#                         print(f"Iter {k}: Theta Error={err_theta:.2e}, f'(inf)={fp_inf_from_f():.6f}")

#                     if max(err_theta, err_f) < EPS:
#                         print(f"  Converged in {k} iterations.")
#                         res = [eta, np.asarray([f,
#                                                np.gradient(f, H),
#                                                theta,
#                                                np.gradient(theta, H)])]
#                         resArray.append(res)
#                         converged = True
#                         break

#                 if not converged:
#                     print("Did not converge.")
#                     res = [eta, np.asarray([f,
#                                            np.gradient(f, H),
#                                            theta,
#                                            np.gradient(theta, H)])]
#                     resArray.append(res)

#     return resArray

# def finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5):
#     # הסרת global - עדיף להימנע מזה בתוך פונקציות
#     MAX_ITER = 10000
#     resArray = []
#     ii = 0
    
#     for lIdx, lamda in enumerate(lambdaArray):
#         for fwIdx, fw_val in enumerate(fw):
#             for cIdx, c in enumerate(oneOverC):
                
#                 # --- תיקון 1: הגדרת רשת ו-H מדויק ---
#                 ETA_INF = initGuessArray[ii, 5] * 1.0 # וודא שזה float
#                 ii += 1
                
#                 # אנו רוצים H בקירוב 0.01. נחשב מספר צעדים שלם שיתן את זה
#                 DESIRED_H = 0.001
#                 STEPS = int(np.ceil(ETA_INF / DESIRED_H)) + 1 
                
#                 # יצירת הרשת וקבלת ה-H האמיתי
#                 eta, H = np.linspace(0, ETA_INF, STEPS, retstep=True)
                
#                 # --- המרת פרמטרים ---
#                 LAMBDA = lamda
#                 C_PARAM_FINAL = c**(-2/3) # השיפוע באינסוף f'
#                 C_PARAM = c**(1/3)
#                 fwTimesC = fw_val * C_PARAM
#                 fwToUse = fwTimesC
                
#                 # --- אתחול משתנים ---
#                 theta = np.exp(-eta)
                
#                 # --- תיקון 2: ניחוש התחלתי חכם יותר ל-f ---
#                 # מתחיל ב-fw ומסתיים בשיפוע הנכון
#                 f = fwToUse + eta * C_PARAM_FINAL 
                
#                 omega_theta = 0.8 
#                 omega_f = 0.1 # רלקסציה נמוכה ל-f זה מצוין
#                 converged = False
                
#                 print(f"Start: Lambda={LAMBDA}, fw={fwToUse:.2f}, Target f'={C_PARAM_FINAL:.4f}")

#                 for k in range(MAX_ITER):
#                     theta_old = theta.copy()
#                     f_old = f.copy()
                    
#                     # חישוב נגזרות "ישנות" (Lagging coefficients)
#                     # השימוש ב-gradient של numpy בסדר גמור כאן
#                     dtheta = np.gradient(theta, H)
                    
#                     # חישוב RHS של משוואת התנע (ddf)
#                     # משוואה 8: f'' = -((lambda-2)/3 * eta * theta' + lambda * theta)
#                     # הערה: זה מחושב פעם אחת לאיטרציה (מחוץ ללולאת i) לטובת ביצועים
#                     rhs_f_vector = -((LAMBDA - 2)/3 * eta * dtheta + LAMBDA * theta)
                    
#                     # נגזרת של f (דרושה למשוואת האנרגיה)
#                     df = np.gradient(f, H)
                    
#                     # --- לולאה פנימית (Gauss-Seidel Sweep) ---
#                     for i in range(1, STEPS-1):
                        
#                         # A. עדכון Theta
#                         p = (LAMBDA + 1)/3 * f[i]
#                         q = -LAMBDA * df[i]
                        
#                         coeff_minus = 1/H**2 - p/(2*H)
#                         coeff_plus = 1/H**2 + p/(2*H)
#                         denom = -2/H**2 + q
                        
#                         # שימוש ב-theta[i-1] (החדש) וב-theta[i+1] (הישן) - זהו GS קלאסי
#                         val_theta = -(coeff_minus * theta[i-1] + coeff_plus * theta[i+1]) / denom
#                         theta[i] = (1 - omega_theta) * theta[i] + omega_theta * val_theta
                        
#                         # B. עדכון f
#                         # משוואת פואסון: (f_left + f_right - h^2*RHS) / 2
#                         val_f = 0.5 * (f[i-1] + f[i+1] - H**2 * rhs_f_vector[i])
#                         f[i] = (1 - omega_f) * f[i] + omega_f * val_f

#                     # --- תנאי שפה ---
#                     theta[0] = 1.0
#                     theta[-1] = 0.0
#                     f[0] = fwToUse
                    
#                     # תנאי שפה לנגזרת f באינסוף (Neumann)
#                     # f_N = (4*f_{N-1} - f_{N-2} + 2*h*Target) / 3
#                     f_target_val = (4*f[-2] - f[-3] + 2*H*C_PARAM_FINAL) / 3.0
#                     f[-1] = (1 - omega_f) * f[-1] + omega_f * f_target_val

#                     # --- בדיקת התכנסות ---
#                     err_theta = np.max(np.abs(theta - theta_old))
#                     err_f = np.max(np.abs(f - f_old))
                    
#                     if k % 1000 == 0:
#                          print(f"  Iter {k}: Err={max(err_theta, err_f):.1e}, f'(inf)={ (f[-1]-f[-2])/H :.4f}")

#                     if max(err_theta, err_f) < EPS:
#                         print(f"  Converged in {k} iterations.")
                        
#                         # --- תיקון 3: חישוב נגזרות סופיות לפני שמירה ---
#                         final_df = np.gradient(f, H)
#                         final_dtheta = np.gradient(theta, H)
                        
#                         res = [eta, np.asarray([f, final_df, theta, final_dtheta])]
#                         resArray.append(res)
#                         converged = True
#                         break

#                 if not converged:
#                     print("Did not converge.")
#                     final_df = np.gradient(f, H)
#                     final_dtheta = np.gradient(theta, H)
#                     resArray.append([eta, np.asarray([f, final_df, theta, final_dtheta])])
    
#     return resArray

# # def finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5):
#     global ETA_INF, STEPS, H, LAMBDA, MAX_ITER
#     MAX_ITER = 10000
#     resArray = []
#     ii = 0
#     for lIdx, lamda in enumerate(lambdaArray):
#         for fwIdx, fw_val in enumerate(fw):
#             for cIdx, c in enumerate(oneOverC):
#                 ETA_INF = initGuessArray[ii, 5] * 4
#                 ii += 1
#                 H = 0.01
#                 STEPS = int(ETA_INF / H)
#                 LAMBDA = lamda
#                 C_PARAM_FINAL = c**(-2/3)
#                 C_PARAM = c**(1/3)
#                 fwTimesC = fw_val * C_PARAM
#                 fwToUse = fwTimesC
#                 eta = np.linspace(0, ETA_INF, STEPS)
                
#                 theta = np.exp(-eta)
#                 f = eta.copy() + fw_val/2
                
#                 omega_theta = 0.8 
#                 omega_f = 0.8
#                 converged = False
#                 for k in range(MAX_ITER):
#                     theta_old = theta.copy()
#                     f_old = f.copy()
                    
#                     dtheta = np.gradient(theta, H)
#                     df = np.gradient(f, H)
#                     ddf = -((LAMBDA - 2)/3 * eta * dtheta + LAMBDA * theta)
                    
#                     # total_integral_fpp = np.trapz(ddf, dx=H)
#                     # c1 = C_PARAM_FINAL - total_integral_fpp
                    
#                     # f_prime = integrate.cumulative_trapezoid(ddf, eta, initial=0) + c1
                    
#                     # f = integrate.cumulative_trapezoid(f_prime, eta, initial=0) + fwToUse
                    
#                     # df = f_prime
                    
#                     for i in range(1, STEPS-1):
#                         p = (LAMBDA + 1)/3 * f[i]
#                         q = -LAMBDA * df[i]
                        
#                         coeff_minus = 1/H**2 - p/(2*H)
#                         coeff_plus = 1/H**2 + p/(2*H)
#                         denom = -2/H**2 + q
                        
#                         val = -(coeff_minus * theta[i-1] + coeff_plus * theta[i+1]) / denom
                        
#                         theta[i] = (1 - omega_theta) * theta[i] + omega_theta * val

                        
#                         val_f = 1/2 * (f[i-1] + f[i+1] - H**2 * ddf[i])
#                         f[i] = (1 - omega_f) * f[i] + omega_f * val_f  

#                     theta[0] = 1.0
#                     theta[-1] = 0.0
#                     f[0] = fwToUse

#                     f_target_val = (4*f[-2] - f[-3] + 2*H*C_PARAM_FINAL) / 3.0
                    
#                     # עדכון הנקודה האחרונה (גם כאן כדאי להשתמש ברלקסציה)
#                     f[-1] = (1 - omega_f) * f[-1] + omega_f * f_target_val

#                     err_theta = np.max(np.abs(theta - theta_old))
#                     err_f = np.max(np.abs(f - f_old))
                    
#                     if k % 100 == 0:
#                         print(f"Iter {k}: Theta Error={err_theta:.2e}, f'(inf)={df[-1]:.4f}")

#                     if max(err_theta, err_f) < EPS:
#                         print(f"  Converged in {k} iterations.")
#                         res = [eta, np.asarray([f, np.gradient(f,H), theta, np.gradient(theta,H)])]
#                         resArray.append(res)
#                         converged = True
#                         break

#                 if not converged:
#                     print("Did not converge.")
#                     res = [eta, np.asarray([f, df, theta, dtheta])]
#                     resArray.append(res)
    
#     return resArray

def finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5):
    global ETA_INF, STEPS, H, LAMBDA, MAX_ITER
    MAX_ITER = 10000
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
                
                ETA_INF = initGuessArray[ii, 5] * 4.0

                # ---- FIX: consistent grid ----
                H = 0.01
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

                    if k % 100 == 0:
                        fp_inf_est = (3*f[-1] - 4*f[-2] + f[-3]) / (2.0 * H)
                        print(f"Iter {k}: Theta Error={err_theta:.2e}, f'(inf)={fp_inf_est:.6f}, STEPS={STEPS}")

                    if max(err_theta, err_f) < EPS:
                        print(f"  Converged in {k} iterations.")
                        dtheta_out = d1_center(theta, H)
                        resArray.append([eta, np.asarray([f, fp, theta, dtheta_out])])
                        paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})
                        converged = True
                        break

                if not converged:
                    print("Did not converge.")
                    dtheta_out = d1_center(theta, H)
                    resArray.append([eta, np.asarray([f, fp, theta, dtheta_out])])
                    paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})

    return resArray, paramsArray

def gauss_seidel_method():
    pass

data = [ # ערכי ההתחלה מהמאמר
    # --- Lambda = 0.5 ---
    # fw = -1
    [0.5, -1.0, 1.0, 0.8862, 1.8862, 3.2154],
    [0.5, -1.0, 2.0, 1.0450, 2.5547, 2.9088],
    [0.5, -1.0, 5.0, 1.3575, 4.1212, 2.4379],
    [0.5, -1.0, 8.0, 1.5724, 5.3921, 1.8704],
    # fw = 0
    [0.5, 0.0, 1.0, 1.1020, 1.7474, 2.8250],
    [0.5, 0.0, 2.0, 1.2495, 2.3479, 2.6049],
    [0.5, 0.0, 5.0, 1.5503, 3.7996, 2.2380],
    [0.5, 0.0, 8.0, 1.7610, 4.9990, 2.0340],
    # fw = 1
    [0.5, 1.0, 1.0, 1.3745, 1.6264, 2.4624],
    [0.5, 1.0, 2.0, 1.5041, 2.1591, 2.3115],
    [0.5, 1.0, 5.0, 1.7825, 3.4927, 2.0347],
    [0.5, 1.0, 8.0, 1.9836, 4.6181, 1.8688],

    # --- Lambda = 2.0 ---
    # fw = -1
    [2.0, -1.0, 1.0, 1.6309, 2.1235, 2.2138],
    [2.0, -1.0, 2.0, 1.9494, 2.9781, 2.0284],
    [2.0, -1.0, 5.0, 2.5716, 4.9815, 1.7304],
    [2.0, -1.0, 8.0, 2.9971, 6.6069, 1.5729],
    # fw = 0, 3.2154
    [2.0, 0.0, 1.0, 2.0044, 1.9159, 1.8532],
    [2.0, 0.0, 2.0, 2.2889, 2.6630, 1.7334],
    [2.0, 0.0, 5.0, 2.8820, 4.4981, 1.5188],
    [2.0, 0.0, 8.0, 3.2927, 6.0105, 1.3954],
    # fw = 1
    [2.0, 1.0, 1.0, 2.5182, 1.7391, 1.5371],
    [2.0, 1.0, 2.0, 2.7574, 2.3852, 1.4656],
    [2.0, 1.0, 5.0, 3.2827, 4.0268, 1.3230],
    [2.0, 1.0, 8.0, 3.6683, 5.4273, 1.2327]
]

dataOptimized = [ # ערכי ההתחלה מהמאמר
    # --- Lambda = 0.5 ---
    # fw = -1
    [0.5, -1.0, 1.0, 0.8862, 1.8862, 3.2154],
    [0.5, -1.0, 2.0, 1.0450-0.1, 2.5547-0.1, 2.9088],
    [0.5, -1.0, 5.0, 1.3575, 4.1212, 2.4379],
    [0.5, -1.0, 8.0, 1.5724, 5.3921, 1.8704],
    # fw = 0
    [0.5, 0.0, 1.0, 1.1020, 1.7474, 2.8250],
    [0.5, 0.0, 2.0, 1.2495, 2.3479, 2.6049],
    [0.5, 0.0, 5.0, 1.5503-2.2, 3.7996-2.2, 2.2380],
    [0.5, 0.0, 8.0, 1.7610-2.2, 4.9990-2.2, 2.0340],
    # fw = 1
    [0.5, 1.0, 1.0, 1.3745, 1.6264, 2.4624],
    [0.5, 1.0, 2.0, 1.5041, 2.1591, 2.3115],
    [0.5, 1.0, 5.0, 1.7825-3, 3.4927-3, 2.0347],
    [0.5, 1.0, 8.0, 1.9836-3, 4.6181-3, 1.8688],

    # --- Lambda = 2.0 ---
    # fw = -1
    [2.0, -1.0, 1.0, 1.6309, 2.1235, 2.2138],
    [2.0, -1.0, 2.0, 1.9494, 2.9781, 2.0284],
    [2.0, -1.0, 5.0, 2.5716+0.35, 4.9815, 1.7304],
    [2.0, -1.0, 8.0, 2.9971, 6.6069, 1.5729],
    # fw = 0, 3.2154
    [2.0, 0.0, 1.0, 2.0044, 1.9159, 1.8532],
    [2.0, 0.0, 2.0, 2.2889, 2.6630, 1.7334],
    [2.0, 0.0, 5.0, 2.8820, 4.4981, 1.5188],
    [2.0, 0.0, 8.0, 3.2927, 6.0105, 1.3954],
    # fw = 1
    [2.0, 1.0, 1.0, 2.5182, 1.7391, 1.5371],
    [2.0, 1.0, 2.0, 2.7574, 2.3852, 1.4656],
    [2.0, 1.0, 5.0, 3.2827, 4.0268, 1.3230],
    [2.0, 1.0, 8.0, 3.6683, 5.4273, 1.2327]
]

# initGuessArray = np.array(data)
initGuessArray = np.array(dataOptimized)
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
oneOverC = np.asarray([1, 2, 5, 8])
# oneOverC = np.asarray([2, 5, 8])
# oneOverC = np.asarray([5, 8])
# oneOverC = np.asarray([8])

resArray = []

# ערכי מטרה באינסוף
# TARGET_F_TAG = C_PARAM**2
TARGET_THETA = 0.0

lambdaLabels = {0.5: '0.5', 2.0: '2.0'}
cLabels = {1: '1', 2: '2', 5: '5', 8: '8'}
labels = ['f(eta)', "f '(eta)", 'theta(eta)', "theta '(eta)"]

# resArray, paramsArray = shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5)
resArray, paramsArray = finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-4)

for idx, label in enumerate(labels):
    # if idx != 1:
    #     continue
    plt.figure()
    plt.grid()
    for i, run in enumerate(resArray):
        eta_vals = run[0]
        f_vals = run[1][idx, :] # For finite differences method
        # f_vals = run[1][:, idx]   # For shooting method
        params = paramsArray[i]
        legend_label = f"λ={params['lambda']}, fw={params['fw']}, 1/C={params['oneOverC']}"
        plt.plot(eta_vals, f_vals, label=legend_label)
        plt.xlabel('Eta')
        plt.ylabel(label)
    plt.legend()
            
plt.show()
