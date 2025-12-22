import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

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
    ii = 0
    for lIdx, lamda in enumerate(lambdaArray):
        if lamda == 2:
            ETA_INF = 2.0  # קירוב של "אינסוף"
            STEPS = 200    # מספר צעדים
            H = ETA_INF / STEPS
        else:
            ETA_INF = 10  # קירוב של "אינסוף"
            STEPS = 1000    # מספר צעדים
            H = ETA_INF / STEPS
        for fwIdx, fw_val in enumerate(fw):
            for cIdx, c in enumerate(oneOverC):
                LAMBDA = lamda
                C_PARAM_FINAL = c**(-2/3)
                C_PARAM = c**(1/3)
                initDThetaGuess = -initGuessArray[ii, 3]
                initDfGuess = initGuessArray[ii, 4]
                ii += 1
                fwTimesC = fw_val * C_PARAM
                # fwToUse = fwTimesC
                fwToUse = fw_val
                res = integrate_system(fwToUse, initDfGuess, initDThetaGuess)
                resArray.append(res)

                boundryValues = res[1][-1, :]
                fFinal = boundryValues[0]
                dfFinal = boundryValues[1]
                thetaFinal = boundryValues[2]
                dthetaFinal = boundryValues[3]

                fTagError = dfFinal - C_PARAM_FINAL
                thetaError = thetaFinal - TARGET_THETA

                errNorm = np.sqrt(fTagError**2 + thetaError**2)
                resDf = np.zeros([2])
                resDtheta = np.zeros([2])
                if errNorm <= EPS:
                    print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}")
                while errNorm > EPS and not (np.any(np.abs(resDf[1]) > 10) or np.any(np.abs(resDtheta[1]) > 10)):
                    delta = 1e-3
                    resDf = integrate_system(fwToUse, initDfGuess + delta, initDThetaGuess)
                    resDtheta = integrate_system(fwToUse, initDfGuess, initDThetaGuess + delta)

                    boundryDf_Df = resDf[1][-1, 1]
                    boundryDf_Theta = resDf[1][-1, 2]
                    boundryDtheta_Df = resDtheta[1][-1, 1]
                    boundryDtheta_Theta = resDtheta[1][-1, 2]

                    new_res = newton_raphson_solver(dfFinal, boundryDf_Df, boundryDf_Theta, thetaFinal, boundryDtheta_Df, boundryDtheta_Theta, delta, np.array([initDfGuess, initDThetaGuess]), np.array([fTagError, thetaError]))
                    
                    initDfGuess, initDThetaGuess = new_res
                    res = integrate_system(fwToUse, initDfGuess, initDThetaGuess)
                    resArray[ii-1] = res

                    boundryValues = res[1][-1, :]
                    fFinal = boundryValues[0]
                    dfFinal = boundryValues[1]
                    thetaFinal = boundryValues[2]
                    dthetaFinal = boundryValues[3]

                    fTagError = dfFinal - C_PARAM_FINAL
                    thetaError = thetaFinal - TARGET_THETA

                    errNorm = np.sqrt(fTagError**2 + thetaError**2)
                    # print(f"Iterating for Lambda={lamda}, fw={fw_val}, 1/C={c}, Error Norm={errNorm}")

                plt.figure()
                plt.plot(res[0], res[1][:, :])  # Plotting eta vs f
                plt.grid()
                plt.xlabel('Eta')
                plt.ylabel('f(eta)')
                plt.title(f'Lambda={lamda}, C^1/3={C_PARAM_FINAL}, fw={fw_val}')
                plt.legend(['f', "f'", 'theta', "theta'"])

    return resArray

data = [ # ערכי ההתחלה מהמאמר
    # --- Lambda = 0.5 ---
    # fw = -1
    [0.5, -1.0, 1.0, 0.8862, 1.8862],
    [0.5, -1.0, 2.0, 1.0450, 2.5547],
    [0.5, -1.0, 5.0, 1.3575, 4.1212],
    [0.5, -1.0, 8.0, 1.5724, 5.3921],
    # fw = 0
    [0.5, 0.0, 1.0, 1.1020, 1.7474],
    [0.5, 0.0, 2.0, 1.2495, 2.3479],
    [0.5, 0.0, 5.0, 1.5503, 3.7996],
    [0.5, 0.0, 8.0, 1.7610, 4.9990],
    # fw = 1
    [0.5, 1.0, 1.0, 1.3745, 1.6264],
    [0.5, 1.0, 2.0, 1.5041, 2.1591],
    [0.5, 1.0, 5.0, 1.7825, 3.4927],
    [0.5, 1.0, 8.0, 1.9836, 4.6181],

    # --- Lambda = 2.0 ---
    # fw = -1
    [2.0, -1.0, 1.0, 1.6309, 2.1235],
    [2.0, -1.0, 2.0, 1.9494, 2.9781],
    [2.0, -1.0, 5.0, 2.5716, 4.9815],
    [2.0, -1.0, 8.0, 2.9971, 6.6069],
    # fw = 0
    [2.0, 0.0, 1.0, 2.0044, 1.9159],
    [2.0, 0.0, 2.0, 2.2889, 2.6630],
    [2.0, 0.0, 5.0, 2.8820, 4.4981],
    [2.0, 0.0, 8.0, 3.2927, 6.0105],
    # fw = 1
    [2.0, 1.0, 1.0, 2.5182, 1.7391],
    [2.0, 1.0, 2.0, 2.7574, 2.3852],
    [2.0, 1.0, 5.0, 3.2827, 4.0268],
    [2.0, 1.0, 8.0, 3.6683, 5.4273]
]

initGuessArray = np.array(data)
# initGuessArray[:, 3:] -= 0.3
ii = 0  # index to select which set of parameters to run

# lambdaArray = np.asarray([0.5, 2])
lambdaArray = np.asarray([2])
fw = np.asarray([-1, 0, 1])
# fw = np.asarray([-1])
# oneOverC = np.asarray([1, 2, 5, 8])
oneOverC = np.asarray([1, 2, 5, 8])
# oneOverC = np.asarray([1])

resArray = []

# ערכי מטרה באינסוף
# TARGET_F_TAG = C_PARAM**2
TARGET_THETA = 0.0

lambdaLabels = {0.5: '0.5', 2.0: '2.0'}
cLabels = {1: '1', 2: '2', 5: '5', 8: '8'}
labels = ['f(eta)', "f '(eta)", 'theta(eta)', "theta '(eta)"]

resArray = shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5)

for idx, label in enumerate(labels):
    plt.figure()
    plt.grid()
    for i, run in enumerate(resArray):
        eta_vals = run[0]
        f_vals = run[1][:, idx]
        plt.plot(eta_vals, f_vals, label=f'Run {i+1}'+label)
        plt.xlabel('Eta')
        plt.ylabel(label)
    plt.legend()
            
plt.show()
