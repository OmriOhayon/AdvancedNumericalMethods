import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

# --- פרמטרים פיזיקליים (לפי המאמר) ---
LAMBDA = 0.5   # מקרה של שטף חום קבוע (Constant Heat Flux)
F_W = 0.0      # פלטה אטומה (ללא יניקה/הזרקה)
ETA_MAX = 10.0 # "אינסוף" נומרי
F_PRIME_INF = 0.0 # עבור קונבקציה טבעית טהורה (C=0 -> f'(inf)=0)

# --- פרמטרים נומריים ---
N = 100        # מספר נקודות רשת
H = ETA_MAX / (N - 1)
TOL = 1e-3     # סף התכנסות
MAX_ITER = 2000 # מקסימום איטרציות

# ==========================================
# 1. שיטת ה-Shooting Method
# ==========================================
def solve_shooting():
    """
    פתרון באמצעות הפיכת הבעיה לבעיית תנאי התחלה וניחוש הערכים החסרים.
    וקטור המצב: y = [f, f', theta, theta']
    """
    def system(eta, y):
        f, df, theta, dtheta = y
        
        # משוואה 8: f''
        ddf = -((LAMBDA - 2)/3 * eta * dtheta + LAMBDA * theta)
        
        # משוואה 9: theta''
        ddtheta = LAMBDA * df * theta - (LAMBDA + 1)/3 * f * dtheta
        
        return [df, ddf, dtheta, ddtheta]

    # פונקציית המטרה לאופטימיזציה (השאריות באינסוף)
    def residuals(guess):
        # הניחושים החסרים הם f'(0) ו-theta'(0)
        # שימו לב: f(0)=f_w, theta(0)=1 נתונים
        df0_guess, dtheta0_guess = guess
        
        y0 = [F_W, df0_guess, 1.0, dtheta0_guess]
        
        sol = solve_ivp(system, [0, ETA_MAX], y0, t_eval=[ETA_MAX], rtol=1e-6)
        
        # אנו רוצים ש-f'(inf) ו-theta(inf) יתקיימו
        f_prime_end = sol.y[1][-1]
        theta_end = sol.y[2][-1]
        
        return [f_prime_end - F_PRIME_INF, theta_end - 0.0]

    # ניחוש ראשוני [f'(0), theta'(0)]
    initial_guess = [1.0, -0.8] 
    
    # מציאת השורשים
    res = root(residuals, initial_guess)
    
    # הרצת הפתרון הסופי
    final_guesses = res.x
    y0_final = [F_W, final_guesses[0], 1.0, final_guesses[1]]
    sol_final = solve_ivp(system, [0, ETA_MAX], y0_final, t_eval=np.linspace(0, ETA_MAX, N))
    
    return sol_final.t, sol_final.y[0], sol_final.y[2]

# ==========================================
# 2. שיטת הפרשים סופיים עם אלגוריתם תומאס (TDMA)
# ==========================================
def tdma_solver(a, b, c, d):
    """ פותר מערכת טרי-דיאגונלית Ax=d """
    n = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) 
    for i in range(1, n):
        mc = ac[i-1] / bc[i-1]
        bc[i] = bc[i] - mc * cc[i-1]
        dc[i] = dc[i] - mc * dc[i-1]
        
    xc = bc
    xc[-1] = dc[-1] / bc[-1]
    
    for i in range(n-2, -1, -1):
        xc[i] = (dc[i] - cc[i] * xc[i+1]) / bc[i]
    return xc

def solve_fdm_thomas():
    eta = np.linspace(0, ETA_MAX, N)
    
    # אתחול ניחושים ראשוניים
    f = eta.copy() # ניחוש לינארי לזרימה
    theta = np.exp(-eta) # ניחוש דועך לטמפרטורה
    
    for k in range(MAX_ITER):
        # --- שלב א: פתרון משוואת האנרגיה עבור Theta בעזרת TDMA ---
        # נגזרות של f (שימוש ב-Central Difference)
        df = np.gradient(f, H)
        
        # בניית המטריצה
        # משוואה: theta'' - lambda*f'*theta + (lambda+1)/3 * f * theta' = 0
        # דיסקרטיזציה: A*theta_{i-1} + B*theta_i + C*theta_{i+1} = 0
        
        lower = np.zeros(N) # אלכסון תחתון (a)
        main  = np.zeros(N) # אלכסון ראשי (b)
        upper = np.zeros(N) # אלכסון עליון (c)
        rhs   = np.zeros(N) # צד ימין (d)
        
        # תנאי שפה ב-i=0
        main[0] = 1.0; rhs[0] = 1.0
        
        # נקודות פנימיות
        for i in range(1, N-1):
            p = (LAMBDA + 1)/3 * f[i]
            q = -LAMBDA * df[i]
            
            lower[i] = 1/H**2 - p/(2*H)
            main[i]  = -2/H**2 + q
            upper[i] = 1/H**2 + p/(2*H)
            rhs[i]   = 0
            
        # תנאי שפה ב-i=N-1 (באינסוף)
        main[-1] = 1.0; rhs[-1] = 0.0
        
        theta_new = tdma_solver(lower[1:], main, upper[:-1], rhs)
        
        # --- שלב ב: עדכון f בעזרת אינטגרציה ---
        # f'' = RHS(theta)
        dtheta = np.gradient(theta_new, H)
        f_dbl_prime = -((LAMBDA - 2)/3 * eta * dtheta + LAMBDA * theta_new)
        
        # אינטגרציה ראשונה למציאת f'
        # f'(eta) = int(f'') + C1. אנו יודעים f'(inf) = F_PRIME_INF
        # C1 = f'(inf) - int_0^inf(f'')
        total_integral = np.trapz(f_dbl_prime, dx=H)
        c1 = F_PRIME_INF - total_integral
        
        f_prime = np.zeros(N)
        for i in range(N):
            f_prime[i] = np.trapz(f_dbl_prime[:i+1], dx=H) + c1
            
        # אינטגרציה שניה למציאת f
        # f(eta) = int(f') + f_w
        f_new = np.zeros(N)
        for i in range(N):
            f_new[i] = np.trapz(f_prime[:i+1], dx=H) + F_W
            
        # בדיקת התכנסות
        if np.max(np.abs(theta_new - theta)) < TOL:
            print(f"TDMA converged in {k} iterations")
            return eta, f_new, theta_new
            
        theta = theta_new
        f = f_new

    print("TDMA did not converge")
    return eta, f, theta

# ==========================================
# 3. שיטת גאוס-זיידל (Gauss-Seidel)
# ==========================================
def solve_gauss_seidel():
    eta = np.linspace(0, ETA_MAX, N)
    f = eta.copy()
    theta = np.exp(-eta)
    
    omega = 1.0 # פקטור רלקסציה (1.0 = ללא האצה)
    
    for k in range(MAX_ITER*2): # דורש יותר איטרציות בד"כ
        max_diff = 0.0
        
        df = np.gradient(f, H)
        
        # --- עדכון Theta ---
        theta_old_iter = theta.copy()
        for i in range(1, N-1):
            # חילוץ theta_i ממשוואת ההפרשים
            p = (LAMBDA + 1)/3 * f[i]
            q = -LAMBDA * df[i]
            
            coeff_minus = 1/H**2 - p/(2*H)
            coeff_plus  = 1/H**2 + p/(2*H)
            denom       = -2/H**2 + q
            
            # שימוש ב-theta[i-1] המעודכן (Gauss-Seidel)
            val = -(coeff_minus * theta[i-1] + coeff_plus * theta[i+1]) / denom
            
            theta[i] = (1-omega)*theta[i] + omega*val
            
        # תנאי שפה Theta
        theta[0] = 1.0
        theta[-1] = 0.0
        
        # --- עדכון f ---
        # נפתור את f'' = RHS כמשוואת פואסון: (f_{i+1}-2f_i+f_{i-1})/h^2 = RHS_i
        # f_i = 0.5 * (f_{i+1} + f_{i-1} - h^2 * RHS_i)
        
        dtheta = np.gradient(theta, H)
        rhs_f = -((LAMBDA - 2)/3 * eta * dtheta + LAMBDA * theta)
        
        f_old_iter = f.copy()
        for i in range(1, N-1):
            val_f = 0.5 * (f[i+1] + f[i-1] - H**2 * rhs_f[i])
            f[i] = (1-omega)*f[i] + omega*val_f
            
        # תנאי שפה f
        f[0] = F_W
        # תנאי שפה נגזרת באינסוף: f_N = f_{N-1} + h * f'(inf)
        f[-1] = f[-2] + H * F_PRIME_INF
        
        # בדיקת שגיאה
        diff_theta = np.max(np.abs(theta - theta_old_iter))
        diff_f = np.max(np.abs(f - f_old_iter))
        
        if max(diff_theta, diff_f) < TOL:
            print(f"Gauss-Seidel converged in {k} iterations")
            return eta, f, np.gradient(f, H), theta
            
    print("Gauss-Seidel did not converge")
    return eta, f, np.gradient(f, H), theta

# ==========================================
# Main Execution & Plotting
# ==========================================
# הרצת הפתרונות
# eta_sh, f_sh, t_sh = solve_shooting()
# eta_td, f_td, t_td = solve_fdm_thomas()
eta_gs, f_gs, df_gs, t_gs = solve_gauss_seidel()

# יצירת גרפים
plt.figure(figsize=(12, 5))

# גרף טמפרטורה
plt.subplot(1, 2, 1)
# plt.plot(eta_sh, t_sh, 'k-', linewidth=4, alpha=0.3, label='Shooting (Reference)')
# plt.plot(eta_td, t_td, 'r--', label='TDMA')
plt.plot(eta_gs, t_gs, 'b:', label='Gauss-Seidel')
plt.title(f'Temperature Profile (theta)\nLambda={LAMBDA}, fw={F_W}')
plt.xlabel('eta')
plt.ylabel('theta')
plt.legend()
plt.grid(True)

# גרף זרימה
plt.subplot(1, 2, 2)
# plt.plot(eta_sh, f_sh, 'k-', linewidth=4, alpha=0.3, label='Shooting (Reference)')
# plt.plot(eta_td, f_td, 'r--', label='TDMA')
plt.plot(eta_gs, df_gs, 'b:', label='Gauss-Seidel')
plt.title(f'Stream Function (f)\nLambda={LAMBDA}, fw={F_W}')
plt.xlabel('eta')
plt.ylabel('f')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# הדפסת השוואה של ערך הנגזרת על הקיר (מדד למעבר חום)
# dtheta_sh = (t_sh[1] - t_sh[0]) / (eta_sh[1] - eta_sh[0])
# dtheta_td = (t_td[1] - t_td[0]) / H
dtheta_gs = (t_gs[1] - t_gs[0]) / H

print("\n--- Comparison of -theta'(0) (Heat Transfer Rate) ---")
# print(f"Shooting Method: {-dtheta_sh:.4f}")
# print(f"TDMA Method:     {-dtheta_td:.4f}")
print(f"Gauss-Seidel:    {-dtheta_gs:.4f}")

# פרמטרים
LAMBDA = 0.5
F_W = 0.0
ETA_MAX = 10.0
F_PRIME_INF = (1/2)**(2/3) # C^2 עבור קונבקציה טבעית
N = 100
H = ETA_MAX / (N - 1)
TOL = 1e-6
MAX_ITER = 5000

def solve_hybrid_gs():
    eta = np.linspace(0, ETA_MAX, N)
    
    # ניחושים התחלתיים
    theta = np.exp(-eta)
    f = eta.copy() # לא קריטי, יחושב מיד
    
    # פרמטר רלקסציה עבור Theta (עוזר להתכנסות במשוואות מצומדות)
    omega_theta = 0.8 
    
    for k in range(MAX_ITER):
        theta_old = theta.copy()
        
        # ---------------------------------------------------------
        # שלב 1: חישוב f ו-f' באמצעות אינטגרציה ישירה (מדויק ויציב)
        # ---------------------------------------------------------
        # מכיוון שמשוואה (8) תלויה רק ב-Theta, אין צורך ב-GS עבור f.
        # אנו מחשבים את f'' מה-Theta הנוכחי ומבצעים אינטגרציה.
        
        dtheta = np.gradient(theta, H)
        # חישוב צד ימין של משוואת התנע (RHS)
        f_dbl_prime = -((LAMBDA - 2)/3 * eta * dtheta + LAMBDA * theta)
        
        # אינטגרציה ראשונה: מציאת f'
        # f'(eta) = integral(f'') + C1
        # אנו יודעים ש-f'(inf) צריך להיות F_PRIME_INF.
        # לכן: C1 = F_PRIME_INF - integral_0^inf(f'')
        total_integral_fpp = np.trapz(f_dbl_prime, dx=H)
        c1 = F_PRIME_INF - total_integral_fpp
        
        # שימוש ב-cumulative sum לאינטגרציה מהירה (כמו cumtrapz)
        # f_prime[i] = int_0^eta_i (f'') + c1
        import scipy.integrate as integrate
        f_prime = integrate.cumulative_trapezoid(f_dbl_prime, eta, initial=0) + c1
        
        # אינטגרציה שניה: מציאת f
        # f(eta) = integral(f') + f(0). ידוע ש-f(0) = F_W
        f = integrate.cumulative_trapezoid(f_prime, eta, initial=0) + F_W
        
        # ---------------------------------------------------------
        # שלב 2: עדכון Theta באמצעות גאוס-זיידל (עם f המעודכן)
        # ---------------------------------------------------------
        df = f_prime # הנגזרת כבר חושבה במדויק למעלה
        
        for i in range(1, N-1):
            # מקדמים למשוואת האנרגיה
            p = (LAMBDA + 1)/3 * f[i]
            q = -LAMBDA * df[i]
            
            coeff_minus = 1/H**2 - p/(2*H)
            coeff_plus  = 1/H**2 + p/(2*H)
            denom       = -2/H**2 + q
            
            # נוסחת גאוס-זיידל
            val = -(coeff_minus * theta[i-1] + coeff_plus * theta[i+1]) / denom
            
            # עדכון עם רלקסציה (מונע אוסצילציות)
            theta[i] = (1 - omega_theta) * theta[i] + omega_theta * val
            
        # תנאי שפה לטמפרטורה
        theta[0] = 1.0
        theta[-1] = 0.0
        
        # בדיקת התכנסות
        # אנו בודקים גם את השינוי ב-theta וגם מוודאים ש-f' בקצה נכון
        err_theta = np.max(np.abs(theta - theta_old))
        bc_error = np.abs(f_prime[-1] - F_PRIME_INF) # בדיקת שפיות לתנאי השפה
        
        if k % 100 == 0:
            print(f"Iter {k}: Theta Error={err_theta:.2e}, f'(inf)={f_prime[-1]:.4f}")

        if err_theta < TOL:
            print(f"\nConverged in {k} iterations.")
            print(f"Final check: f'(inf) actual = {f_prime[-1]:.5f}, required = {F_PRIME_INF}")
            return eta, f, f_prime, theta

    print("Did not converge.")
    return eta, f, f_prime, theta

# הרצה והצגה
eta, f, fp, theta = solve_hybrid_gs()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(eta, theta, 'r', label=r'$\theta$')
plt.title('Temperature')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(eta, fp, 'b', label=r"$f'$")
plt.axhline(F_PRIME_INF, color='k', linestyle='--', label="Target f'(inf)")
plt.title("Velocity Profile (f')")
plt.legend()
plt.grid()
plt.show()