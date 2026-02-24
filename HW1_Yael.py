import numpy as np # מייבא ספרית נומפיי לעבודה עם מערכים מתמטיים
from scipy.integrate import solve_ivp # מייבא פונקציה לפתרון משוואות דיפרנציאליות
from scipy.optimize import root # מייבא פונקציה למציאת שורשים של מערכות משוואות
from matplotlib import pyplot as plt # מייבא ספרית matplotlib לציור גרפים

# x = np.linspace(0, 10, 100) # יוצר מערך של ערכים מ-0 עד 10 עם 100 נקודות
# y = np.sin(x) # מחשב את ערכי הסינוס של כל ערך

# plt.plot(x, y) # מצייר את הגרף של y כפונקציה של x
# plt.title("גרף של פונקציית הסינוס") # מוסיף כותרת לגרף
# plt.xlabel("x") # מוסיף תווית לציר ה-x  
# plt.ylabel("sin(x)") # מוסיף תווית לציר ה-y 
# plt.grid() # מוסיף רשת לגרף
# plt.show() # מציג את הגרף

def dynamic_system(y0, v_0, g, dt):
    y = y0 + v_0 * dt - 0.5 * g * dt**2
    v_y = v_0 - g * dt
    return y, v_y

def derivative(y1, y2, dt):
    dydt = (y2 - y1) / dt
    return dydt

t_span = 30.0  # זמן כולל של הסימולציה בשניות
dt = 0.1       # צעד זמן בשניות
g = 9.81      # תאוצת הכובד במטר לשנייה בריבוע
num_steps = int(t_span / dt)  # מספר הצעדים בסימולציה
y_positions = np.zeros(num_steps)  # מערך לאחסון מיקומי הגובה
y_velocities = np.zeros(num_steps)  # מערך לאחסון מהירויות
y0 = 100.0  # גובה התחלתי במטרים
v_0 = 0.0  # מהירות התחלתית במטר לשנייה
for i in range(num_steps):
    y_positions[i], y_velocities[i] = dynamic_system(y0, v_0, g, dt)
    y0 = y_positions[i]
    v_0 = y_velocities[i]

time = np.linspace(0, t_span, num_steps)  # מערך זמן עבור הציור
# # Create time array for plotting
# plt.plot(time, y_positions)  # Plot height position as a function
# plt.plot(time, y_velocities)  # Plot velocity as a function of time
# plt.grid()
# plt.legend(['Height (meters)', 'Velocity (meters/second)'])
# plt.show()

# # Plot y position
# plt.figure(figsize=(10, 5))
# plt.plot(time, y_positions, 'b-', linewidth=2)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Height (meters)')
# plt.title('Height Position as a Function of Time')
# plt.grid(True)
# plt.tight_layout()
# plt.show(block=False)

# # Plot y velocity
# plt.figure(figsize=(10, 5))
# plt.plot(time, y_velocities, 'r-', linewidth=2)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Velocity (meters/second)')
# plt.title('Velocity as a Function of Time')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
U=[U1,U2,U3,U4]                           #הורדת סדר אינטגרציה
U1_tag=U2
U2_tag=-(((lamda-2)/3) *eta*u4)
U3_tag=U4
U4_tag= lamda*U2U3-((lamda+1/3)*U1*U4)



