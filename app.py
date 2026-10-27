import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 0. CONFIGURATION & ENGINE UTILITIES
# ==========================================
st.set_page_config(
    page_title="VAYU TARAL - Numerical Math Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Universal Math Execution Scope
MATH_SCOPE = {
    "np": np, 
    "sin": np.sin, 
    "cos": np.cos, 
    "tan": np.tan, 
    "exp": np.exp, 
    "log": np.log, 
    "sqrt": np.sqrt, 
    "pi": np.pi, 
    "e": np.e
}

def safe_eval(func_str, variable_dict):
    """Safely evaluates a user-entered math expression string."""
    scope = {**MATH_SCOPE, **variable_dict}
    try:
        return eval(func_str, {"__builtins__": None}, scope)
    except Exception:
        return None

# ==========================================
# 1. GANITYA SOLVER ENGINES
# ==========================================

# --- Root Finding: Bisection ---
def bisection_method(func_str, a, b, tol=1e-5, max_iter=100):
    fa, fb = safe_eval(func_str, {"x": a}), safe_eval(func_str, {"x": b})
    if fa is None or fb is None:
        return None, "Math Error: Unable to evaluate equation."
    if fa * fb >= 0:
        return None, "Invalid Bounds: f(a) and f(b) must have opposite signs."

    history = []
    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = safe_eval(func_str, {"x": c})
        if fc is None: break
        history.append((c, fc))
        
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return history, "Success"
        
        if np.sign(fc) == np.sign(fa):
            a, fa = c, fc
        else:
            b, fb = c, fc
            
    return history, "Maximum iterations reached"

# --- Root Finding: Regula Falsi ---
def regula_falsi_method(func_str, a, b, tol=1e-5, max_iter=100):
    fa, fb = safe_eval(func_str, {"x": a}), safe_eval(func_str, {"x": b})
    if fa is None or fb is None:
        return None, "Math Error: Unable to evaluate equation."
    if fa * fb >= 0:
        return None, "Invalid Bounds: f(a) and f(b) must have opposite signs."

    history = []
    for _ in range(max_iter):
        c = (a * fb - b * fa) / (fb - fa)
        fc = safe_eval(func_str, {"x": c})
        if fc is None: break
        history.append((c, fc))
        
        if abs(fc) < tol:
            return history, "Success"
        
        if np.sign(fc) == np.sign(fa):
            a, fa = c, fc
        else:
            b, fb = c, fc
            
    return history, "Success (Max Iterations)"

# --- Root Finding: Newton-Raphson ---
def newton_raphson_method(func_str, x0, tol=1e-7, max_iter=100):
    def f_prime(val, h=1e-5):
        v1 = safe_eval(func_str, {"x": val + h})
        v2 = safe_eval(func_str, {"x": val - h})
        if v1 is None or v2 is None: return None
        return (v1 - v2) / (2 * h)

    history = []
    curr_x = x0
    
    for _ in range(max_iter):
        fx = safe_eval(func_str, {"x": curr_x})
        dfx = f_prime(curr_x)
        
        if fx is None or dfx is None:
            return None, "Math Error: Unable to evaluate equation or derivative."
        if abs(dfx) < 1e-12:
            return None, "Zero Slope Encountered (Local Minima/Maxima hit)."

        next_x = curr_x - fx / dfx
        history.append((curr_x, fx))
        
        if abs(next_x - curr_x) < tol:
            return history, "Success"
            
        curr_x = next_x
            
    return history, "Maximum iterations reached"

# --- Root Finding: Secant ---
def secant_method(func_str, x0, x1, tol=1e-6, max_iter=100):
    history = []
    for _ in range(max_iter):
        f_x0 = safe_eval(func_str, {"x": x0})
        f_x1 = safe_eval(func_str, {"x": x1})
        
        if f_x0 is None or f_x1 is None:
            return None, "Math Error: Unable to evaluate equation."
        if abs(f_x1 - f_x0) < 1e-15:
            return None, "Slope zero encountered between current guesses."

        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        f_x2 = safe_eval(func_str, {"x": x2})
        history.append((x2, f_x2))
        
        if abs(f_x2) < tol:
            return history, "Success"
        
        x0, x1 = x1, x2
            
    return history, "Maximum iterations reached"

# --- Calculus: Numerical Differentiation ---
def numerical_diff(func_str, x_val, h):
    f_val = safe_eval(func_str, {"x": x_val})
    f_fwd = safe_eval(func_str, {"x": x_val + h})
    f_bwd = safe_eval(func_str, {"x": x_val - h})
    
    forward = (f_fwd - f_val) / h
    backward = (f_val - f_bwd) / h
    central = (f_fwd - f_bwd) / (2 * h)
    
    return f_val, forward, backward, central

# --- Calculus: Numerical Integration ---
def trapezoidal_rule(func_str, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = safe_eval(func_str, {"x": x})
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

def simpsons_rule(func_str, a, b, n):
    if n % 2 != 0: n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = safe_eval(func_str, {"x": x})
    sum_val = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * sum_val

# --- Differential Equations: RK4 ---
def runge_kutta_4(func_str, y0, t_span, h):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        t, y = t_values[i], y_values[i]
        
        k1 = h * safe_eval(func_str, {"t": t, "y": y})
        k2 = h * safe_eval(func_str, {"t": t + h/2, "y": y + k1/2})
        k3 = h * safe_eval(func_str, {"t": t + h/2, "y": y + k2/2})
        k4 = h * safe_eval(func_str, {"t": t + h, "y": y + k3})
        
        y_values[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
    return t_values, y_values

# ==========================================
# 2. UI NAVIGATION & CATEGORIZATION
# ==========================================
st.sidebar.title("VAYU TARAL 🚀")
st.sidebar.markdown("**Ganitya Engine: Math Suite**")

category = st.sidebar.selectbox(
    "Select Solver Category:",
    ["1. Non-Linear Root Finders", "2. Numerical Calculus", "3. Differential Equations (ODEs)"]
)

if category == "1. Non-Linear Root Finders":
    method = st.sidebar.radio(
        "Select Algorithm:",
        ["Bisection Method", "Regula Falsi (False Position)", "Newton-Raphson", "Secant Method"]
    )
elif category == "2. Numerical Calculus":
    method = st.sidebar.radio(
        "Select Operation:",
        ["Numerical Differentiation", "Numerical Integration (Trapezoidal & Simpson)"]
    )
else:
    method = "Runge-Kutta 4th Order (RK4)"

st.sidebar.divider()

# ==========================================
# 3. ROUTE EXECUTIONS
# ==========================================

# ------------------------------------------
# CATEGORY 1: ROOT FINDERS
# ------------------------------------------
if category == "1. Non-Linear Root Finders":
    st.title(f"Vayu Taral: {method}")
    
    col_input, col_view = st.columns([1, 2])
    
    with col_input:
        st.subheader("Input Configuration")
        equation = st.text_input("Function f(x) =", "x**3 - x - 2")
        st.caption("Supports: `x**2`, `np.sin(x)`, `np.exp(x)`, `np.log(x)`, `np.sqrt(x)`")
        
        if method == "Bisection Method":
            a_val = st.number_input("Lower Bound (a)", value=1.0)
            b_val = st.number_input("Upper Bound (b)", value=2.0)
            tol_val = st.number_input("Tolerance", value=1e-5, format="%.6f")
            history, status = bisection_method(equation, a_val, b_val, tol_val)
            
        elif method == "Regula Falsi (False Position)":
            a_val = st.number_input("Lower Bound (a)", value=0.0)
            b_val = st.number_input("Upper Bound (b)", value=3.0)
            tol_val = st.number_input("Tolerance", value=1e-5, format="%.6f")
            history, status = regula_falsi_method(equation, a_val, b_val, tol_val)
            
        elif method == "Newton-Raphson":
            x_guess = st.number_input("Initial Guess (x0)", value=1.0)
            tol_val = st.number_input("Tolerance", value=1e-7, format="%.7f")
            history, status = newton_raphson_method(equation, x_guess, tol_val)
            
        elif method == "Secant Method":
            x_start = st.number_input("Initial Guess (x0)", value=0.0)
            x_next = st.number_input("Initial Guess (x1)", value=3.0)
            tol_val = st.number_input("Tolerance", value=1e-6, format="%.7f")
            history, status = secant_method(equation, x_start, x_next, tol_val)

    with col_view:
        if history and status.startswith("Success"):
            root = history[-1][0]
            st.success(f"✅ Root Converged at **x ≈ {root:.7f}** in **{len(history)}** iterations.")
            
            # Auto-bound generation for plots
            if method in ["Bisection Method", "Regula Falsi (False Position)"]:
                x_range = np.linspace(a_val - 0.5, b_val + 0.5, 500)
            elif method == "Newton-Raphson":
                x_range = np.linspace(min(x_guess, root) - 1, max(x_guess, root) + 1, 500)
            else:
                x_range = np.linspace(min(x_start, root) - 1, max(x_next, root) + 1, 500)

            y_range = safe_eval(equation, {"x": x_range})

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=y_range, name="f(x)", line=dict(color='#00f2ff', width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.add_trace(go.Scatter(x=[root], y=[0], mode='markers+text', text=["Root"], textposition="top center",
                                     marker=dict(size=12, color='red'), name="Root"))
            fig.update_layout(template="plotly_dark", title=f"Convergence Visualizer ({method})",
                              xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Show Iteration History"):
                st.write(history)
        else:
            st.error(f"❌ Convergence Failure: {status}")

# ------------------------------------------
# CATEGORY 2: NUMERICAL CALCULUS
# ------------------------------------------
elif category == "2. Numerical Calculus":
    st.title(f"Vayu Taral: {method}")
    
    if method == "Numerical Differentiation":
        c1, c2 = st.columns([1, 2])
        with c1:
            equation = st.text_input("Function f(x) =", "x**2 + sin(x)")
            x_target = st.number_input("Point of Interest (x)", value=1.0)
            h_step = st.number_input("Step Size (h)", value=0.01, format="%.4f")
            
            f_val, fwd, bwd, central = numerical_diff(equation, x_target, h_step)
            
            st.metric("Forward Difference", f"{fwd:.6f}")
            st.metric("Backward Difference", f"{bwd:.6f}")
            st.metric("Central Difference (Best)", f"{central:.6f}")

        with c2:
            x_range = np.linspace(x_target - 2, x_target + 2, 500)
            y_range = safe_eval(equation, {"x": x_range})
            tangent_y = f_val + central * (x_range - x_target)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=y_range, name="f(x)", line=dict(color='cyan', width=3)))
            fig.add_trace(go.Scatter(x=x_range, y=tangent_y, name="Tangent Line", line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=[x_target], y=[f_val], name="Evaluation Point", marker=dict(size=10, color='yellow')))
            fig.update_layout(template="plotly_dark", title=f"Local Tangent Approximation at x={x_target}")
            st.plotly_chart(fig, use_container_width=True)

    elif method == "Numerical Integration (Trapezoidal & Simpson)":
        c1, c2 = st.columns([1, 2])
        with c1:
            equation = st.text_input("Function f(x) =", "sin(x) + 2")
            a_limit = st.number_input("Lower Limit (a)", value=0.0)
            b_limit = st.number_input("Upper Limit (b)", value=np.pi)
            n_intervals = st.slider("Intervals (n)", 2, 100, 20)

            trap_res = trapezoidal_rule(equation, a_limit, b_limit, n_intervals)
            simp_res = simpsons_rule(equation, a_limit, b_limit, n_intervals)

            st.metric("Trapezoidal Rule Result", f"{trap_res:.6f}")
            st.metric("Simpson's 1/3 Rule Result", f"{simp_res:.6f}")

        with c2:
            x_vals = np.linspace(a_limit, b_limit, 500)
            y_vals = safe_eval(equation, {"x": x_vals})

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', name="Integrated Area", line=dict(color='cyan', width=3)))
            fig.update_layout(template="plotly_dark", title=f"Definite Integral from {a_limit} to {b_limit}",
                              xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# CATEGORY 3: DIFFERENTIAL EQUATIONS (RK4)
# ------------------------------------------
elif category == "3. Differential Equations (ODEs)":
    st.title("Vayu Taral: Runge-Kutta 4th Order (RK4)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        equation = st.text_input("dy/dt =", "-0.5 * y + sin(t)")
        y_init = st.number_input("Initial Condition y(0)", value=1.0)
        t_end = st.slider("Simulation End Time (t)", 1, 50, 10)
        step_size = st.number_input("Step Size (h)", value=0.1, format="%.3f")

        t_pts, y_pts = runge_kutta_4(equation, y_init, [0, t_end], step_size)

        st.info("RK4 achieves $O(h^4)$ global accuracy, taking four weighted slope evaluations per time step.")

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_pts, y=y_pts, name="State y(t)", line=dict(color='#00f2ff', width=3)))
        fig.update_layout(template="plotly_dark", title=f"RK4 Integration: dy/dt = {equation}",
                          xaxis_title="Time (t)", yaxis_title="Output y(t)")
        st.plotly_chart(fig, use_container_width=True)