from flask import Flask, render_template, jsonify
import math, random
import matplotlib
matplotlib.use('Agg')  # headless backend for server-side rendering
import matplotlib.pyplot as plt
import io, base64

# Try SciPy for Z critical; fall back to Acklam approximation if not available
try:
    from scipy.stats import norm
    def zcrit_two_tailed(alpha):
        return abs(norm.ppf(1 - alpha/2.0))
except Exception:
    # Acklam approximation for inverse normal CDF
    def _norm_inv(p):
        if not (0.0 < p < 1.0):
            return float('nan')
        a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
              1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
              6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
              3.754408661907416e+00 ]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2*math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = math.sqrt(-2*math.log(1-p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                     ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    def zcrit_two_tailed(alpha):
        return abs(_norm_inv(1 - alpha/2.0))

app = Flask(__name__)

def compute_autocorr(sequence, alpha, i, m):
    """
    Compute the autocorrelation test exactly like the manual method:
    - Build pairs (R_{i+km}, R_{i+(k+1)m})
    - Sum products, estimate rho_hat, sigma, Z0
    - Return rows, equations (as strings), and graphs
    """
    seq = [float(x) for x in sequence]
    N = len(seq)
    if i < 1 or m < 1:
        raise ValueError("i and m must be positive integers (1-based).")
    if N < 2:
        raise ValueError("Provide at least two numbers in the series.")

    # Largest integer M with i + (M+1)m <= N  => M = floor((N - i)/m) - 1
    M = math.floor((N - i) / m) - 1
    if M < 0:
        raise ValueError("Parameters produce negative M. Increase N or adjust i/m.")

    rows = []
    sum_prod = 0.0
    Ri_vals = []
    Rim_vals = []
    for k in range(M + 1):
        a = i + k*m           # 1-based
        b = i + (k+1)*m       # 1-based
        Ra = seq[a-1]
        Rb = seq[b-1]
        prod = Ra * Rb
        rows.append({
            "k": k,
            "a": a, "b": b,
            "Ra": Ra, "Rb": Rb,
            "prod": prod
        })
        sum_prod += prod
        Ri_vals.append(Ra)
        Rim_vals.append(Rb)

    rho_hat = (sum_prod / (M + 1)) - 0.25
    sd = math.sqrt(13*M + 7) / (12*(M + 1))
    Z0 = rho_hat / sd
    Zcrit = zcrit_two_tailed(alpha)
    independent = abs(Z0) <= Zcrit
    decision = "Do NOT Reject H₀ (Independent)" if independent else "Reject H₀ (Dependent)"

    # Plot 1: sequence line
    plt.figure(figsize=(6.2, 3.3))
    plt.plot(range(1, N+1), seq, marker='o', linewidth=1.6)
    plt.title("Random Number Sequence")
    plt.xlabel("Index"); plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    buf1 = io.BytesIO(); plt.tight_layout(); plt.savefig(buf1, format='png'); buf1.seek(0)
    graph_seq = base64.b64encode(buf1.read()).decode('utf-8'); plt.close()

    # Plot 2: scatter Ri vs Ri+m
    plt.figure(figsize=(4.2, 4.2))
    plt.scatter(Ri_vals, Rim_vals, alpha=0.85, edgecolors='black')
    plt.title(f"Scatter: Rᵢ vs Rᵢ₊{m}")
    plt.xlabel("Rᵢ"); plt.ylabel(f"Rᵢ₊{m}")
    plt.grid(True, linestyle='--', alpha=0.6)
    buf2 = io.BytesIO(); plt.tight_layout(); plt.savefig(buf2, format='png'); buf2.seek(0)
    graph_scatter = base64.b64encode(buf2.read()).decode('utf-8'); plt.close()

    # Step-by-step strings (for UI)
    m_eq = f"i + (M+1)m ≤ N  ⇒  {i} + (M+1){m} ≤ {N}  ⇒  M = {M}"
    rho_eq = (
        "ρ̂ = (1/(M+1)) Σ[R_{i+km} · R_{i+(k+1)m}] − 0.25  "
        f"= (1/{M+1})({sum_prod:.6f}) − 0.25 = {rho_hat:.6f}"
    )
    sd_eq = (
        "σ(ρ̂) = √(13M+7) / (12(M+1))  "
        f"= √(13·{M}+7) / (12·({M}+1)) = {sd:.6f}"
    )
    z_eq = f"Z₀ = ρ̂/σ = {rho_hat:.6f}/{sd:.6f} = {Z0:.6f}"
    crit_eq = f"Z_crit = Z_(α/2) with α={alpha}  ⇒  {Zcrit:.4f}"
    final_eq = "Decision: " + ("Do NOT reject H₀ (independent)" if independent else "Reject H₀ (dependent)")

    return {
        "N": N, "i": i, "m": m, "M": M,
        "alpha": alpha,
        "sum_prod": sum_prod,
        "rho_hat": rho_hat,
        "sd": sd,
        "Z0": Z0,
        "Zcrit": Zcrit,
        "independent": bool(independent),
        "decision": decision,
        "rows": rows,
        "sequence": seq,  # include the generated numbers
        "eq": {
            "m_eq": m_eq,
            "rho_eq": rho_eq,
            "sd_eq": sd_eq,
            "z_eq": z_eq,
            "crit_eq": crit_eq,
            "final_eq": final_eq
        },
        "graphs": {
            "sequence": graph_seq,
            "scatter": graph_scatter
        }
    }

# @app.route('/')
# def index():
#     return render_template('index.html')
#     @app.route('/autocorrelation', methods=['POST'])
# def autocorrelation():
#     try:
#         # Auto-generate random values for parameters instead of fixed
#         alpha = random.choice([0.01, 0.05, 0.10])   # common significance levels
#         i = random.randint(1, 8)                    # random start index
#         m = random.randint(2, 8)                    # random lag
#         N = random.randint(30, 80)                  # random sequence length

#         seq = [round(random.random(), 2) for _ in range(N)]
#         result = compute_autocorr(seq, alpha, i, m)

#         return jsonify(result)

#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"error": str(e)}), 400
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocorrelation', methods=['POST'])
def autocorrelation():
    try:
        # Auto-generate random values for parameters instead of fixed
        alpha = random.choice([0.01, 0.05, 0.10])   # common significance levels
        i = random.randint(1, 8)                    # random start index
        m = random.randint(2, 8)                    # random lag
        N = random.randint(30, 80)                  # random sequence length

        seq = [round(random.random(), 2) for _ in range(N)]
        result = compute_autocorr(seq, alpha, i, m)

        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400



# @app.route('/autocorrelation', methods=['POST'])
# def autocorrelation():
#     try:
#         # Auto-generate sequence & fixed params (tune as needed)
#         # alpha = 0.05
#         # i = 5
#         # m = 5
#         # N = 50
#         # seq = [round(random.random(), 2) for _ in range(N)]
#         # result = compute_autocorr(seq, alpha, i, m)
#         # Auto-generate random values for parameters instead of fixed
# alpha = random.choice([0.01, 0.05, 0.10])   # common significance levels
# i = random.randint(1, 8)                    # random start index
# m = random.randint(2, 8)                    # random lag
# N = random.randint(30, 80)                  # random sequence length

# seq = [round(random.random(), 2) for _ in range(N)]
# result = compute_autocorr(seq, alpha, i, m)

#         return jsonify(result)

#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
