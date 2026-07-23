import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";


const FEATURES = [
  { title: "Live Decision Boundaries", desc: "Watch the model's boundary reshape itself every epoch in real time." },
  { title: "Probe Any Point", desc: "Hover anywhere to see live math: logits, distances, support vectors." },
  { title: "Dual Model Comparison", desc: "Train two algorithms side-by-side and let the Comparison Engine explain why one wins." },
  { title: "Model Encyclopedia", desc: "Deep-dive into the math, complexity, and trade-offs of every algorithm." },
];

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isRegister, setIsRegister] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (isRegister) {
        try {
          await axios.post(`${API_URL}/api/register/`, { username, password });
        } catch (regErr) {
          const detail = regErr.response?.data;
          if (!detail) {
            setError("Cannot reach the server. Check your connection.");
          } else if (detail?.username) {
            setError(`Username: ${detail.username[0]}`);
          } else if (detail?.password) {
            setError(`Password: ${detail.password[0]}`);
          } else {
            // Show the raw backend message for easier debugging
            const raw = JSON.stringify(detail);
            setError(`Registration failed: ${raw}`);
          }
          setLoading(false);
          return;
        }
      }
      const res = await axios.post(`${API_URL}/api/token/`, { username, password });
      localStorage.setItem("access_token", res.data.access);
      localStorage.setItem("refresh_token", res.data.refresh);
      localStorage.setItem("username", username);
      navigate("/lab");
    } catch (loginErr) {
      const detail = loginErr.response?.data;
      if (!detail) setError("Cannot reach the server. Is the backend running?");
      else setError("Login failed. Check your credentials.");
      setLoading(false);
    }
  };


  return (
    <div className="login-page">
      {/* Ambient glow orbs */}
      <div style={{ position: "fixed", top: "15%", left: "8%", width: "500px", height: "500px", borderRadius: "50%", background: "radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%)", pointerEvents: "none" }} />
      <div style={{ position: "fixed", bottom: "15%", right: "8%", width: "400px", height: "400px", borderRadius: "50%", background: "radial-gradient(circle, rgba(6,182,212,0.05) 0%, transparent 70%)", pointerEvents: "none" }} />

      <div style={{ display: "flex", gap: "80px", alignItems: "center", maxWidth: "1000px", width: "100%" }}>

        {/* Left — Branding + feature list */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "28px" }}>
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
            <div style={{
              width: "52px", height: "52px", borderRadius: "16px",
              background: "linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "22px", fontWeight: 800, color: "#fff",
              boxShadow: "0 0 30px rgba(59,130,246,0.4)",
              fontFamily: "var(--font-display)",
              letterSpacing: "-0.04em",
            }}>M</div>
            <div>
              <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.6rem", letterSpacing: "-0.04em", color: "#fff" }}>ML Lab</div>
              <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "var(--font-display)" }}>Explainable Machine Learning</div>
            </div>
          </div>

          {/* Tagline */}
          <div style={{ fontFamily: "var(--font-display)", fontSize: "1.9rem", fontWeight: 700, letterSpacing: "-0.04em", lineHeight: 1.25, color: "#fff" }}>
            See how AI<br />
            <span style={{ background: "linear-gradient(135deg, #3b82f6, #06b6d4)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              actually thinks.
            </span>
          </div>

          <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", lineHeight: 1.7, margin: 0 }}>
            An interactive laboratory for probing machine learning decision boundaries in real time. No more black boxes.
          </p>

          {/* Features */}
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {FEATURES.map((f, i) => (
              <div key={i} className="login-feature">
                <div className="login-feature-icon" />
                <div>
                  <div style={{ fontWeight: 600, color: "#e2e8f0", fontFamily: "var(--font-display)", marginBottom: "2px", fontSize: "0.85rem" }}>{f.title}</div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>{f.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right — Login card */}
        <div className="glass-panel login-card">
          <div style={{ textAlign: "center", marginBottom: "4px" }}>
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.3rem", letterSpacing: "-0.03em", color: "#fff", marginBottom: "6px" }}>
              {isRegister ? "Create Account" : "Welcome back"}
            </div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              {isRegister ? "Start exploring your models" : "Sign in to ML Lab"}
            </div>
          </div>

          <hr className="section-divider" style={{ margin: "4px 0 8px" }} />

          {/* Error */}
          {error && (
            <div style={{
              background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.25)",
              color: "#fca5a5", padding: "10px 14px", borderRadius: "10px",
              fontSize: "0.82rem", lineHeight: 1.4,
            }}>
              {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "6px", fontFamily: "var(--font-display)", fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Username</label>
              <input
                id="login-username"
                className="styled-select"
                type="text"
                placeholder="e.g. researcher"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                style={{ backgroundImage: "none", width: "100%", padding: "11px 14px" }}
              />
            </div>
            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "6px", fontFamily: "var(--font-display)", fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Password</label>
              <input
                id="login-password"
                className="styled-select"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{ backgroundImage: "none", width: "100%", padding: "11px 14px" }}
              />
            </div>

            <button
              id="login-submit"
              className="btn-train"
              type="submit"
              disabled={loading}
              style={{ marginTop: "6px", padding: "12px", fontSize: "0.95rem", letterSpacing: "-0.01em" }}
            >
              {loading ? "Authenticating..." : isRegister ? "Create Account" : "Enter Lab"}
            </button>
          </form>

          {/* Toggle */}
          <div
            style={{ textAlign: "center", color: "var(--text-muted)", fontSize: "0.8rem", cursor: "pointer", paddingTop: "4px" }}
            onClick={() => { setIsRegister(!isRegister); setError(""); }}
          >
            {isRegister
              ? <span>Already have an account? <span style={{ color: "var(--alpha-cyan)" }}>Sign in</span></span>
              : <span>No account? <span style={{ color: "var(--alpha-cyan)" }}>Register for free</span></span>
            }
          </div>

          {/* Version badge */}
          <div style={{ textAlign: "center" }}>
            <span className="alpha-badge blue" style={{ fontSize: "9px" }}>ML Lab v1.0 · XAI Research Platform</span>
          </div>
        </div>
      </div>
    </div>
  );
}
