import { BrowserRouter as Router, Routes, Route, Navigate, NavLink, useNavigate } from "react-router-dom";
import { useState } from "react";
import TrainingPanel from "./TrainingPanel";
import Login from "./Login";
import Dashboard from "./Dashboard";
import ModelsPage from "./ModelsPage";
import "./App.css";
import axios from "axios";

// Intercept 401 responses to automatically clear expired tokens
axios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem("access_token");
      window.location.href = "/";
    }
    return Promise.reject(error);
  }
);

/* ============================================================
   GLOBAL NAVBAR
   ============================================================ */
function Navbar() {
  const navigate = useNavigate();
  const username = localStorage.getItem("username");

  const handleLogout = () => {
    localStorage.clear();
    navigate("/");
  };

  return (
    <nav className="alpha-navbar">
      <NavLink to="/lab" className="alpha-logo" style={{ textDecoration: "none" }}>
        <div className="alpha-logo-mark">M</div>
        <span>ML Lab</span>
      </NavLink>

      <div className="nav-links">
        <NavLink to="/lab" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
          Lab
        </NavLink>
        <NavLink to="/models" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
          Models
        </NavLink>
        <NavLink to="/dashboard" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
          History
        </NavLink>
      </div>

      <div className="nav-user">
        {username && (
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}>
            @{username}
          </span>
        )}
        <button
          onClick={handleLogout}
          style={{
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.2)",
            color: "#f87171",
            padding: "5px 14px",
            borderRadius: "8px",
            fontSize: "0.8rem",
            cursor: "pointer",
            fontFamily: "var(--font-display)",
            fontWeight: 500,
            transition: "all 0.2s",
          }}
          onMouseEnter={e => { e.target.style.background = "rgba(239,68,68,0.2)"; e.target.style.borderColor = "rgba(239,68,68,0.4)"; }}
          onMouseLeave={e => { e.target.style.background = "rgba(239,68,68,0.1)"; e.target.style.borderColor = "rgba(239,68,68,0.2)"; }}
        >
          Logout
        </button>
      </div>
    </nav>
  );
}

/* ============================================================
   COMPARISON ENGINE
   ============================================================ */
function ComparisonEngine({ a, b }) {
  const [expanded, setExpanded] = useState(false);
  if (!a?.accuracy || !b?.accuracy) return null;

  const insights = [];

  if (a.dataset === b.dataset) {
    if (a.accuracy > b.accuracy) {
      insights.push({
        type: "info",
        text: `Model A (${a.model.toUpperCase()}) outperforms Model B by ${((a.accuracy - b.accuracy) * 100).toFixed(1)}% on the ${a.dataset} dataset.`,
        detail: "Accuracy = correctly classified / total. A higher accuracy on the same dataset tells you one model generalizes better — but check if it's due to overfitting."
      });
    } else if (b.accuracy > a.accuracy) {
      insights.push({
        type: "info",
        text: `Model B (${b.model.toUpperCase()}) outperforms Model A by ${((b.accuracy - a.accuracy) * 100).toFixed(1)}% on the ${a.dataset} dataset.`,
        detail: "Accuracy = correctly classified / total. A higher accuracy on the same dataset tells you one model generalizes better — but check if it's due to overfitting."
      });
    } else {
      insights.push({
        type: "info",
        text: `Both models achieve identical final accuracy on ${a.dataset}.`,
        detail: "Equal accuracy doesn't mean equal behavior — the decision boundaries may be shaped very differently. Check the confusion matrices for subtler differences."
      });
    }
  } else {
    insights.push({
      type: "warning",
      text: `Models are on different datasets (${a.dataset} vs ${b.dataset}). Accuracy comparison is invalid.`,
      detail: "Accuracy is only comparable between models trained on the same dataset and test split. Switch both models to the same dataset for a meaningful comparison."
    });
  }

  const isLinear = (m) => m === "logreg" || m === "svm";
  const isCurved = (m) => m === "mlp" || m === "knn" || m === "rf";
  const isOverfit = (m, acc) => (m === "mlp" || m === "rf") && acc >= 0.99;
  const nonLinearDataset = ["circles", "moons"].includes(a.dataset);

  if (a.accuracy > b.accuracy && isCurved(a.model) && isLinear(b.model)) {
    insights.push({
      type: "explain",
      text: `Structural mismatch: ${a.model.toUpperCase()} captures the curved geometry; ${b.model.toUpperCase()} can only draw a straight line.`,
      detail: `On the ${a.dataset} dataset, the true decision boundary is non-linear. Logistic Regression and SVM (linear kernel) are constrained to hyperplanes — they cannot fit a circle or crescent no matter how much you train them. This is called an inductive bias mismatch.`
    });
  } else if (b.accuracy > a.accuracy && isLinear(a.model) && isCurved(b.model)) {
    insights.push({
      type: "explain",
      text: `Structural advantage: the data clusters linearly, so ${a.model.toUpperCase()}'s simplicity wins. ${b.model.toUpperCase()} wastes capacity on a simple problem.`,
      detail: "Occam's Razor: the simplest model that fits the data is preferred. Complex models like MLPs and Random Forests have higher variance — on clean linearly-separable data they may even perform worse due to chasing noise."
    });
  }

  if (isOverfit(a.model, a.accuracy) && !isOverfit(b.model, b.accuracy)) {
    insights.push({
      type: "warning",
      text: `Model A (${a.model.toUpperCase()}) may be dangerously overfitting. 100% accuracy on training data is a red flag.`,
      detail: "Overfitting = the model memorizes training examples instead of learning the pattern. Notice how the decision boundary wraps tightly around individual points. The real-world accuracy on unseen data would be much lower."
    });
  }

  if (!nonLinearDataset && isCurved(a.model) && b.model === "logreg") {
    insights.push({
      type: "tip",
      text: `On blob-like data, Logistic Regression is often the best choice — interpretable, fast, and sufficient.`,
      detail: "When data is linearly separable, a simpler model wins on all fronts: faster, more interpretable, and better generalizing. Neural Nets and Random Forests offer no benefit here."
    });
  }

  return (
    <div className="comparison-engine">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "18px" }}>
        <h3 style={{ margin: 0, fontFamily: "var(--font-display)", fontSize: "1rem", fontWeight: 700, color: "#fff" }}>
          Comparison Engine
        </h3>
        <button
          onClick={() => setExpanded(!expanded)}
          style={{ background: "rgba(139,92,246,0.12)", border: "1px solid rgba(139,92,246,0.25)", color: "#a78bfa", padding: "4px 14px", borderRadius: "8px", fontSize: "0.78rem", cursor: "pointer", fontFamily: "var(--font-display)" }}
        >
          {expanded ? "Collapse" : "Expand Details"}
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "18px" }}>
        {[
          { label: "Model A", data: a, color: "#3b82f6" },
          { label: "Model B", data: b, color: "#06b6d4" },
        ].map(({ label, data, color }) => (
          <div key={label} style={{ background: "rgba(0,0,0,0.2)", borderRadius: "12px", padding: "14px 16px", border: `1px solid ${color}22` }}>
            <div style={{ fontSize: "0.72rem", color, fontFamily: "var(--font-display)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px" }}>{label}</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.88rem", color: "#fff", marginBottom: "4px" }}>{data.model?.toUpperCase()}</div>
            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>{data.dataset} · {(data.accuracy * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {insights.map((ins, i) => (
          <div key={i}>
            <div className="comparison-insight" style={{
              borderLeftColor: ins.type === "warning" ? "var(--warning)" : ins.type === "tip" ? "var(--success)" : ins.type === "explain" ? "var(--alpha-cyan)" : "var(--alpha-violet)"
            }}>
              {ins.text}
            </div>
            {expanded && ins.detail && (
              <div className="explain-box" style={{ marginTop: "4px", borderRadius: "0 0 8px 8px", borderTopLeftRadius: 0, borderTop: "none" }}>
                <div className="explain-title">Why?</div>
                {ins.detail}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ============================================================
   LAB PAGE
   ============================================================ */
function LabPage() {
  const [stateA, setStateA] = useState(null);
  const [stateB, setStateB] = useState(null);

  return (
    <div className="app-container">
      <div className="page-hero">
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
          <span className="alpha-badge blue">Side-by-Side</span>
          <span className="alpha-badge cyan">Real-Time</span>
          <span className="alpha-badge green">Interactive</span>
        </div>
        <h1 className="page-hero-title">Explainable ML Lab</h1>
        <p className="page-hero-sub">
          Train two models simultaneously and watch their decision boundaries evolve in real time. Hover the canvas to probe the math at any point. Click confusion matrix cells to spotlight specific prediction types.
          <br />
          <span style={{ color: "var(--alpha-cyan)" }}>Drag any data point</span> to see how the model adapts on the fly.
        </p>
      </div>

      <div className="panels-grid">
        <TrainingPanel title="A" onStateChange={setStateA} />
        <TrainingPanel title="B" onStateChange={setStateB} />
      </div>

      <ComparisonEngine a={stateA} b={stateB} />
    </div>
  );
}

/* ============================================================
   PROTECTED ROUTE
   ============================================================ */
function ProtectedRoute({ children }) {
  const token = localStorage.getItem("access_token");
  if (!token) return <Navigate to="/" />;
  return (
    <>
      <Navbar />
      {children}
    </>
  );
}

/* ============================================================
   APP ROOT
   ============================================================ */
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/lab" element={<ProtectedRoute><LabPage /></ProtectedRoute>} />
        <Route path="/models" element={<ProtectedRoute><ModelsPage /></ProtectedRoute>} />
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
      </Routes>
    </Router>
  );
}

export default App;