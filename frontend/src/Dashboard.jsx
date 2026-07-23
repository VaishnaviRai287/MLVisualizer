import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";

const MODEL_COLORS = {
  mlp: { color: "#8b5cf6", label: "Neural Net", badge: "violet" },
  svm: { color: "#3b82f6", label: "SVM", badge: "blue" },
  knn: { color: "#f59e0b", label: "KNN", badge: "orange" },
  logreg: { color: "#10b981", label: "LogReg", badge: "green" },
  rf: { color: "#06b6d4", label: "Rand. Forest", badge: "cyan" },
};

function AccuracyBar({ value }) {
  const pct = (value * 100).toFixed(1);
  const color = value > 0.9 ? "#10b981" : value > 0.7 ? "#3b82f6" : value > 0.55 ? "#f59e0b" : "#ef4444";
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", marginBottom: "5px" }}>
        <span style={{ color: "var(--text-muted)" }}>Accuracy</span>
        <span style={{ fontFamily: "var(--font-mono)", color, fontWeight: 600 }}>{pct}%</span>
      </div>
      <div className="accuracy-bar-outer">
        <div className="accuracy-bar-inner" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})` }} />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [experiments, setExperiments] = useState([]);
  const [sortBy, setSortBy] = useState("date");
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) { navigate("/"); return; }
    axios.get("http://localhost:8000/api/experiments/", {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => setExperiments(res.data))
    .catch(err => {
      if (err.response?.status === 401) { localStorage.clear(); navigate("/"); }
    });
  }, [navigate]);

  const sorted = [...experiments].sort((a, b) =>
    sortBy === "accuracy"
      ? b.accuracy - a.accuracy
      : new Date(b.run_date) - new Date(a.run_date)
  );

  const bestAccuracy = experiments.length > 0 ? Math.max(...experiments.map(e => e.accuracy)) : null;
  const avgAccuracy = experiments.length > 0 ? experiments.reduce((s, e) => s + e.accuracy, 0) / experiments.length : null;
  const modelCounts = experiments.reduce((acc, e) => { acc[e.model_type] = (acc[e.model_type] || 0) + 1; return acc; }, {});

  return (
    <div className="app-container">
      {/* Hero */}
      <div className="page-hero">
        <h1 className="page-hero-title">Experiment History</h1>
        <p className="page-hero-sub">All your saved runs. Compare models, track progress, and see which algorithms work best on which datasets.</p>
      </div>

      {/* Stats row */}
      {experiments.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "14px", marginBottom: "32px" }}>
          {[
            { label: "Total Runs", value: experiments.length },
            { label: "Best Accuracy", value: bestAccuracy ? `${(bestAccuracy * 100).toFixed(1)}%` : "--" },
            { label: "Avg Accuracy", value: avgAccuracy ? `${(avgAccuracy * 100).toFixed(1)}%` : "--" },
            { label: "Models Tried", value: Object.keys(modelCounts).length },
          ].map((s, i) => (
            <div key={i} className="glass-panel" style={{ padding: "18px 20px" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.3rem", fontWeight: 700, color: "#fff", marginBottom: "4px" }}>{s.value}</div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-display)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{s.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Model usage breakdown */}
      {Object.keys(modelCounts).length > 0 && (
        <div className="glass-panel" style={{ padding: "20px 24px", marginBottom: "28px" }}>
          <div style={{ fontSize: "0.72rem", color: "var(--alpha-cyan)", fontFamily: "var(--font-display)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "14px" }}>
            Model Usage
          </div>
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            {Object.entries(modelCounts).map(([m, count]) => {
              const meta = MODEL_COLORS[m] || { color: "#8b8b9e", label: m.toUpperCase(), badge: "blue" };
              return (
                <div key={m} style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(0,0,0,0.2)", padding: "8px 14px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.06)" }}>
                  <span className={`alpha-badge ${meta.badge}`}>{meta.label}</span>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "#fff", fontWeight: 600 }}>{count}</span>
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>run{count > 1 ? "s" : ""}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Sort controls */}
      {experiments.length > 0 && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "18px", gap: "8px" }}>
          <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", alignSelf: "center" }}>Sort by:</span>
          {["date", "accuracy"].map(s => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              style={{
                background: sortBy === s ? "rgba(59,130,246,0.15)" : "rgba(255,255,255,0.04)",
                border: `1px solid ${sortBy === s ? "rgba(59,130,246,0.35)" : "rgba(255,255,255,0.08)"}`,
                color: sortBy === s ? "#60a5fa" : "var(--text-muted)",
                padding: "5px 14px",
                borderRadius: "8px",
                fontSize: "0.8rem",
                cursor: "pointer",
                fontFamily: "var(--font-display)",
                transition: "all 0.2s",
                textTransform: "capitalize",
              }}
            >
              {s === "date" ? "Recent First" : "Best Accuracy"}
            </button>
          ))}
        </div>
      )}

      {/* Experiment cards */}
      {experiments.length === 0 ? (
        <div className="glass-panel" style={{ padding: "60px 40px", textAlign: "center" }}>
          <h3 style={{ fontFamily: "var(--font-display)", color: "#fff", margin: "0 0 10px", fontSize: "1.2rem" }}>No experiments yet</h3>
          <p style={{ color: "var(--text-muted)", fontSize: "0.88rem", margin: "0 0 24px", lineHeight: 1.6 }}>
            Head to the Lab, train a model, and click <strong style={{ color: "#e2e8f0" }}>Save</strong> to record it here.
          </p>
          <Link to="/lab">
            <button className="btn-train">Open Lab</button>
          </Link>
        </div>
      ) : (
        <div className="panels-grid">
          {sorted.map(exp => {
            const meta = MODEL_COLORS[exp.model_type] || { color: "#8b8b9e", label: exp.model_type.toUpperCase(), badge: "blue" };
            const isBest = exp.accuracy === bestAccuracy;
            return (
              <div key={exp.id} className="glass-panel exp-card" style={{ borderColor: isBest ? "rgba(16,185,129,0.2)" : undefined }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div style={{
                      width: "36px", height: "36px", borderRadius: "10px",
                      background: `${meta.color}18`,
                      border: `1px solid ${meta.color}30`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontFamily: "var(--font-mono)", fontSize: "9px", fontWeight: 700,
                      color: meta.color, letterSpacing: "-0.02em",
                    }}>
                      {exp.model_type.toUpperCase().slice(0, 3)}
                    </div>
                    <div>
                      <h3 style={{ margin: 0, fontFamily: "var(--font-display)", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                        {exp.dataset_name}
                      </h3>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>
                        {new Date(exp.run_date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "6px" }}>
                    <span className={`alpha-badge ${meta.badge}`}>{meta.label}</span>
                    {isBest && <span className="alpha-badge green">Best</span>}
                  </div>
                </div>

                <AccuracyBar value={exp.accuracy} />

                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.6, background: "rgba(0,0,0,0.15)", borderRadius: "8px", padding: "8px 10px" }}>
                  {exp.model_type === "mlp" && `Neural net learned a non-linear boundary over 50 epochs on ${exp.dataset_name}.`}
                  {exp.model_type === "svm" && `SVM found the maximum-margin boundary on ${exp.dataset_name} using an RBF kernel.`}
                  {exp.model_type === "knn" && `KNN classified by majority vote of nearest neighbors on ${exp.dataset_name}.`}
                  {exp.model_type === "logreg" && `Logistic Regression fit a linear decision boundary on ${exp.dataset_name}.`}
                  {exp.model_type === "rf" && `Random Forest ensemble of trees voted on ${exp.dataset_name}.`}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {experiments.length > 0 && (
        <div style={{ marginTop: "32px", padding: "16px 20px", background: "rgba(59,130,246,0.05)", border: "1px solid rgba(59,130,246,0.1)", borderRadius: "12px", fontSize: "0.8rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
          <strong style={{ color: "#93c5fd" }}>Tip:</strong> Try the same dataset with LogReg vs MLP to see how boundary complexity affects accuracy. On linearly-separable data (Blobs), LogReg often matches or beats more complex models.
        </div>
      )}
    </div>
  );
}
