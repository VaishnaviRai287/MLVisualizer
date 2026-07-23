import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

/* ============================================================
   SVG DIAGRAMS — inline, no external deps
   ============================================================ */

function SVMDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
      {/* Background */}
      <rect width="200" height="160" fill="#020510" rx="8" />
      {/* Region fills */}
      <rect x="0" y="0" width="200" height="160" fill="none" />
      <polygon points="0,0 200,0 200,160" fill="rgba(59,130,246,0.06)" />
      <polygon points="0,0 0,160 200,160" fill="rgba(245,158,11,0.06)" />
      {/* Margin lines */}
      <line x1="30" y1="130" x2="170" y2="30" stroke="rgba(59,130,246,0.25)" strokeWidth="1" strokeDasharray="4,4" />
      <line x1="50" y1="130" x2="190" y2="30" stroke="rgba(59,130,246,0.15)" strokeWidth="1" strokeDasharray="3,3" />
      {/* Decision boundary */}
      <line x1="40" y1="130" x2="180" y2="30" stroke="#3b82f6" strokeWidth="2.5" />
      {/* Support vectors — circled */}
      <circle cx="52" cy="112" r="5" fill="#f59e0b" stroke="#3b82f6" strokeWidth="2" />
      <circle cx="78" cy="95" r="5" fill="#f59e0b" stroke="#3b82f6" strokeWidth="2" />
      <circle cx="130" cy="55" r="5" fill="#60a5fa" stroke="#3b82f6" strokeWidth="2" />
      {/* Other points */}
      <circle cx="32" cy="125" r="4" fill="#f59e0b" opacity="0.6" />
      <circle cx="45" cy="140" r="4" fill="#f59e0b" opacity="0.6" />
      <circle cx="65" cy="118" r="4" fill="#f59e0b" opacity="0.6" />
      <circle cx="150" cy="40" r="4" fill="#60a5fa" opacity="0.6" />
      <circle cx="165" cy="55" r="4" fill="#60a5fa" opacity="0.6" />
      <circle cx="140" cy="35" r="4" fill="#60a5fa" opacity="0.6" />
      <circle cx="110" cy="68" r="4" fill="#60a5fa" opacity="0.6" />
      {/* Margin bracket annotation */}
      <text x="95" y="68" fill="#64748b" fontSize="9" fontFamily="monospace">margin</text>
      <text x="5" y="15" fill="#3b82f6" fontSize="9" fontFamily="monospace">decision</text>
      <text x="5" y="25" fill="#3b82f6" fontSize="9" fontFamily="monospace">boundary</text>
    </svg>
  );
}

function KNNDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
      <rect width="200" height="160" fill="#020510" rx="8" />
      {/* Query point */}
      <circle cx="100" cy="80" r="7" fill="#06b6d4" stroke="#fff" strokeWidth="1.5" />
      <text x="108" y="72" fill="#06b6d4" fontSize="9" fontFamily="monospace">query</text>
      {/* k-radius circle */}
      <circle cx="100" cy="80" r="50" fill="none" stroke="rgba(6,182,212,0.2)" strokeWidth="1.5" strokeDasharray="4,3" />
      {/* Neighbor lines */}
      <line x1="100" y1="80" x2="60" y2="50" stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="3,3" opacity="0.8" />
      <line x1="100" y1="80" x2="140" y2="45" stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="3,3" opacity="0.8" />
      <line x1="100" y1="80" x2="130" y2="110" stroke="#60a5fa" strokeWidth="1.2" strokeDasharray="3,3" opacity="0.8" />
      {/* k=3 neighbors */}
      <circle cx="60" cy="50" r="5" fill="#f59e0b" />
      <circle cx="140" cy="45" r="5" fill="#f59e0b" />
      <circle cx="130" cy="110" r="5" fill="#60a5fa" />
      {/* Other background points */}
      <circle cx="30" cy="30" r="4" fill="#f59e0b" opacity="0.45" />
      <circle cx="50" cy="130" r="4" fill="#60a5fa" opacity="0.45" />
      <circle cx="170" cy="120" r="4" fill="#60a5fa" opacity="0.45" />
      <circle cx="165" cy="75" r="4" fill="#f59e0b" opacity="0.45" />
      <circle cx="25" cy="95" r="4" fill="#f59e0b" opacity="0.45" />
      <circle cx="85" cy="145" r="4" fill="#60a5fa" opacity="0.45" />
      {/* Labels */}
      <text x="5" y="15" fill="#f59e0b" fontSize="9" fontFamily="monospace">vote: 2</text>
      <text x="5" y="25" fill="#60a5fa" fontSize="9" fontFamily="monospace">vote: 1</text>
      <text x="130" y="155" fill="#06b6d4" fontSize="9" fontFamily="monospace">→ Yellow wins</text>
    </svg>
  );
}

function MLPDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
      <rect width="200" height="160" fill="#020510" rx="8" />
      {/* Curved boundary */}
      <path d="M 0,80 C 50,20 150,140 200,80" fill="none" stroke="#6366f1" strokeWidth="2.5" />
      <path d="M 0,80 C 50,20 150,140 200,80 L 200,160 L 0,160 Z" fill="rgba(245,158,11,0.06)" />
      <path d="M 0,80 C 50,20 150,140 200,80 L 200,0 L 0,0 Z" fill="rgba(99,102,241,0.06)" />
      {/* Points class 0 */}
      <circle cx="30" cy="100" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="55" cy="115" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="20" cy="130" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="170" cy="100" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="155" cy="120" r="4" fill="#f59e0b" opacity="0.8" />
      {/* Points class 1 */}
      <circle cx="40" cy="30" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="80" cy="25" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="120" cy="28" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="160" cy="35" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="100" cy="140" r="4" fill="#818cf8" opacity="0.8" />
      {/* Neural net mini diagram on right */}
      <circle cx="162" cy="58" r="4" fill="none" stroke="#6366f1" strokeWidth="1" />
      <circle cx="162" cy="70" r="4" fill="none" stroke="#6366f1" strokeWidth="1" />
      <circle cx="175" cy="64" r="4" fill="none" stroke="#818cf8" strokeWidth="1" />
      <line x1="162" y1="58" x2="175" y2="64" stroke="#6366f150" strokeWidth="0.8" />
      <line x1="162" y1="70" x2="175" y2="64" stroke="#6366f150" strokeWidth="0.8" />
      <text x="5" y="15" fill="#6366f1" fontSize="9" fontFamily="monospace">non-linear</text>
      <text x="5" y="25" fill="#6366f1" fontSize="9" fontFamily="monospace">boundary</text>
    </svg>
  );
}

function LogRegDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
      <rect width="200" height="160" fill="#020510" rx="8" />
      {/* Region fills */}
      <polygon points="0,0 200,0 200,60 0,140" fill="rgba(99,102,241,0.07)" />
      <polygon points="0,140 200,60 200,160 0,160" fill="rgba(245,158,11,0.07)" />
      {/* Decision line */}
      <line x1="0" y1="140" x2="200" y2="60" stroke="#6366f1" strokeWidth="2.5" />
      {/* Sigmoid curve overlay small */}
      <path d="M 10,148 C 25,145 35,115 50,90 C 65,65 75,45 100,38 C 125,31 140,28 150,26" fill="none" stroke="#06b6d4" strokeWidth="1.5" strokeDasharray="3,3" opacity="0.6" />
      {/* Points */}
      <circle cx="30" cy="30" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="60" cy="20" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="90" cy="15" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="120" cy="10" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="40" cy="150" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="80" cy="135" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="150" cy="130" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="170" cy="145" r="4" fill="#f59e0b" opacity="0.8" />
      {/* P=0.5 label */}
      <text x="100" y="95" fill="#6366f1" fontSize="8.5" fontFamily="monospace">P=0.5</text>
      <text x="135" y="22" fill="#06b6d4" fontSize="8" fontFamily="monospace">σ(z)</text>
      <text x="5" y="15" fill="#6366f1" fontSize="9" fontFamily="monospace">linear</text>
      <text x="5" y="25" fill="#6366f1" fontSize="9" fontFamily="monospace">boundary</text>
    </svg>
  );
}

function RFDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
      <rect width="200" height="160" fill="#020510" rx="8" />
      {/* Grid splits */}
      <line x1="100" y1="0" x2="100" y2="160" stroke="rgba(20,184,166,0.3)" strokeWidth="1" strokeDasharray="3,3" />
      <line x1="0" y1="80" x2="100" y2="80" stroke="rgba(20,184,166,0.3)" strokeWidth="1" strokeDasharray="3,3" />
      <line x1="100" y1="60" x2="200" y2="60" stroke="rgba(20,184,166,0.3)" strokeWidth="1" strokeDasharray="3,3" />
      <line x1="150" y1="60" x2="150" y2="160" stroke="rgba(20,184,166,0.3)" strokeWidth="1" strokeDasharray="3,3" />
      {/* Region fills */}
      <rect x="0" y="0" width="100" height="80" fill="rgba(99,102,241,0.07)" />
      <rect x="0" y="80" width="100" height="80" fill="rgba(245,158,11,0.07)" />
      <rect x="100" y="0" width="100" height="60" fill="rgba(245,158,11,0.07)" />
      <rect x="100" y="60" width="50" height="100" fill="rgba(99,102,241,0.07)" />
      <rect x="150" y="60" width="50" height="100" fill="rgba(245,158,11,0.07)" />
      {/* Points class 0 */}
      <circle cx="30" cy="40" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="70" cy="30" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="50" cy="55" r="4" fill="#818cf8" opacity="0.8" />
      <circle cx="120" cy="100" r="4" fill="#818cf8" opacity="0.8" />
      {/* Points class 1 */}
      <circle cx="40" cy="120" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="75" cy="130" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="140" cy="30" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="170" cy="110" r="4" fill="#f59e0b" opacity="0.8" />
      <circle cx="160" cy="140" r="4" fill="#f59e0b" opacity="0.8" />
      {/* Labels */}
      <text x="5" y="15" fill="#14b8a6" fontSize="9" fontFamily="monospace">axis-aligned</text>
      <text x="5" y="25" fill="#14b8a6" fontSize="9" fontFamily="monospace">splits</text>
    </svg>
  );
}

/* ============================================================
   MODEL DATA
   ============================================================ */

const MODEL_DATA = [
  {
    id: "mlp",
    name: "Neural Network",
    subtitle: "Multi-Layer Perceptron",
    icon: "🧠",
    iconBg: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    badge: "violet",
    tagline: "Universal approximator for arbitrarily complex decision surfaces.",
    overview: `A neural network stacks layers of neurons. Each neuron computes a weighted sum of its inputs and passes it through a non-linear activation function (like ReLU or sigmoid). Chaining these transformations allows the network to learn arbitrarily complex functions — curving, bending, and folding the decision boundary to fit intricate data patterns.`,
    whenToUse: [
      "Data has complex, non-linear structure",
      "Large datasets with many features",
      "Image, text, or audio classification tasks",
    ],
    whenNot: [
      "Small datasets (high overfitting risk)",
      "When interpretability is required",
      "Data is naturally linearly separable",
    ],
    math: {
      core: [
        { label: "Forward pass (layer l)", formula: "aˡ = σ(Wˡ · aˡ⁻¹ + bˡ)" },
        { label: "ReLU activation", formula: "σ(z) = max(0, z)" },
        { label: "Cross-entropy loss", formula: "L = −Σ yᵢ · log(ŷᵢ)" },
        { label: "Backprop gradient", formula: "∂L/∂W = δˡ · (aˡ⁻¹)ᵀ" },
        { label: "SGD weight update", formula: "W ← W − η · ∂L/∂W" },
      ],
      explanation: "Gradient descent iteratively nudges each weight in the direction that reduces loss. The chain rule propagates these gradients backward through every layer."
    },
    complexity: [
      { name: "Training", value: "O(epochs · layers · n)", color: "#f59e0b" },
      { name: "Prediction", value: "O(layers · features)", color: "#10b981" },
      { name: "Overfitting Risk", value: "High without regularization", color: "#ef4444" },
    ],
    pros: ["Learns complex non-linear patterns", "State-of-the-art on many tasks", "Flexible architecture"],
    cons: ["Black-box — hard to interpret", "Needs lots of data & tuning", "Computationally expensive"],
    Diagram: MLPDiagram,
  },
  {
    id: "svm",
    name: "Support Vector Machine",
    subtitle: "SVM with RBF Kernel",
    icon: "📐",
    iconBg: "linear-gradient(135deg, #3b82f6, #06b6d4)",
    badge: "blue",
    tagline: "Finds the widest possible margin between classes.",
    overview: `An SVM finds the hyperplane that maximally separates two classes. The key insight: only the points closest to the boundary — the "support vectors" — determine the boundary. Using the kernel trick (RBF by default), SVM implicitly maps points into a higher-dimensional space where a linear separator exists, enabling non-linear decision boundaries.`,
    whenToUse: [
      "Small to medium datasets",
      "High-dimensional feature spaces (text, genomics)",
      "Clear margin of separation exists",
    ],
    whenNot: [
      "Very large datasets (O(n²) or O(n³) training)",
      "Lots of overlapping classes",
      "Need probability estimates",
    ],
    math: {
      core: [
        { label: "Optimization objective", formula: "min ½||w||² subject to yᵢ(w·xᵢ+b) ≥ 1" },
        { label: "Margin width", formula: "margin = 2 / ||w||" },
        { label: "RBF kernel", formula: "K(x,z) = exp(−γ||x−z||²)" },
        { label: "Decision function", formula: "f(x) = sign(Σ αᵢ yᵢ K(xᵢ,x) + b)" },
      ],
      explanation: "The Lagrange dual turns the problem into maximizing support vector weights (αᵢ). Only vectors with αᵢ>0 are support vectors — all others are irrelevant to the boundary."
    },
    complexity: [
      { name: "Training", value: "O(n² to n³)", color: "#ef4444" },
      { name: "Prediction", value: "O(support vectors · d)", color: "#10b981" },
      { name: "Overfitting Risk", value: "Low (max-margin)", color: "#10b981" },
    ],
    pros: ["Great generalization on small data", "Resistant to overfitting (max-margin)", "Works in high dimensions"],
    cons: ["Slow on large datasets", "Kernel & C must be tuned carefully", "No built-in probability output"],
    Diagram: SVMDiagram,
  },
  {
    id: "knn",
    name: "K-Nearest Neighbors",
    subtitle: "Instance-Based Learning",
    icon: "🔍",
    iconBg: "linear-gradient(135deg, #f59e0b, #ef4444)",
    badge: "orange",
    tagline: "Classify by majority vote of your k closest neighbors.",
    overview: `KNN is the simplest classifier: to classify a new point, find the k training points nearest to it (by Euclidean distance) and take a majority vote of their labels. There is no explicit training phase — the algorithm simply memorizes all data. The decision boundary is implicitly defined by the data distribution and is always non-linear.`,
    whenToUse: [
      "Small, low-dimensional datasets",
      "Non-linear decision boundaries",
      "When a quick baseline is needed",
    ],
    whenNot: [
      "Large datasets (slow prediction)",
      "High-dimensional data (distance concentration)",
      "Memory-constrained environments",
    ],
    math: {
      core: [
        { label: "Euclidean distance", formula: "d(x,z) = √(Σ (xᵢ−zᵢ)²)" },
        { label: "Classification vote", formula: "ŷ = argmax_c |{xᵢ ∈ Nₖ(x) : yᵢ=c}|" },
        { label: "Decision boundary", formula: "Voronoi partition of training set" },
      ],
      explanation: "The choice of k is critical. k=1 creates a very jagged boundary (overfitting), while large k smooths the boundary toward a simpler shape. Optimal k is typically found via cross-validation."
    },
    complexity: [
      { name: "Training", value: "O(1) — lazy learner", color: "#10b981" },
      { name: "Prediction", value: "O(n · d) per query", color: "#ef4444" },
      { name: "Overfitting Risk", value: "High when k=1", color: "#f59e0b" },
    ],
    pros: ["Simple, intuitive, no training", "Naturally multiclass", "Adapts to complex boundaries"],
    cons: ["Slow at prediction time", "Sensitive to irrelevant features", "Needs feature scaling"],
    Diagram: KNNDiagram,
  },
  {
    id: "logreg",
    name: "Logistic Regression",
    subtitle: "Linear Probabilistic Classifier",
    icon: "⚖️",
    iconBg: "linear-gradient(135deg, #10b981, #06b6d4)",
    badge: "green",
    tagline: "A linear boundary expressed as a calibrated probability.",
    overview: `Logistic Regression models the probability that a sample belongs to class 1 using the sigmoid function applied to a linear combination of features. It draws a straight (hyperplane) decision boundary and outputs well-calibrated probabilities. Despite the name, it is a classification — not regression — algorithm.`,
    whenToUse: [
      "Linearly separable data",
      "When probability estimates matter",
      "Interpretability is required",
    ],
    whenNot: [
      "Non-linear decision boundaries (circles, moons)",
      "Complex feature interactions",
      "Very imbalanced classes without reweighting",
    ],
    math: {
      core: [
        { label: "Log-odds (logit)", formula: "z = w₀x₀ + w₁x₁ + … + b" },
        { label: "Sigmoid function", formula: "σ(z) = 1 / (1 + e⁻ᶻ)" },
        { label: "P(y=1|x)", formula: "P = σ(wᵀx + b)" },
        { label: "Log-likelihood loss", formula: "L = −Σ [yᵢlog(ŷᵢ) + (1−yᵢ)log(1−ŷᵢ)]" },
        { label: "Decision rule", formula: "predict 1 if P ≥ 0.5 → wᵀx + b ≥ 0" },
      ],
      explanation: "The decision boundary is the set of points where z=0, i.e., wᵀx + b = 0 — a hyperplane. Moving away from it increases confidence exponentially."
    },
    complexity: [
      { name: "Training", value: "O(epochs · n · d)", color: "#10b981" },
      { name: "Prediction", value: "O(d) — dot product", color: "#10b981" },
      { name: "Overfitting Risk", value: "Low (implicit regularization)", color: "#10b981" },
    ],
    pros: ["Fast & interpretable", "Calibrated probabilities", "Works well with regularization (L1/L2)"],
    cons: ["Cannot learn non-linear boundaries", "Sensitive to outliers", "Assumes feature independence"],
    Diagram: LogRegDiagram,
  },
  {
    id: "rf",
    name: "Random Forest",
    subtitle: "Ensemble of Decision Trees",
    icon: "🌲",
    iconBg: "linear-gradient(135deg, #14b8a6, #10b981)",
    badge: "cyan",
    tagline: "Ensemble of decorrelated trees — robust, powerful, non-linear.",
    overview: `A Random Forest builds many Decision Trees, each on a random bootstrap sample of the data and a random subset of features (decorrelation). At prediction time, each tree votes and the majority wins. This ensemble strategy dramatically reduces variance compared to a single tree, yielding a powerful non-linear classifier that is hard to overfit in practice.`,
    whenToUse: [
      "Tabular data with complex interactions",
      "Feature importance is needed",
      "Robust baseline without much tuning",
    ],
    whenNot: [
      "Very low latency prediction required",
      "Interpretability of individual decisions matters",
      "Data is sparse (text, high-dim)",
    ],
    math: {
      core: [
        { label: "Bootstrap sample", formula: "Dₜ = sample n points from D with replacement" },
        { label: "Node split criterion", formula: "Gini = 1 − Σ pₖ²" },
        { label: "Feature selection per split", formula: "m = √(total features)" },
        { label: "Final prediction", formula: "ŷ = majority_vote{T₁(x), T₂(x), ..., Tₙ(x)}" },
        { label: "Bias-variance", formula: "Var(forest) = ρσ² + (1−ρ)σ²/n → ρσ² as n→∞" },
      ],
      explanation: "Decorrelation (ρ → 0) between trees is the key. Each tree sees different features, so they make different errors — when averaged, errors cancel. Bias stays the same, variance drops dramatically."
    },
    complexity: [
      { name: "Training", value: "O(n · trees · d · log n)", color: "#f59e0b" },
      { name: "Prediction", value: "O(trees · log n)", color: "#10b981" },
      { name: "Overfitting Risk", value: "Very Low (ensemble)", color: "#10b981" },
    ],
    pros: ["Robust to noise and outliers", "Built-in feature importance", "No feature scaling needed"],
    cons: ["Less interpretable than single tree", "Large memory footprint", "Slow for real-time inference"],
    Diagram: RFDiagram,
  },
];

const BADGE_COLORS = { blue: "#3b82f6", cyan: "#06b6d4", green: "#10b981", violet: "#8b5cf6", orange: "#f59e0b" };

function ModelCard({ model }) {
  const [tab, setTab] = useState("overview");
  const { Diagram } = model;

  return (
    <div className="model-card fade-up">
      {/* Header */}
      <div className="model-card-header">
        <div className="model-icon" style={{ background: model.iconBg, fontFamily: "var(--font-mono)", fontSize: "11px", fontWeight: 700, letterSpacing: "-0.02em", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center" }}>
          {model.id.toUpperCase().slice(0, 3)}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "3px" }}>
            <h3 style={{ margin: 0, fontFamily: "var(--font-display)", fontSize: "1.05rem", fontWeight: 700, color: "#fff", letterSpacing: "-0.02em" }}>
              {model.name}
            </h3>
            <span className={`alpha-badge ${model.badge}`}>{model.id.toUpperCase()}</span>
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "var(--font-display)" }}>{model.subtitle}</div>
          <div style={{ marginTop: "6px", fontSize: "0.8rem", color: "#94a3b8", fontStyle: "italic", lineHeight: 1.4 }}>
            "{model.tagline}"
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="model-tabs">
        {["overview", "math", "diagnostics"].map(t => (
          <button
            key={t}
            className={`model-tab ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Body */}
      <div className="model-card-body">
        {tab === "overview" && (
          <>
            {/* Diagram */}
            <div className="model-diagram">
              <Diagram />
            </div>

            {/* Core Idea */}
            <div>
              <div className="model-card-section-title">How it works</div>
              <p style={{ margin: 0, fontSize: "0.83rem", color: "#94a3b8", lineHeight: 1.7 }}>
                {model.overview}
              </p>
            </div>

            {/* When to use */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
              <div>
                <div className="model-card-section-title">When to use</div>
                <ul className="pro-con-list">
                  {model.whenToUse.map((w, i) => <li key={i} className="pro">{w}</li>)}
                </ul>
              </div>
              <div>
                <div className="model-card-section-title">When to avoid</div>
                <ul className="pro-con-list" style={{ listStyle: "none", padding: 0, margin: 0 }}>
                  {model.whenNot.map((w, i) => <li key={i} className="con">{w}</li>)}
                </ul>
              </div>
            </div>
          </>
        )}

        {tab === "math" && (
          <>
            <div>
              <div className="model-card-section-title">Key Equations</div>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {model.math.core.map((eq, i) => (
                  <div key={i} style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
                    <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-display)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                      {eq.label}
                    </span>
                    <div className="math-box" style={{ padding: "8px 12px", fontSize: "12px" }}>
                      {eq.formula}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="explain-box">
              <div className="explain-title">Intuition</div>
              {model.math.explanation}
            </div>
          </>
        )}

        {tab === "diagnostics" && (
          <>
            <div>
              <div className="model-card-section-title">Complexity</div>
              <div style={{ borderRadius: "10px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.06)" }}>
                {model.complexity.map((c, i) => (
                  <div key={i} className="complexity-row" style={{ padding: "9px 14px" }}>
                    <span style={{ color: "var(--text-muted)", fontSize: "0.81rem" }}>{c.name}</span>
                    <span style={{ color: c.color, fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}>{c.value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
              <div>
                <div className="model-card-section-title">Strengths</div>
                <ul className="pro-con-list" style={{ listStyle: "none", padding: 0, margin: 0 }}>
                  {model.pros.map((p, i) => <li key={i} className="pro" style={{ marginBottom: "4px" }}>+ {p}</li>)}
                </ul>
              </div>
              <div>
                <div className="model-card-section-title">Weaknesses</div>
                <ul className="pro-con-list" style={{ listStyle: "none", padding: 0, margin: 0 }}>
                  {model.cons.map((c, i) => <li key={i} className="con" style={{ marginBottom: "4px" }}>− {c}</li>)}
                </ul>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function ModelsPage() {
  return (
    <div className="app-container">
      {/* Hero */}
      <div className="page-hero">
        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
          <span className="alpha-badge blue">5 Models</span>
          <span className="alpha-badge cyan">Interactive</span>
          <span className="alpha-badge green">With Math</span>
        </div>
        <h1 className="page-hero-title">Model Encyclopedia</h1>
        <p className="page-hero-sub">
          Deep-dive into every algorithm available in the lab. Understand the math, the intuition, when to use each model, and why it fails on certain datasets. Click <strong>Math</strong> for the equations, or <strong>Diagnostics</strong> for complexity & trade-offs.
        </p>
      </div>

      {/* Comparison quick-ref */}
      <div className="glass-panel" style={{ padding: "20px 24px", marginBottom: "32px" }}>
        <div style={{ fontSize: "0.72rem", color: "var(--alpha-cyan)", fontFamily: "var(--font-display)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "14px" }}>
          Quick Comparison
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
            <thead>
              <tr style={{ color: "var(--text-muted)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                <th style={{ textAlign: "left", padding: "8px 12px", fontWeight: 600 }}>Model</th>
                <th style={{ textAlign: "center", padding: "8px 12px", fontWeight: 600 }}>Boundary</th>
                <th style={{ textAlign: "center", padding: "8px 12px", fontWeight: 600 }}>Train Speed</th>
                <th style={{ textAlign: "center", padding: "8px 12px", fontWeight: 600 }}>Interpretable</th>
                <th style={{ textAlign: "center", padding: "8px 12px", fontWeight: 600 }}>Overfit Risk</th>
              </tr>
            </thead>
            <tbody>
              {[
                { name: "Neural Net", boundary: "Non-linear", train: "Slow", interp: "No", overfit: "High" },
                { name: "SVM", boundary: "Non-linear*", train: "Medium", interp: "Partial", overfit: "Low" },
                { name: "KNN", boundary: "Non-linear", train: "Instant", interp: "Yes", overfit: "Med" },
                { name: "LogReg", boundary: "Linear", train: "Fast", interp: "Yes", overfit: "Low" },
                { name: "Rand. Forest", boundary: "Non-linear", train: "Medium", interp: "Partial", overfit: "Very Low" },
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)", transition: "background 0.2s" }}
                  onMouseEnter={e => e.currentTarget.style.background = "rgba(59,130,246,0.04)"}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                  <td style={{ padding: "9px 12px", color: "#e2e8f0", fontWeight: 500 }}>{row.name}</td>
                  <td style={{ padding: "9px 12px", textAlign: "center", color: row.boundary === "Linear" ? "#6ee7b7" : "#a5f3fc" }}>{row.boundary}</td>
                  <td style={{ padding: "9px 12px", textAlign: "center", color: row.train === "Slow" ? "#fca5a5" : row.train === "Fast" || row.train === "Instant" ? "#6ee7b7" : "#fcd34d" }}>{row.train}</td>
                  <td style={{ padding: "9px 12px", textAlign: "center", color: row.interp === "Yes" ? "#6ee7b7" : row.interp === "No" ? "#fca5a5" : "#fcd34d" }}>{row.interp}</td>
                  <td style={{ padding: "9px 12px", textAlign: "center", color: row.overfit === "High" ? "#fca5a5" : row.overfit === "Low" || row.overfit === "Very Low" ? "#6ee7b7" : "#fcd34d" }}>{row.overfit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginTop: "10px" }}>* With RBF kernel — linear kernel gives a linear boundary.</div>
      </div>

      {/* Model cards grid */}
      <div className="models-grid">
        {MODEL_DATA.map(m => <ModelCard key={m.id} model={m} />)}
      </div>

      {/* Footer note */}
      <div style={{ marginTop: "48px", textAlign: "center", color: "var(--text-dim)", fontSize: "0.8rem", lineHeight: 1.7, paddingBottom: "40px" }}>
        All five models are available in the Lab. Train them side-by-side and observe how the decision boundary adapts to each dataset.<br />
        The visual traces (hover over the canvas) reveal the live math for each model in real time.
      </div>
    </div>
  );
}
