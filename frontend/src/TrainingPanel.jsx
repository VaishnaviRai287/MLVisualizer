import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";

/* ============================================================
   MODEL META — explanations for each model
   ============================================================ */
const MODEL_META = {
  mlp: {
    name: "Neural Network (MLP)",
    boundaryType: "Non-linear",
    badge: "violet",
    color: "#8b5cf6",
    shortDesc: "Stacked layers of neurons learning non-linear patterns via backpropagation.",
    formula: "aˡ = σ(Wˡ·aˡ⁻¹ + bˡ)",
    hoverHint: "Boundary curves due to cascaded non-linear activations across all layers.",
  },
  svm: {
    name: "Support Vector Machine",
    boundaryType: "Non-linear (RBF)",
    badge: "blue",
    color: "#3b82f6",
    shortDesc: "Finds the hyperplane with maximum margin between classes.",
    formula: "f(x) = sign(Σαᵢyᵢ K(xᵢ,x) + b)",
    hoverHint: "Lines show the 3 nearest support vectors that anchor the margin.",
  },
  knn: {
    name: "K-Nearest Neighbors",
    boundaryType: "Non-linear (Voronoi)",
    badge: "orange",
    color: "#f59e0b",
    shortDesc: "Classifies by majority vote of the k closest training points.",
    formula: "ŷ = majority vote of k-NN",
    hoverHint: "Lines connect your cursor to the 3 nearest neighbors casting votes.",
  },
  logreg: {
    name: "Logistic Regression",
    boundaryType: "Linear",
    badge: "green",
    color: "#10b981",
    shortDesc: "Models P(class=1|x) via sigmoid of a linear combination of features.",
    formula: "P(y=1|x) = σ(w₀x₀ + w₁x₁ + b)",
    hoverHint: "The glowing line is where P=0.5 — the exact decision boundary.",
  },
  rf: {
    name: "Random Forest",
    boundaryType: "Non-linear (axis-aligned)",
    badge: "cyan",
    color: "#06b6d4",
    shortDesc: "Ensemble of decision trees, each voting on the majority class.",
    formula: "ŷ = majority{T₁(x), T₂(x), …, Tₙ(x)}",
    hoverHint: "The highlighted rectangle shows the decision cell your cursor is in.",
  },
};

const CONFUSION_DEFS = {
  TP: {
    label: "True Positive",
    color: "#10b981",
    def: "Model predicted class 1, and the ground truth is class 1. A correct positive prediction.",
    formula: "Precision uses TP: P = TP / (TP + FP)",
  },
  FP: {
    label: "False Positive",
    color: "#ef4444",
    def: "Model predicted class 1, but ground truth is class 0. Also called a Type I error.",
    formula: "Precision penalizes FP: P = TP / (TP + FP)",
  },
  TN: {
    label: "True Negative",
    color: "#10b981",
    def: "Model predicted class 0, and the ground truth is class 0. A correct negative prediction.",
    formula: "Specificity uses TN: Spec = TN / (TN + FP)",
  },
  FN: {
    label: "False Negative",
    color: "#ef4444",
    def: "Model predicted class 0, but ground truth is class 1. Also called a Type II error (miss).",
    formula: "Recall penalizes FN: R = TP / (TP + FN)",
  },
};

const DATASET_DEFS = {
  moons: {
    name: "Two Moons",
    desc: "Two interleaved crescent-shaped clusters. Requires a non-linear boundary — linear models will fail significantly here.",
    structure: "Non-linear",
    bestFor: ["MLP", "KNN", "SVM (RBF)", "RF"],
    worstFor: ["LogReg"],
  },
  circles: {
    name: "Concentric Circles",
    desc: "One class wraps around the other in concentric rings. No straight line can ever separate them.",
    structure: "Radially non-linear",
    bestFor: ["SVM (RBF)", "MLP", "KNN"],
    worstFor: ["LogReg"],
  },
  blobs: {
    name: "Gaussian Blobs",
    desc: "Two well-separated Gaussian clusters. A linear model is perfectly adequate and even preferred here.",
    structure: "Linearly separable",
    bestFor: ["LogReg", "SVM", "MLP"],
    worstFor: ["(all work — LogReg is most interpretable)"],
  },
};

/* ============================================================
   MAIN COMPONENT
   ============================================================ */
export default function TrainingPanel({ title, onStateChange }) {
  const [data, setData] = useState([]);
  const [boundary, setBoundary] = useState(null);
  const [dataset, setDataset] = useState("moons");
  const [model, setModel] = useState("mlp");
  const [accuracy, setAccuracy] = useState(null);
  const [points, setPoints] = useState([]);
  const [labels, setLabels] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [range, setRange] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [hoverPos, setHoverPos] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(1);
  const [scrubEpoch, setScrubEpoch] = useState(null);
  const [activeFilter, setActiveFilter] = useState(null);
  const [selectedCell, setSelectedCell] = useState(null);
  const [showDatasetInfo, setShowDatasetInfo] = useState(false);

  const [customDatasets, setCustomDatasets] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [saveStatus, setSaveStatus] = useState("");

  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const accuracyHistory = useRef([]);
  const historicalStates = useRef([]);
  const offscreenCanvasRef = useRef(null);
  const visualState = useRef({ points: [], labels: [], predictions: [], range: null, metadata: null, hoverPos: null, model: "mlp", activeFilter: null });
  const dragTargetIndex = useRef(null);
  const isDragging = useRef(false);
  const dragDebounceRef = useRef(null);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (token) {
      axios.get("http://localhost:8000/api/datasets/", {
        headers: { Authorization: `Bearer ${token}` }
      }).then(res => setCustomDatasets(res.data)).catch(() => {});
    }
  }, []);

  useEffect(() => {
    if (onStateChange) onStateChange({ model, dataset, accuracy, status });
  }, [model, dataset, accuracy, status, onStateChange]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("csv_file", file);
    formData.append("name", file.name.split(".")[0]);
    const token = localStorage.getItem("access_token");
    try {
      setUploadStatus("Uploading...");
      const res = await axios.post("http://localhost:8000/api/datasets/", formData, {
        headers: { "Content-Type": "multipart/form-data", Authorization: `Bearer ${token}` }
      });
      setCustomDatasets(prev => [res.data, ...prev]);
      setDataset(res.data.name);
      setUploadStatus("Uploaded!");
      setTimeout(() => setUploadStatus(""), 2000);
    } catch {
      setUploadStatus("Upload failed.");
      setTimeout(() => setUploadStatus(""), 2000);
    }
  };

  const saveExperiment = async () => {
    if (!accuracy || !model) return;
    const token = localStorage.getItem("access_token");
    if (!token) return;
    try {
      setSaveStatus("Saving...");
      await axios.post("http://localhost:8000/api/experiments/", {
        dataset_name: dataset, model_type: model, accuracy
      }, { headers: { Authorization: `Bearer ${token}` } });
      setSaveStatus("Saved!");
      setTimeout(() => setSaveStatus(""), 2500);
    } catch {
      setSaveStatus("Error.");
      setTimeout(() => setSaveStatus(""), 2000);
    }
  };

  const ingestFrame = useCallback((newData) => {
    setCurrentEpoch(newData.epoch + 1);
    if (newData.total_epochs) setTotalEpochs(newData.total_epochs);
    setData(prev => [...prev, { epoch: newData.epoch, loss: newData.loss }]);
    if (newData.boundary) setBoundary(newData.boundary);
    if (newData.accuracy !== null) {
      setAccuracy(newData.accuracy);
      accuracyHistory.current.push(newData.accuracy);
      if (accuracyHistory.current.length > 5) accuracyHistory.current.shift();
      if (["mlp", "rf"].includes(newData.model_name || model)) {
        if (accuracyHistory.current.length === 5 && new Set(accuracyHistory.current).size === 1) {
          setStatus("Converged");
        } else {
          setStatus("Learning...");
        }
      } else {
        setStatus("Done");
      }
    }
    if (newData.points) { setPoints(newData.points); setLabels(newData.labels); }
    if (newData.predictions) setPredictions(newData.predictions);
    if (newData.range) setRange(newData.range);
    if (newData.metadata) setMetadata(newData.metadata);
    historicalStates.current.push({
      boundary: newData.boundary,
      accuracy: newData.accuracy,
      metadata: newData.metadata,
      points: newData.points,
      labels: newData.labels,
      predictions: newData.predictions,
      range: newData.range
    });
  }, [model]);

  const startTraining = () => {
    setData([]);
    setBoundary(null);
    setCurrentEpoch(0);
    setScrubEpoch(null);
    setActiveFilter(null);
    setSelectedCell(null);
    setStatus("Initializing...");
    accuracyHistory.current = [];
    historicalStates.current = [];
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/train/${title}/`);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const newData = JSON.parse(e.data);
      ingestFrame(newData);
    };
    ws.onopen = () => {
      ws.send(JSON.stringify({ action: "train", epochs: 50, dataset, model }));
    };
  };

  // Offscreen heatmap
  useEffect(() => {
    if (!boundary || boundary.length === 0) return;
    const rows = boundary.length;
    const cols = boundary[0].length;
    const offscreen = document.createElement("canvas");
    offscreen.width = cols;
    offscreen.height = rows;
    const offCtx = offscreen.getContext("2d");
    const imgData = offCtx.createImageData(cols, rows);
    for (let i = 0; i < rows; i++) {
      const drawY = rows - 1 - i;
      for (let j = 0; j < cols; j++) {
        const prob = boundary[i][j];
        const p = Math.max(0, Math.min(1, prob));
        const idx = (drawY * cols + j) * 4;
        let r, g, b;
        if (p > 0.48 && p < 0.52) {
          r = 255; g = 255; b = 255;
        } else {
          // Blue → (uncertainty gray) → Amber
          r = Math.round(245 + p * (59 - 245));
          g = Math.round(158 + p * (130 - 158));
          b = Math.round(11  + p * (246 - 11));
        }
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    offCtx.putImageData(imgData, 0, 0);
    offscreenCanvasRef.current = offscreen;
  }, [boundary]);

  useEffect(() => {
    visualState.current = { points, labels, predictions, range, metadata, hoverPos, model, activeFilter };
  }, [points, labels, predictions, range, metadata, hoverPos, model, activeFilter]);

  // 60fps render loop
  useEffect(() => {
    let animId;
    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) { animId = requestAnimationFrame(render); return; }
      const ctx = canvas.getContext("2d");
      const { points: pts, labels: lbls, predictions: preds, range: rr, metadata: md, hoverPos: hp, model: m, activeFilter: af } = visualState.current;
      if (offscreenCanvasRef.current) {
        ctx.imageSmoothingEnabled = true;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(offscreenCanvasRef.current, 0, 0, canvas.width, canvas.height);
      }
      if (rr && pts) {
        const { xMin, xMax, yMin, yMax } = rr;
        pts.forEach((p, i) => {
          const x = ((p[0] - xMin) / (xMax - xMin)) * canvas.width;
          const y = canvas.height - ((p[1] - yMin) / (yMax - yMin)) * canvas.height;
          const isMisclassified = preds.length > 0 && preds[i] !== lbls[i];
          let category = null;
          if (preds.length > 0) {
            if (lbls[i] === 1 && preds[i] === 1) category = "TP";
            else if (lbls[i] === 0 && preds[i] === 1) category = "FP";
            else if (lbls[i] === 0 && preds[i] === 0) category = "TN";
            else if (lbls[i] === 1 && preds[i] === 0) category = "FN";
          }
          const dimmed = af && category !== af;
          ctx.globalAlpha = dimmed ? 0.12 : 1.0;
          ctx.beginPath();
          ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
          ctx.fillStyle = lbls[i] === 0 ? "#f59e0b" : "#3b82f6";
          ctx.fill();
          if (!dimmed && af && category === af) {
            const pulse = (Math.sin(Date.now() / 100) + 1) / 2;
            ctx.strokeStyle = af === "FP" || af === "FN" ? `rgba(239,68,68,${0.5 + pulse * 0.5})` : `rgba(16,185,129,${0.5 + pulse * 0.5})`;
            ctx.lineWidth = 3;
            ctx.shadowColor = ctx.strokeStyle;
            ctx.shadowBlur = 8 + pulse * 8;
          } else if (!dimmed && isMisclassified) {
            const pulse = (Math.sin(Date.now() / 150) + 1) / 2;
            ctx.strokeStyle = `rgba(239,68,68,${0.4 + pulse * 0.6})`;
            ctx.lineWidth = 2.5;
            ctx.shadowColor = "#ef4444";
            ctx.shadowBlur = 5 + pulse * 5;
          } else if (!dimmed) {
            ctx.strokeStyle = "rgba(255,255,255,0.6)";
            ctx.lineWidth = 1.5;
            ctx.shadowBlur = 0;
          }
          ctx.stroke();
          ctx.shadowBlur = 0;
          ctx.globalAlpha = 1.0;
        });

        // Visual reasoning traces
        if (m === "svm" && hp && md?.support_vectors) {
          const svs = md.support_vectors.map(sv => {
            const sx = ((sv[0] - xMin) / (xMax - xMin)) * canvas.width;
            const sy = canvas.height - ((sv[1] - yMin) / (yMax - yMin)) * canvas.height;
            return { sx, sy, d: Math.hypot(sx - hp.cx, sy - hp.cy) };
          }).sort((a, b) => a.d - b.d).slice(0, 3);
          svs.forEach(sv => {
            ctx.beginPath();
            ctx.moveTo(hp.cx, hp.cy);
            ctx.lineTo(sv.sx, sv.sy);
            ctx.strokeStyle = "rgba(59,130,246,0.5)";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.arc(sv.sx, sv.sy, 6, 0, 2 * Math.PI);
            ctx.strokeStyle = "#3b82f6";
            ctx.shadowColor = "#3b82f6";
            ctx.shadowBlur = 10;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.shadowBlur = 0;
          });
        }
        if (m === "knn" && hp && pts) {
          const distances = pts.map((pt, idx) => ({
            idx, d: Math.pow(pt[0] - hp.realX, 2) + Math.pow(pt[1] - hp.realY, 2)
          })).sort((a, b) => a.d - b.d).slice(0, 3);
          const maxD = distances[distances.length - 1]?.d || 1;
          distances.forEach(neighbor => {
            const pt = pts[neighbor.idx];
            const nx = ((pt[0] - xMin) / (xMax - xMin)) * canvas.width;
            const ny = canvas.height - ((pt[1] - yMin) / (yMax - yMin)) * canvas.height;
            const opacity = Math.max(0.25, 1 - (neighbor.d / (maxD * 1.5)));
            ctx.beginPath();
            ctx.moveTo(hp.cx, hp.cy);
            ctx.lineTo(nx, ny);
            ctx.strokeStyle = lbls[neighbor.idx] === 0 ? `rgba(245,158,11,${opacity})` : `rgba(59,130,246,${opacity})`;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([2, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
          });
        }
        if (m === "rf" && hp) {
          ctx.fillStyle = "rgba(6,182,212,0.06)";
          ctx.fillRect(hp.cx - 25, hp.cy - 25, 50, 50);
          ctx.strokeStyle = "rgba(6,182,212,0.7)";
          ctx.lineWidth = 2;
          ctx.strokeRect(hp.cx - 25, hp.cy - 25, 50, 50);
        }
        if (m === "logreg" && hp && md?.weights) {
          const w0 = md.weights[0], w1 = md.weights[1], b = md.bias;
          const getPy = (px) => (-w0 * px - b) / w1;
          const cx1 = 0, cy1 = canvas.height - ((getPy(xMin) - yMin) / (yMax - yMin)) * canvas.height;
          const cx2 = canvas.width, cy2 = canvas.height - ((getPy(xMax) - yMin) / (yMax - yMin)) * canvas.height;
          ctx.beginPath();
          ctx.moveTo(cx1, cy1);
          ctx.lineTo(cx2, cy2);
          ctx.strokeStyle = "rgba(255,255,255,0.9)";
          ctx.lineWidth = 2.5;
          ctx.shadowColor = "#fff";
          ctx.shadowBlur = 8;
          ctx.stroke();
          ctx.shadowBlur = 0;
        }
        if (m === "mlp" && hp) {
          ctx.beginPath();
          ctx.arc(hp.cx, hp.cy, 30, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(139,92,246,0.06)";
          ctx.fill();
          ctx.strokeStyle = "rgba(139,92,246,0.3)";
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        if (isDragging.current && dragTargetIndex.current !== null) {
          const dp = pts[dragTargetIndex.current];
          if (dp) {
            const dx = ((dp[0] - xMin) / (xMax - xMin)) * canvas.width;
            const dy = canvas.height - ((dp[1] - yMin) / (yMax - yMin)) * canvas.height;
            ctx.beginPath();
            ctx.arc(dx, dy, 10, 0, 2 * Math.PI);
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.shadowColor = "#fff";
            ctx.shadowBlur = 12;
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        }
      }
      animId = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(animId);
  }, []);

  // Mouse handlers
  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top, rect };
  };
  const canvasToData = (x, y, rect) => {
    if (!range) return null;
    return {
      realX: range.xMin + (x / rect.width) * (range.xMax - range.xMin),
      realY: range.yMin + ((rect.height - y) / rect.height) * (range.yMax - range.yMin),
    };
  };
  const findNearestPoint = (cx, cy, rect) => {
    if (!points || !range) return null;
    const { xMin, xMax, yMin, yMax } = range;
    let best = null, bestDist = 12;
    points.forEach((p, i) => {
      const px = ((p[0] - xMin) / (xMax - xMin)) * rect.width;
      const py = rect.height - ((p[1] - yMin) / (yMax - yMin)) * rect.height;
      const d = Math.hypot(cx - px, cy - py);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
  };
  const handleMouseDown = (e) => {
    const { x, y, rect } = getCanvasCoords(e);
    const idx = findNearestPoint(x, y, rect);
    if (idx !== null) { dragTargetIndex.current = idx; isDragging.current = true; }
  };
  const handleMouseUp = () => {
    isDragging.current = false;
    dragTargetIndex.current = null;
    if (dragDebounceRef.current) clearTimeout(dragDebounceRef.current);
  };
  const handleMouseMove = (e) => {
    const { x, y, rect } = getCanvasCoords(e);
    if (isDragging.current && dragTargetIndex.current !== null) {
      const coords = canvasToData(x, y, rect);
      if (!coords) return;
      setPoints(prev => {
        const updated = [...prev];
        updated[dragTargetIndex.current] = [coords.realX, coords.realY];
        return updated;
      });
      if (dragDebounceRef.current) clearTimeout(dragDebounceRef.current);
      dragDebounceRef.current = setTimeout(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ action: "update_point", index: dragTargetIndex.current, new_coords: [coords.realX, coords.realY] }));
        }
      }, 80);
      return;
    }
    if (!range || !boundary) { setHoverPos(null); return; }
    if (activeFilter) { setHoverPos(null); return; }
    const realX = range.xMin + (x / rect.width) * (range.xMax - range.xMin);
    const realY = range.yMin + ((rect.height - y) / rect.height) * (range.yMax - range.yMin);
    const cols = boundary[0].length, rows = boundary.length;
    const gridX = Math.min(cols - 1, Math.max(0, Math.floor((x / rect.width) * cols)));
    const gridY = Math.min(rows - 1, Math.max(0, rows - 1 - Math.floor((y / rect.height) * rows)));
    const p = boundary[gridY][gridX];
    const prediction = p >= 0.5 ? 1 : 0;
    const confidence = Math.round(Math.abs(p - 0.5) * 200);
    const meta = MODEL_META[model] || {};
    const uncertaintyInsight = confidence < 30
      ? "Near the decision boundary — the model is uncertain here (P≈0.5)."
      : confidence > 70
      ? "Deep inside a class region — the model is confident here."
      : "Transitional zone — moderate confidence.";
    const modelInsights = meta.hoverHint || "";
    let mathContext = null;
    if (model === "knn" && points && range) {
      const top3 = points.map((pt, i) => ({ i, d: Math.sqrt(Math.pow(pt[0] - realX, 2) + Math.pow(pt[1] - realY, 2)) })).sort((a, b) => a.d - b.d).slice(0, 3);
      mathContext = {
        formula: "d = √((x₂−x₁)² + (y₂−y₁)²)",
        rows: top3.map(n => ({ label: `Neighbor ${n.i}`, value: `d=${n.d.toFixed(3)}`, vote: labels[n.i] === 0 ? "Amber" : "Blue" }))
      };
    } else if (model === "logreg" && metadata?.weights) {
      const [w0, w1] = metadata.weights;
      const b = metadata.bias;
      const logit = w0 * realX + w1 * realY + b;
      const prob = (1 / (1 + Math.exp(-logit))).toFixed(3);
      mathContext = {
        formula: "σ(z) = 1 / (1 + e⁻ᶻ)",
        rows: [
          { label: "w₀, w₁, b", value: `${w0.toFixed(2)}, ${w1.toFixed(2)}, ${b.toFixed(2)}` },
          { label: "z (logit)", value: logit.toFixed(3) },
          { label: "P(class=1)", value: prob }
        ]
      };
    } else if (model === "svm" && metadata?.support_vectors) {
      mathContext = {
        formula: "margin = 2 / ||w||",
        rows: [{ label: "Support Vectors", value: `${metadata.support_vectors.length} anchors` }]
      };
    }
    setHoverPos({ cx: x, cy: y, cWidth: rect.width, cHeight: rect.height, realX, realY, prediction, confidence, modelInsights, uncertaintyInsight, mathContext });
  };

  // Confusion matrix
  const confusionMatrix = (() => {
    if (!predictions.length || !labels.length) return null;
    let TP = 0, FP = 0, TN = 0, FN = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (labels[i] === 1 && predictions[i] === 1) TP++;
      else if (labels[i] === 0 && predictions[i] === 1) FP++;
      else if (labels[i] === 0 && predictions[i] === 0) TN++;
      else if (labels[i] === 1 && predictions[i] === 0) FN++;
    }
    const precision = TP + FP > 0 ? (TP / (TP + FP)) : null;
    const recall = TP + FN > 0 ? (TP / (TP + FN)) : null;
    const f1 = precision !== null && recall !== null && (precision + recall) > 0
      ? (2 * precision * recall) / (precision + recall)
      : null;
    return { TP, FP, TN, FN, precision, recall, f1 };
  })();

  const generateInsights = () => {
    const insights = [];
    if (!accuracy || !model) return insights;
    if (accuracy === 1.0) {
      insights.push({ type: "success", text: "Perfect score: all points classified correctly." });
      if (model === "mlp" || model === "rf") insights.push({ type: "danger", text: "Warning: 100% accuracy often means overfitting. The boundary is wrapping every point — it will likely fail on unseen data." });
    } else if (accuracy < 0.6) {
      insights.push({ type: "danger", text: "Accuracy below 60% — the model is near-random." });
      if (model === "logreg" && ["circles", "moons"].includes(dataset)) {
        insights.push({ type: "warning", text: "Structural mismatch: Logistic Regression can only draw a straight line. The moons/circles datasets require a curved boundary — a linear model will always fail here." });
      }
    }
    if (predictions && labels && predictions.length > 0) {
      let errs = 0;
      for (let i = 0; i < predictions.length; i++) if (predictions[i] !== labels[i]) errs++;
      if (errs > 0) insights.push({ type: "danger", text: `${errs} point${errs > 1 ? "s" : ""} are glowing red — misclassified. Click "FP" or "FN" in the matrix to spotlight them.` });
    }
    if (model === "svm" && metadata?.support_vectors) {
      insights.push({ type: "info", text: `${metadata.support_vectors.length} support vectors physically define the margin. Every other training point is irrelevant to the boundary.` });
    }
    if (model === "knn") insights.push({ type: "info", text: "Hover the canvas to see the 3 nearest voters controlling each local decision zone." });
    if (model === "logreg" && metadata?.weights) insights.push({ type: "info", text: "The glowing line shows exactly where w₀x₀ + w₁x₁ + b = 0. Confidence increases exponentially as you move away from it." });
    return insights;
  };
  const insights = generateInsights();

  const meta = MODEL_META[model] || {};
  const datasetDef = DATASET_DEFS[dataset];
  const currentStatus = accuracy ? status : "Idle";
  const statusColor = currentStatus === "Converged" || currentStatus === "Done" ? "#10b981" : currentStatus === "Learning..." ? "#f59e0b" : currentStatus === "Initializing..." ? "#3b82f6" : "var(--text-muted)";

  // Cell click handler
  const handleCellClick = (category) => {
    setActiveFilter(prev => prev === category ? null : category);
    setSelectedCell(prev => prev === category ? null : category);
  };

  const MatrixCell = ({ label, value, category, color }) => (
    <div
      className="matrix-cell"
      onClick={() => handleCellClick(category)}
      style={{
        flex: 1,
        textAlign: "center",
        padding: "10px 6px",
        background: activeFilter === category ? `${color}22` : "rgba(255,255,255,0.025)",
        border: `1px solid ${activeFilter === category ? color : "rgba(255,255,255,0.06)"}`,
        borderRadius: "10px",
        cursor: "pointer",
        transition: "all 0.2s",
      }}
    >
      <div style={{ fontSize: "10px", color: "var(--text-muted)", marginBottom: "3px", fontFamily: "var(--font-display)", letterSpacing: "0.06em" }}>{label}</div>
      <div style={{ fontSize: "20px", fontWeight: 700, color, fontFamily: "var(--font-mono)" }}>{value ?? "--"}</div>
    </div>
  );

  return (
    <div className="glass-panel training-panel">
      {/* ── HEADER ────────────────────────────────────────── */}
      <div className="panel-header">
        <div>
          <h2 className="panel-title" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            Model {title}
            {accuracy && <span className={`alpha-badge ${meta.badge || "blue"}`}>{(accuracy * 100).toFixed(1)}%</span>}
          </h2>
          {accuracy && <div style={{ fontSize: "0.73rem", color: "var(--text-muted)", marginTop: "3px" }}>{meta.name}</div>}
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          <button className="btn-train" onClick={startTraining}>Run</button>
          <button
            className="btn-train"
            style={{ background: "rgba(255,255,255,0.06)", boxShadow: "none", border: "1px solid rgba(255,255,255,0.1)" }}
            onClick={saveExperiment}
            disabled={!accuracy}
          >
            {saveStatus || "Save"}
          </button>
        </div>
      </div>

      {/* ── MODEL SHORT DESC ─────────────────────────────── */}
      {model && (
        <div className="explain-box">
          <div>
            <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "#e2e8f0", marginBottom: "3px", fontFamily: "var(--font-display)" }}>{meta.name}</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.5 }}>{meta.shortDesc}</div>
            <div className="formula-block" style={{ marginTop: "6px", fontSize: "11px" }}>{meta.formula}</div>
          </div>
        </div>
      )}

      {/* ── CONTROLS ─────────────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        <div className="controls-row">
          <select className="styled-select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <optgroup label="Default Datasets">
              <option value="moons">Two Moons</option>
              <option value="circles">Concentric Circles</option>
              <option value="blobs">Gaussian Blobs</option>
            </optgroup>
            {customDatasets.length > 0 && (
              <optgroup label="Your Datasets">
                {customDatasets.map(d => <option key={d.id} value={d.name}>{d.name}</option>)}
              </optgroup>
            )}
          </select>
          <label className="btn-train" style={{ background: "rgba(255,255,255,0.06)", cursor: "pointer", textAlign: "center", boxShadow: "none", border: "1px solid rgba(255,255,255,0.1)", fontSize: "0.8rem", padding: "9px 12px" }}>
            {uploadStatus || "↑ CSV"}
            <input type="file" accept=".csv" style={{ display: "none" }} onChange={handleFileUpload} />
          </label>
          <select className="styled-select" value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="mlp">Neural Net (MLP)</option>
            <option value="svm">Support Vector Machine</option>
            <option value="rf">Random Forest</option>
            <option value="logreg">Logistic Regression</option>
            <option value="knn">K-Nearest Neighbors</option>
          </select>
        </div>

        {/* Dataset info toggle */}
        {datasetDef && (
          <button
            onClick={() => setShowDatasetInfo(v => !v)}
            style={{ background: "none", border: "1px solid rgba(255,255,255,0.06)", color: "var(--text-muted)", padding: "5px 12px", borderRadius: "8px", fontSize: "0.73rem", cursor: "pointer", textAlign: "left", fontFamily: "var(--font-body)" }}
          >
            {showDatasetInfo ? "▲" : "▼"} About "{datasetDef.name}" dataset
          </button>
        )}
        {showDatasetInfo && datasetDef && (
          <div className="explain-box" style={{ animation: "fadeIn 0.25s ease-out" }}>
            <div className="explain-title">Dataset: {datasetDef.name}</div>
            <div style={{ marginBottom: "8px", lineHeight: 1.6 }}>{datasetDef.desc}</div>
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Structure:</span>
              <span className={`alpha-badge ${datasetDef.structure.includes("linear") && !datasetDef.structure.includes("Non") ? "green" : "orange"}`}>{datasetDef.structure}</span>
            </div>
            <div style={{ marginTop: "8px", fontSize: "0.73rem", color: "var(--text-muted)" }}>
              Best models: <span style={{ color: "#6ee7b7" }}>{datasetDef.bestFor.join(", ")}</span>
              {datasetDef.worstFor[0] !== "(all work — LogReg is most interpretable)" &&
                <> &nbsp;·&nbsp; Avoid: <span style={{ color: "#fca5a5" }}>{datasetDef.worstFor.join(", ")}</span></>
              }
            </div>
          </div>
        )}
      </div>

      {/* ── STATUS + PROGRESS ────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        <div style={{ display: "flex", gap: "10px" }}>
          {/* Status */}
          <div className="metric-bar" style={{ flex: 1 }}>
            <div className="metric-explain-wrap">
              <div className="metric-label-row">
                <span>Status</span>
              </div>
              <div className="metric-explain">
                {currentStatus === "Learning..." && "Weights are updating each epoch via gradient descent."}
                {currentStatus === "Converged" && "Loss has plateaued — no more improvement expected."}
                {currentStatus === "Done" && "Non-iterative model: fits in a single pass."}
                {currentStatus === "Idle" && "Press Run to start training."}
                {currentStatus === "Initializing..." && "Building the model and loading data..."}
              </div>
            </div>
            <span className="metric-value" style={{ color: statusColor }}>{currentStatus}</span>
          </div>

          {/* Accuracy */}
          <div className="metric-bar" style={{ flex: 1 }}>
            <div className="metric-explain-wrap">
              <div className="metric-label-row">
                <span>Accuracy</span>
              </div>
              <div className="metric-explain">
                Validation acc = correct / total. High ≠ good if overfitting.
              </div>
            </div>
            <span className="metric-value" style={{ color: accuracy > 0.9 ? "#10b981" : accuracy > 0.7 ? "#f59e0b" : accuracy ? "#ef4444" : "var(--text-muted)" }}>
              {accuracy ? `${(accuracy * 100).toFixed(1)}%` : "--"}
            </span>
          </div>
        </div>

        {/* Epoch progress */}
        {currentEpoch > 0 && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "4px" }}>
              <span>Epoch {currentEpoch} / {totalEpochs}</span>
              <span>{Math.round((currentEpoch / totalEpochs) * 100)}%</span>
            </div>
            <div className="training-progress-bar">
              <div className="training-progress-fill" style={{ width: `${(currentEpoch / totalEpochs) * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      {/* ── VISUALIZATION ────────────────────────────────── */}
      <div className="viz-container">
        {/* Canvas */}
        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef}
            width={320}
            height={320}
            className="ml-canvas"
            style={{ cursor: isDragging.current ? "grabbing" : "crosshair" }}
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseOut={() => { setHoverPos(null); handleMouseUp(); }}
          />

          {/* Probe Tooltip */}
          {hoverPos && !activeFilter && (
            <div className="probe-tooltip" style={{
              left: hoverPos.cx > hoverPos.cWidth / 2 ? hoverPos.cx - 250 : hoverPos.cx + 18,
              top: hoverPos.cy > hoverPos.cHeight / 2 ? hoverPos.cy - 190 : hoverPos.cy + 18,
              maxWidth: "240px",
            }}>
              <div style={{ fontWeight: 700, color: "#fff", marginBottom: "8px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "5px", fontFamily: "var(--font-display)", fontSize: "0.82rem" }}>
                Probe Region
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px" }}>
                <span style={{ color: "var(--text-muted)" }}>Coords:</span>
                <span>({hoverPos.realX.toFixed(2)}, {hoverPos.realY.toFixed(2)})</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px" }}>
                <span style={{ color: "var(--text-muted)" }}>Prediction:</span>
                <span style={{ color: hoverPos.prediction === 0 ? "#f59e0b" : "#3b82f6", fontWeight: 700 }}>
                  {hoverPos.prediction === 0 ? "Amber (0)" : "Blue (1)"}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                <span style={{ color: "var(--text-muted)" }}>Confidence:</span>
                <span style={{ color: hoverPos.confidence < 30 ? "#f59e0b" : "#10b981", fontFamily: "var(--font-mono)" }}>
                  {hoverPos.confidence}%
                </span>
              </div>
              <div style={{ color: "#93c5fd", fontSize: "11px", lineHeight: 1.5, marginBottom: "5px" }}>{hoverPos.uncertaintyInsight}</div>
              <div style={{ color: "#a5f3fc", fontSize: "11px", lineHeight: 1.5, borderTop: "1px solid rgba(255,255,255,0.07)", paddingTop: "5px" }}>{hoverPos.modelInsights}</div>
            </div>
          )}

          {/* Math Context Panel */}
          {hoverPos?.mathContext && !activeFilter && (
            <div style={{
              position: "absolute",
              left: hoverPos.cx > hoverPos.cWidth / 2 ? hoverPos.cx - 250 : hoverPos.cx + 18,
              top: hoverPos.cy > hoverPos.cHeight / 2 ? hoverPos.cy - 360 : hoverPos.cy + 200,
              width: "240px",
              background: "rgba(5,8,18,0.92)",
              backdropFilter: "blur(12px)",
              border: "1px solid rgba(6,182,212,0.3)",
              borderRadius: "10px",
              padding: "10px 14px",
              pointerEvents: "none",
              zIndex: 11,
            }}>
              <div style={{ fontSize: "0.65rem", color: "var(--alpha-cyan)", fontWeight: 700, letterSpacing: "0.1em", marginBottom: "7px", fontFamily: "var(--font-display)", textTransform: "uppercase" }}>
                Live Math
              </div>
              <div className="formula-block" style={{ marginBottom: "8px", fontSize: "11px", display: "block" }}>
                {hoverPos.mathContext.formula}
              </div>
              {hoverPos.mathContext.rows.map((r, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", marginBottom: "4px" }}>
                  <span style={{ color: "var(--text-muted)" }}>{r.label}</span>
                  <span style={{ color: "#e0e8ff", fontFamily: "var(--font-mono)" }}>
                    {r.value}
                    {r.vote && <span style={{ color: r.vote === "Amber" ? "#f59e0b" : "#3b82f6", marginLeft: "4px" }}>({r.vote})</span>}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── EPOCH SCRUBBER ───────────────────────────── */}
        {historicalStates.current.length > 1 && (
          <div className="scrubber-wrap">
            <div className="scrubber-header">
              <span className="section-label" style={{ margin: 0, fontSize: "0.72rem" }}>Epoch Replay</span>
              <span style={{ fontSize: "0.72rem", color: "var(--alpha-cyan)", fontFamily: "var(--font-mono)" }}>
                {scrubEpoch !== null ? `Epoch ${scrubEpoch + 1}` : "Live"}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={historicalStates.current.length - 1}
              value={scrubEpoch ?? historicalStates.current.length - 1}
              onChange={(e) => {
                const idx = parseInt(e.target.value);
                setScrubEpoch(idx);
                const snap = historicalStates.current[idx];
                if (snap) {
                  if (snap.boundary) setBoundary(snap.boundary);
                  if (snap.points) setPoints(snap.points);
                  if (snap.labels) setLabels(snap.labels);
                  if (snap.predictions) setPredictions(snap.predictions);
                  if (snap.range) setRange(snap.range);
                  if (snap.metadata) setMetadata(snap.metadata);
                  if (snap.accuracy) setAccuracy(snap.accuracy);
                }
              }}
              style={{ width: "100%", accentColor: "var(--alpha-blue)" }}
            />
            {scrubEpoch !== null && (
              <button
                onClick={() => setScrubEpoch(null)}
                style={{ fontSize: "0.72rem", marginTop: "6px", background: "none", border: "1px solid rgba(255,255,255,0.1)", color: "#60a5fa", padding: "3px 10px", borderRadius: "6px", cursor: "pointer", fontFamily: "var(--font-display)" }}
              >
                Return to Live →
              </button>
            )}
            <div className="scrubber-explain">
              Drag to replay the model's learning history. Watch how the decision boundary shapes itself epoch by epoch as gradient descent updates the weights.
            </div>
          </div>
        )}

        {/* ── UNCERTAINTY LEGEND ───────────────────────── */}
        <div className="uncertainty-wrap">
          <div className="section-label">Decision Boundary Heatmap</div>
          <div className="uncertainty-bar" />
          <div className="uncertainty-labels">
            <span>P(Blue)=0%</span>
            <span>P=50% (boundary)</span>
            <span>P(Blue)=100%</span>
          </div>
          <div className="uncertainty-explain">
            Each pixel shows <strong>P(y=Blue | x)</strong> — the model's confidence at that location. The <strong>white band</strong> marks the decision boundary where P=0.5. As you move away from it, confidence grows. Wide uncertainty zones = the model is unsure; narrow zones = confident and sharp boundary.
            {meta.formula && (
              <span> For this model: <span className="formula-block" style={{ fontSize: "10px", verticalAlign: "middle" }}>{meta.formula}</span></span>
            )}
          </div>
        </div>

        {/* ── CONFUSION MATRIX ─────────────────────────── */}
        {confusionMatrix && (
          <div className="confusion-wrap">
            <div className="section-label">
              Confusion Matrix
              {activeFilter && <span style={{ fontSize: "0.7rem", color: "var(--warning)", marginLeft: "8px" }}>Filtering: {activeFilter} · click to clear</span>}
            </div>

            <div className="confusion-cells">
              <MatrixCell label="True Pos TP" value={confusionMatrix.TP} category="TP" color="#10b981" />
              <MatrixCell label="False Pos FP" value={confusionMatrix.FP} category="FP" color="#ef4444" />
              <MatrixCell label="True Neg TN" value={confusionMatrix.TN} category="TN" color="#10b981" />
              <MatrixCell label="False Neg FN" value={confusionMatrix.FN} category="FN" color="#ef4444" />
            </div>

            {/* Cell definition box */}
            {selectedCell && CONFUSION_DEFS[selectedCell] && (
              <div className="confusion-definition" style={{ borderLeftColor: CONFUSION_DEFS[selectedCell].color, marginBottom: "10px" }}>
                <strong style={{ color: CONFUSION_DEFS[selectedCell].color }}>{CONFUSION_DEFS[selectedCell].label}</strong>
                <div style={{ marginTop: "4px" }}>{CONFUSION_DEFS[selectedCell].def}</div>
                <div className="formula-block" style={{ marginTop: "6px", fontSize: "10px", display: "inline-block" }}>{CONFUSION_DEFS[selectedCell].formula}</div>
              </div>
            )}

            {/* Derived metrics */}
            <div className="derived-metrics">
              {[
                {
                  name: "Precision",
                  value: confusionMatrix.precision !== null ? `${(confusionMatrix.precision * 100).toFixed(0)}%` : "N/A",
                  formula: "TP/(TP+FP)",
                  tip: "Of all predicted positives, how many are actually positive?"
                },
                {
                  name: "Recall",
                  value: confusionMatrix.recall !== null ? `${(confusionMatrix.recall * 100).toFixed(0)}%` : "N/A",
                  formula: "TP/(TP+FN)",
                  tip: "Of all actual positives, how many did we catch?"
                },
                {
                  name: "F1 Score",
                  value: confusionMatrix.f1 !== null ? `${(confusionMatrix.f1 * 100).toFixed(0)}%` : "N/A",
                  formula: "2·P·R/(P+R)",
                  tip: "Harmonic mean of Precision and Recall. Best single metric for imbalanced data."
                },
              ].map((m, i) => (
                <div key={i} className="derived-metric-pill" title={m.tip}>
                  <div className="derived-metric-value">{m.value}</div>
                  <div className="derived-metric-name">{m.name}</div>
                  <div className="formula-block" style={{ fontSize: "9px", marginTop: "4px", display: "inline-block", padding: "2px 6px" }}>{m.formula}</div>
                </div>
              ))}
            </div>

            {/* What do they mean */}
            <div className="explain-box" style={{ marginTop: "10px", fontSize: "0.73rem" }}>
              <div className="explain-title">How to read the matrix</div>
              <strong style={{ color: "#6ee7b7" }}>TP & TN</strong> are correct predictions (green). <strong style={{ color: "#fca5a5" }}>FP</strong> (false alarm) and <strong style={{ color: "#fca5a5" }}>FN</strong> (miss) are errors. Click any cell to highlight those exact points on the canvas.
              A high FP rate = the model is trigger-happy. A high FN rate = the model is too conservative.
            </div>
          </div>
        )}

        {/* ── INSIGHTS ─────────────────────────────────── */}
        <div className="insights-engine">
          <div className="section-label">Live Analysis</div>
          {insights.length === 0 && !accuracy && (
            <div style={{ color: "var(--text-muted)", fontSize: "0.8rem", padding: "8px 0" }}>
              Press Run to start training. Insights and analysis appear here in real time.
            </div>
          )}
          {insights.map((ins, i) => (
            <div key={i} className={`insight-card ${ins.type === "danger" ? "danger" : ins.type === "warning" ? "warning" : ins.type === "success" ? "success" : ""}`}>
              {ins.text}
            </div>
          ))}

          {/* Loss chart */}
          {(model === "mlp" || model === "rf") && data.length > 0 && (
            <div className="loss-chart-wrap">
              <div className="section-label" style={{ marginBottom: "8px" }}>Training Loss</div>
              <ResponsiveContainer width="100%" height={100}>
                <LineChart data={data}>
                  <XAxis dataKey="epoch" tick={{ fill: "var(--text-muted)", fontSize: 9 }} stroke="none" />
                  <YAxis tick={{ fill: "var(--text-muted)", fontSize: 9 }} stroke="none" width={28} />
                  <Tooltip
                    contentStyle={{ background: "#0d1426", border: "1px solid rgba(59,130,246,0.3)", borderRadius: "8px", fontSize: "11px", color: "#fff" }}
                    labelFormatter={(v) => `Epoch ${v}`}
                  />
                  <Line type="monotone" dataKey="loss" stroke="var(--alpha-blue)" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <div className="loss-explain">
                <strong style={{ color: "#93c5fd" }}>Cross-entropy loss</strong> measures how wrong the model's probability estimates are. Each epoch, backpropagation nudges all weights to reduce this value. A smooth downward curve = healthy learning. A plateau = convergence. A spike = unstable learning rate.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}