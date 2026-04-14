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
  const [scrubEpoch, setScrubEpoch] = useState(null); // null = live mode
  const [activeFilter, setActiveFilter] = useState(null); // "TP" | "FP" | "TN" | "FN"

  const [customDatasets, setCustomDatasets] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [saveStatus, setSaveStatus] = useState("");

  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const accuracyHistory = useRef([]);

  // --- Performance-critical refs (no re-render) ---
  const historicalStates = useRef([]); // stores {boundary, accuracy, metadata, points, labels, predictions, range}
  const offscreenCanvasRef = useRef(null);
  const visualState = useRef({ points: [], labels: [], predictions: [], range: null, metadata: null, hoverPos: null, model: "mlp", activeFilter: null });

  // Drag state — all in refs to avoid re-render jank
  const dragTargetIndex = useRef(null);
  const isDragging = useRef(false);

  // Debounce timer ref for WebSocket drag updates
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
      setSaveStatus("Saved to Dashboard!");
      setTimeout(() => setSaveStatus(""), 2500);
    } catch {
      setSaveStatus("Error saving.");
      setTimeout(() => setSaveStatus(""), 2000);
    }
  };

  // Ingest a WebSocket frame into application state
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
          setStatus("Stable");
        } else {
          setStatus("Learning...");
        }
      } else {
        setStatus("Converged Immediately");
      }
    }
    if (newData.points) { setPoints(newData.points); setLabels(newData.labels); }
    if (newData.predictions) setPredictions(newData.predictions);
    if (newData.range) setRange(newData.range);
    if (newData.metadata) setMetadata(newData.metadata);

    // Store in historical snapshot ref (no re-render!)
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

  // ---- Offscreen Heatmap Cache ----
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
          r = Math.round(255 + p * (108 - 255));
          g = Math.round(183 + p * (99 - 183));
          b = Math.round(3 + p * (255 - 3));
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

  // Keep visualState ref synced
  useEffect(() => {
    visualState.current = { points, labels, predictions, range, metadata, hoverPos, model, activeFilter };
  }, [points, labels, predictions, range, metadata, hoverPos, model, activeFilter]);

  // ---- Main 60fps Render Loop ----
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

          // Confusion matrix category
          let category = null;
          if (preds.length > 0) {
            if (lbls[i] === 1 && preds[i] === 1) category = "TP";
            else if (lbls[i] === 0 && preds[i] === 1) category = "FP";
            else if (lbls[i] === 0 && preds[i] === 0) category = "TN";
            else if (lbls[i] === 1 && preds[i] === 0) category = "FN";
          }

          // Dim non-filtered points when a filter is active
          const dimmed = af && category !== af;

          ctx.globalAlpha = dimmed ? 0.15 : 1.0;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fillStyle = lbls[i] === 0 ? "#FFB703" : "#6C63FF";
          ctx.fill();

          if (!dimmed && af && category === af) {
            // Filtered points get intense strobe halos
            const pulse = (Math.sin(Date.now() / 100) + 1) / 2;
            ctx.strokeStyle = af === "FP" || af === "FN" ? `rgba(255, 71, 87, ${0.5 + pulse * 0.5})` : `rgba(80, 255, 160, ${0.5 + pulse * 0.5})`;
            ctx.lineWidth = 3;
            ctx.shadowColor = ctx.strokeStyle;
            ctx.shadowBlur = 8 + pulse * 8;
          } else if (!dimmed && isMisclassified) {
            const pulse = (Math.sin(Date.now() / 150) + 1) / 2;
            ctx.strokeStyle = `rgba(255, 71, 87, ${0.4 + pulse * 0.6})`;
            ctx.lineWidth = 2.5;
            ctx.shadowColor = "#ff4757";
            ctx.shadowBlur = 5 + pulse * 5;
          } else if (!dimmed) {
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1.5;
            ctx.shadowBlur = 0;
          }
          ctx.stroke();
          ctx.shadowBlur = 0;
          ctx.globalAlpha = 1.0;
        });

        // Visual Reasoning Traces
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
            ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.arc(sv.sx, sv.sy, 6, 0, 2 * Math.PI);
            ctx.strokeStyle = "#fff";
            ctx.shadowColor = "#fff";
            ctx.shadowBlur = 8;
            ctx.stroke();
            ctx.shadowBlur = 0;
          });
        }

        if (m === "knn" && hp && pts) {
          const distances = pts.map((p, idx) => ({
            idx,
            d: Math.pow(p[0] - hp.realX, 2) + Math.pow(p[1] - hp.realY, 2)
          })).sort((a, b) => a.d - b.d).slice(0, 3);
          const maxD = distances[distances.length - 1]?.d || 1;
          distances.forEach(neighbor => {
            const p = pts[neighbor.idx];
            const nx = ((p[0] - xMin) / (xMax - xMin)) * canvas.width;
            const ny = canvas.height - ((p[1] - yMin) / (yMax - yMin)) * canvas.height;
            const opacity = Math.max(0.2, 1 - (neighbor.d / (maxD * 1.5)));
            ctx.beginPath();
            ctx.moveTo(hp.cx, hp.cy);
            ctx.lineTo(nx, ny);
            ctx.strokeStyle = lbls[neighbor.idx] === 0 ? `rgba(255, 183, 3, ${opacity})` : `rgba(108, 99, 255, ${opacity})`;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([2, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
          });
        }

        if (m === "rf" && hp) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
          ctx.fillRect(hp.cx - 25, hp.cy - 25, 50, 50);
          ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
          ctx.lineWidth = 2.5;
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
          ctx.strokeStyle = "rgba(255, 255, 255, 1)";
          ctx.lineWidth = 3;
          ctx.shadowColor = "#fff";
          ctx.shadowBlur = 10;
          ctx.stroke();
          ctx.shadowBlur = 0;
        }

        if (m === "mlp" && hp) {
          ctx.beginPath();
          ctx.arc(hp.cx, hp.cy, 30, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
          ctx.fill();
          ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
          ctx.lineWidth = 4;
          ctx.stroke();
        }

        // Draw dragged point highlight
        if (isDragging.current && dragTargetIndex.current !== null) {
          const dp = pts[dragTargetIndex.current];
          if (dp) {
            const dx = ((dp[0] - xMin) / (xMax - xMin)) * canvas.width;
            const dy = canvas.height - ((dp[1] - yMin) / (yMax - yMin)) * canvas.height;
            ctx.beginPath();
            ctx.arc(dx, dy, 10, 0, 2 * Math.PI);
            ctx.strokeStyle = "#ffffff";
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

  // ---- Mouse interaction handlers ----
  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    return { x, y, rect };
  };

  const canvasToData = (x, y, rect) => {
    if (!range) return null;
    const realX = range.xMin + (x / rect.width) * (range.xMax - range.xMin);
    const realY = range.yMin + ((rect.height - y) / rect.height) * (range.yMax - range.yMin);
    return { realX, realY };
  };

  const findNearestPoint = (cx, cy, rect) => {
    if (!points || !range) return null;
    const { xMin, xMax, yMin, yMax } = range;
    let best = null, bestDist = 12; // 12px snap radius
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
    if (idx !== null) {
      dragTargetIndex.current = idx;
      isDragging.current = true;
    }
  };

  const handleMouseUp = () => {
    isDragging.current = false;
    dragTargetIndex.current = null;
    if (dragDebounceRef.current) clearTimeout(dragDebounceRef.current);
  };

  const handleMouseMove = (e) => {
    const { x, y, rect } = getCanvasCoords(e);

    // --- Dragging logic ---
    if (isDragging.current && dragTargetIndex.current !== null) {
      const coords = canvasToData(x, y, rect);
      if (!coords) return;

      // Optimistic local update — instant 60fps
      setPoints(prev => {
        const updated = [...prev];
        updated[dragTargetIndex.current] = [coords.realX, coords.realY];
        return updated;
      });

      // Debounced WebSocket send — max 1 payload per 80ms
      if (dragDebounceRef.current) clearTimeout(dragDebounceRef.current);
      dragDebounceRef.current = setTimeout(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            action: "update_point",
            index: dragTargetIndex.current,
            new_coords: [coords.realX, coords.realY]
          }));
        }
      }, 80);
      return; // don't update hover tooltip while dragging
    }

    // --- Hover probe logic (only when not dragging) ---
    if (!range || !boundary) { setHoverPos(null); return; }
    const realX = range.xMin + (x / rect.width) * (range.xMax - range.xMin);
    const realY = range.yMin + ((rect.height - y) / rect.height) * (range.yMax - range.yMin);
    const cols = boundary[0].length, rows = boundary.length;
    const gridX = Math.min(cols - 1, Math.max(0, Math.floor((x / rect.width) * cols)));
    const gridY = Math.min(rows - 1, Math.max(0, rows - 1 - Math.floor((y / rect.height) * rows)));
    const p = boundary[gridY][gridX];
    const prediction = p >= 0.5 ? 1 : 0;
    const confidence = Math.round(Math.abs(p - 0.5) * 200);

    // Only show hover if mouse isn't over a filter-active canvas
    if (activeFilter) {
      setHoverPos(null);
      return;
    }

    let modelInsights = "";
    if (model === "svm") modelInsights = "These nearby support vectors define the margin separating the classes.";
    else if (model === "knn") modelInsights = "These nearby points determine the classification at this location.";
    else if (model === "logreg") modelInsights = "This model uses a single straight boundary to separate the two classes.";
    else if (model === "mlp") modelInsights = "Boundary bends here due to complex non-linear combinations of distant points.";
    else if (model === "rf") modelInsights = "This region is created by multiple decision splits, and all points inside are classified the same way.";

    let uncertaintyInsight = "";
    if (confidence < 30) uncertaintyInsight = "This point lies near the boundary, so the model is uncertain.";
    else if (confidence > 70) uncertaintyInsight = "This point lies deep inside a region, so the model is confident.";
    else uncertaintyInsight = "This point is in a transitional zone where confidence is moderate.";

    // Math context for knowledge panel
    let mathContext = null;
    if (model === "knn" && points && range) {
      const top3 = points.map((pt, i) => ({
        i, d: Math.sqrt(Math.pow(pt[0] - realX, 2) + Math.pow(pt[1] - realY, 2))
      })).sort((a, b) => a.d - b.d).slice(0, 3);
      mathContext = {
        formula: "d = √((x₂−x₁)² + (y₂−y₁)²)",
        rows: top3.map(n => ({
          label: `Neighbor ${n.i}`,
          value: `d = ${n.d.toFixed(3)}`,
          vote: labels[n.i] === 0 ? "Yellow" : "Purple"
        }))
      };
    } else if (model === "logreg" && metadata?.weights) {
      const [w0, w1] = metadata.weights;
      const b = metadata.bias;
      const logit = w0 * realX + w1 * realY + b;
      const prob = (1 / (1 + Math.exp(-logit))).toFixed(3);
      mathContext = {
        formula: `z = w₀·x + w₁·y + b`,
        rows: [
          { label: "w₀, w₁, b", value: `${w0.toFixed(2)}, ${w1.toFixed(2)}, ${b.toFixed(2)}` },
          { label: "z (logit)", value: logit.toFixed(3) },
          { label: "P(class=1)", value: `σ(z) = ${prob}` }
        ]
      };
    } else if (model === "svm" && metadata?.support_vectors) {
      mathContext = {
        formula: "Margin = 2 / ||w||",
        rows: [{ label: "Anchors", value: `${metadata.support_vectors.length} Support Vectors` }]
      };
    }

    setHoverPos({ cx: x, cy: y, cWidth: rect.width, cHeight: rect.height, realX, realY, prediction, confidence, modelInsights, uncertaintyInsight, mathContext });
  };

  // Confusion matrix compute
  const confusionMatrix = (() => {
    if (!predictions.length || !labels.length) return null;
    let TP = 0, FP = 0, TN = 0, FN = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (labels[i] === 1 && predictions[i] === 1) TP++;
      else if (labels[i] === 0 && predictions[i] === 1) FP++;
      else if (labels[i] === 0 && predictions[i] === 0) TN++;
      else if (labels[i] === 1 && predictions[i] === 0) FN++;
    }
    return { TP, FP, TN, FN };
  })();

  const generateInsights = () => {
    const insights = [];
    if (!accuracy || !model) return insights;
    if (accuracy === 1.0) {
      insights.push("🟢 Model captured all points perfectly.");
      if (model === "mlp" || model === "rf") insights.push("⚠️ Look at the boundary wrapping points tightly. It is dangerously overfitting.");
    } else if (accuracy < 0.6) {
      insights.push("🔴 Model is effectively guessing blindly.");
      if (model === "logreg" && ["circles", "moons"].includes(dataset))
        insights.push("💡 A straight line cannot cut a circle. This linear model is structurally incompatible.");
    }
    if (predictions && labels && predictions.length > 0) {
      let errs = 0;
      for (let i = 0; i < predictions.length; i++) if (predictions[i] !== labels[i]) errs++;
      if (errs > 0) insights.push(`🔴 ${errs} points are glowing red because they are misclassified.`);
    }
    if (model === "svm" && metadata?.support_vectors)
      insights.push(`📐 Exactly ${metadata.support_vectors.length} Support Vectors are physically outlining the visible margin.`);
    if (model === "knn") insights.push(`🔍 Hover anywhere to see the 3 specific voters controlling that local zone.`);
    if (model === "logreg" && metadata?.weights) insights.push(`⚖️ LogReg is projecting rigid confidence purely based on distance from the line.`);
    return insights;
  };

  const insights = generateInsights();
  const MatrixCell = ({ label, value, category, color }) => (
    <div
      onClick={() => setActiveFilter(prev => prev === category ? null : category)}
      style={{
        flex: 1, textAlign: 'center', padding: '8px 4px',
        background: activeFilter === category ? `${color}33` : 'rgba(255,255,255,0.03)',
        border: `1px solid ${activeFilter === category ? color : 'rgba(255,255,255,0.08)'}`,
        borderRadius: '8px', cursor: 'pointer', transition: 'all 0.2s'
      }}
    >
      <div style={{ fontSize: '10px', color: '#8b8b9e', marginBottom: '2px' }}>{label}</div>
      <div style={{ fontSize: '18px', fontWeight: '700', color }}>{value ?? '--'}</div>
    </div>
  );

  return (
    <div className="glass-panel training-panel">
      <div className="panel-header">
        <h2 className="panel-title">Model {title}</h2>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn-train" onClick={startTraining}>Start Run</button>
          <button className="btn-train" style={{ background: 'rgba(255,255,255,0.1)', boxShadow: 'none' }} onClick={saveExperiment} disabled={!accuracy}>
            {saveStatus || "Save Run"}
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="controls-row">
        <select className="styled-select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
          <optgroup label="Default Datasets">
            <option value="moons">Moons Dataset</option>
            <option value="circles">Circles Dataset</option>
            <option value="blobs">Blobs Dataset</option>
          </optgroup>
          {customDatasets.length > 0 && (
            <optgroup label="Your Datasets">
              {customDatasets.map(d => <option key={d.id} value={d.name}>{d.name}</option>)}
            </optgroup>
          )}
        </select>
        <label className="btn-train" style={{ background: 'rgba(255,255,255,0.1)', cursor: 'pointer', textAlign: 'center', boxShadow: 'none' }}>
          {uploadStatus || "Upload CSV"}
          <input type="file" accept=".csv" style={{ display: 'none' }} onChange={handleFileUpload} />
        </label>
        <select className="styled-select" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="mlp">Neural Net (MLP)</option>
          <option value="svm">Support Vector Machine</option>
          <option value="rf">Random Forest</option>
          <option value="logreg">Logistic Regression</option>
          <option value="knn">K-Nearest Neighbors</option>
        </select>
      </div>

      {/* Status + Progress */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', gap: '10px' }}>
          <div className="metric-bar" style={{ flex: 1 }}>
            <span style={{ fontSize: '12px' }}>STATUS</span>
            <span className="metric-value">{accuracy ? status : "Idle"}</span>
          </div>
          <div className="metric-bar" style={{ flex: 1 }}>
            <span style={{ fontSize: '12px' }}>VALIDATION ACCURACY</span>
            <span className="metric-value">{accuracy ? (accuracy * 100).toFixed(1) + "%" : "--"}</span>
          </div>
        </div>
        {currentEpoch > 0 && (
          <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${(currentEpoch / totalEpochs) * 100}%`, background: '#6C63FF', transition: 'width 0.15s linear' }} />
          </div>
        )}
      </div>

      <div className="viz-container">
        {/* Canvas */}
        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef}
            width={320}
            height={320}
            className="ml-canvas"
            style={{ cursor: isDragging.current ? 'grabbing' : 'crosshair' }}
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseOut={() => { setHoverPos(null); handleMouseUp(); }}
          />

          {/* Probe Tooltip */}
          {hoverPos && !activeFilter && (
            <div className="probe-tooltip" style={{
              left: hoverPos.cx > hoverPos.cWidth / 2 ? hoverPos.cx - 240 : hoverPos.cx + 25,
              top: hoverPos.cy > hoverPos.cHeight / 2 ? hoverPos.cy - 180 : hoverPos.cy + 25,
              maxWidth: '220px', background: 'rgba(20, 20, 30, 0.75)',
              backdropFilter: 'blur(8px)', border: '1px solid rgba(255, 255, 255, 0.1)',
              pointerEvents: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.5)', zIndex: 10
            }}>
              <div style={{ fontWeight: '600', color: '#fff', marginBottom: '8px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '4px' }}>Probe Region</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                <span style={{ color: '#8b8b9e' }}>X, Y:</span>
                <span>({hoverPos.realX.toFixed(2)}, {hoverPos.realY.toFixed(2)})</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                <span style={{ color: '#8b8b9e' }}>Confidence:</span>
                <span style={{ color: hoverPos.confidence < 30 ? '#FFB703' : '#6c63ff' }}>{hoverPos.confidence}%</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ color: '#8b8b9e' }}>Prediction:</span>
                <span style={{ color: hoverPos.prediction === 0 ? '#FFB703' : '#6c63ff', fontWeight: 'bold' }}>
                  {hoverPos.prediction === 0 ? 'Yellow' : 'Purple'}
                </span>
              </div>
              <div style={{ color: '#c5c9ff', paddingBottom: '4px', fontSize: '11px', lineHeight: '1.4' }}>{hoverPos.uncertaintyInsight}</div>
              <div style={{ color: '#aab2ff', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '6px', fontSize: '11px', lineHeight: '1.4' }}>{hoverPos.modelInsights}</div>
            </div>
          )}

          {/* Math Context Panel */}
          {hoverPos?.mathContext && !activeFilter && (
            <div style={{
              position: 'absolute',
              left: hoverPos.cx > hoverPos.cWidth / 2 ? hoverPos.cx - 240 : hoverPos.cx + 25,
              top: hoverPos.cy > hoverPos.cHeight / 2 ? hoverPos.cy - 340 : hoverPos.cy + 200,
              width: '220px', background: 'rgba(10, 10, 20, 0.85)',
              backdropFilter: 'blur(10px)', border: '1px solid rgba(108,99,255,0.3)',
              borderRadius: '10px', padding: '10px 12px', pointerEvents: 'none', zIndex: 11
            }}>
              <div style={{ fontSize: '10px', color: '#6c63ff', fontWeight: '700', letterSpacing: '1px', marginBottom: '6px' }}>MATH CONTEXT</div>
              <div style={{ fontFamily: 'monospace', fontSize: '12px', color: '#fff', background: 'rgba(108,99,255,0.15)', padding: '6px 8px', borderRadius: '6px', marginBottom: '8px' }}>
                {hoverPos.mathContext.formula}
              </div>
              {hoverPos.mathContext.rows.map((r, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '3px' }}>
                  <span style={{ color: '#8b8b9e' }}>{r.label}</span>
                  <span style={{ color: '#e0e0ff', fontFamily: 'monospace' }}>
                    {r.value}
                    {r.vote && <span style={{ color: r.vote === 'Yellow' ? '#FFB703' : '#6C63FF', marginLeft: '4px' }}>({r.vote})</span>}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Epoch Scrubber */}
        {historicalStates.current.length > 1 && (
          <div style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#8b8b9e', marginBottom: '4px' }}>
              <span>Epoch Replay</span>
              <span>{scrubEpoch !== null ? `Epoch ${scrubEpoch + 1}` : "Live"}</span>
            </div>
            <input
              type="range" min={0} max={historicalStates.current.length - 1}
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
              style={{ width: '100%', accentColor: '#6C63FF' }}
            />
            {scrubEpoch !== null && (
              <button onClick={() => setScrubEpoch(null)} style={{ fontSize: '11px', marginTop: '4px', background: 'none', border: '1px solid rgba(255,255,255,0.1)', color: '#aab2ff', padding: '3px 10px', borderRadius: '6px', cursor: 'pointer' }}>
                Return to Live
              </button>
            )}
          </div>
        )}

        {/* Uncertainty Legend */}
        <div style={{ width: '100%', height: '8px', background: 'linear-gradient(to right, #FFB703, #8a88a0, #6C63FF)', borderRadius: '4px' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#8b8b9e', marginTop: '4px' }}>
          <span>100% Yellow</span><span>High Uncertainty Zone</span><span>100% Purple</span>
        </div>

        {/* Confusion Matrix */}
        {confusionMatrix && (
          <div style={{ width: '100%' }}>
            <div className="insights-header" style={{ marginBottom: '8px' }}>
              Confusion Matrix
              {activeFilter && <span style={{ fontSize: '10px', color: '#FFB703', marginLeft: '8px' }}>Filtering: {activeFilter} — click to clear</span>}
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <MatrixCell label="True Pos" value={confusionMatrix.TP} category="TP" color="#50ffa0" />
              <MatrixCell label="False Pos" value={confusionMatrix.FP} category="FP" color="#ff4757" />
              <MatrixCell label="True Neg" value={confusionMatrix.TN} category="TN" color="#50ffa0" />
              <MatrixCell label="False Neg" value={confusionMatrix.FN} category="FN" color="#ff4757" />
            </div>
          </div>
        )}

        {/* Insights */}
        <div className="insights-engine">
          <div className="insights-header">Live Analytics</div>
          {insights.map((msg, i) => {
            let type = "";
            if (msg.includes("⚠️") || msg.includes("🔴")) type = "danger";
            if (msg.includes("💡") || msg.includes("📐")) type = "warning";
            return <div key={i} className={`insight-card ${type}`}>{msg}</div>;
          })}

          {(model === "mlp" || model === "rf") && data.length > 0 && (
            <div style={{ marginTop: "10px", background: "rgba(0,0,0,0.2)", borderRadius: "10px", padding: "10px", border: "1px solid rgba(255,255,255,0.05)", width: '100%', height: '140px' }}>
              <div className="insights-header" style={{ marginBottom: "5px" }}>Loss Progression</div>
              <ResponsiveContainer width="100%" height="80%">
                <LineChart data={data}>
                  <XAxis dataKey="epoch" tick={{ fill: '#8b8b9e', fontSize: 10 }} stroke="none" />
                  <YAxis tick={{ fill: '#8b8b9e', fontSize: 10 }} stroke="none" width={30} />
                  <Tooltip contentStyle={{ background: '#1c1c24', border: '1px solid #6c63ff', borderRadius: '8px', fontSize: '12px', color: '#fff' }} />
                  <Line type="monotone" dataKey="loss" stroke="#6C63FF" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}