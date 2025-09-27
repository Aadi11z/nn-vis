import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Modern, elegant 2-2-1 MLP visualization with animated forward & backward passes
// Tailwind CSS classes assumed. Install framer-motion: `npm i framer-motion`

// Small neural utilities
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const sigmoidDerivative = (y) => y * (1 - y); // y is sigmoid(x)

// Helpful numeric formatting
const fmt = (n, digits = 3) => Number(n).toFixed(digits);

export default function MLPAnimation() {
  // network structure: 2 inputs -> 2 hidden -> 1 output
  const [weightsIH, setWeightsIH] = useState([ // 2x2 matrix (hidden neurons x inputs)
    [Math.random() * 2 - 1, Math.random() * 2 - 1],
    [Math.random() * 2 - 1, Math.random() * 2 - 1],
  ]);
  const [biasH, setBiasH] = useState([Math.random() * 2 - 1, Math.random() * 2 - 1]);

  const [weightsHO, setWeightsHO] = useState([Math.random() * 2 - 1, Math.random() * 2 - 1]); // 1x2
  const [biasO, setBiasO] = useState(Math.random() * 2 - 1);

  // example training pair (XOR-like-ish): we'll animate a single training step repeatedly
  const [input, setInput] = useState([1, 0]);
  const [target, setTarget] = useState(1);

  // visual states
  const [hiddenActivations, setHiddenActivations] = useState([0, 0]);
  const [outputActivation, setOutputActivation] = useState(0);
  const [phase, setPhase] = useState("idle"); // idle | forward | backward | applied
  const [log, setLog] = useState([]);
  const [learningRate, setLearningRate] = useState(0.8);

  const runningRef = useRef(false);

  // reset network quick
  const randomize = () => {
    setWeightsIH([
      [Math.random() * 2 - 1, Math.random() * 2 - 1],
      [Math.random() * 2 - 1, Math.random() * 2 - 1],
    ]);
    setBiasH([Math.random() * 2 - 1, Math.random() * 2 - 1]);
    setWeightsHO([Math.random() * 2 - 1, Math.random() * 2 - 1]);
    setBiasO(Math.random() * 2 - 1);
    setLog([]);
  };

  // single animated training step (forward -> backward -> update)
  const trainStep = async () => {
    if (phase !== "idle") return;
    setPhase("forward");

    // ---------- FORWARD ----------
    // compute hidden layer
    const netH = [0, 0];
    for (let i = 0; i < 2; i++) {
      netH[i] = weightsIH[i][0] * input[0] + weightsIH[i][1] * input[1] + biasH[i];
    }

    const actH = netH.map(sigmoid);
    // animate hidden activations
    await animateValues(setHiddenActivations, actH, 350);

    // compute output
    const netO = weightsHO[0] * actH[0] + weightsHO[1] * actH[1] + biasO;
    const actO = sigmoid(netO);

    await animateValues(setOutputActivation, actO, 400);

    // log forward
    pushLog(`forward -> output ${fmt(actO)}`);

    // ---------- BACKWARD (calculate gradients) ----------
    setPhase("backward");

    // output error
    const errorO = target - actO;
    const deltaO = errorO * sigmoidDerivative(actO); // dE/dNetO

    // hidden deltas
    const deltaH = [0, 0];
    for (let i = 0; i < 2; i++) {
      deltaH[i] = sigmoidDerivative(actH[i]) * (weightsHO[i] * deltaO);
    }

    // gradient visualization: show proposed weight deltas without applying yet
    const proposed_dWeightsHO = weightsHO.map((w, i) => learningRate * deltaO * actH[i]);
    const proposed_dBiasO = learningRate * deltaO;

    const proposed_dWeightsIH = [
      [learningRate * deltaH[0] * input[0], learningRate * deltaH[0] * input[1]],
      [learningRate * deltaH[1] * input[0], learningRate * deltaH[1] * input[1]],
    ];
    const proposed_dBiasH = [learningRate * deltaH[0], learningRate * deltaH[1]];

    pushLog(`backward -> deltaO ${fmt(deltaO)}, deltaH [${fmt(deltaH[0])}, ${fmt(deltaH[1])}]`);

    // animate weight-change pulses for a moment
    await animateWeightPulse(proposed_dWeightsHO, proposed_dWeightsIH, proposed_dBiasO, proposed_dBiasH);

    setPhase("applied");

    // ---------- APPLY UPDATES ----------
    setWeightsHO((prev) => prev.map((w, i) => w + proposed_dWeightsHO[i]));
    setBiasO((b) => b + proposed_dBiasO);

    setWeightsIH((prev) => prev.map((row, i) => row.map((w, j) => w + proposed_dWeightsIH[i][j])));
    setBiasH((prev) => prev.map((b, i) => b + proposed_dBiasH[i]));

    pushLog(`applied updates`);

    // small pause then return to idle
    await sleep(600);
    setPhase("idle");
  };

  // helper: push log entry
  const pushLog = (s) => setLog((l) => [new Date().toLocaleTimeString() + " — " + s, ...l].slice(0, 6));

  // animate numeric setter through small increments
  function animateValues(setter, targetVal, ms) {
    return new Promise((res) => {
      const start = performance.now();
      const initial = Array.isArray(targetVal) ? (setter === setHiddenActivations ? hiddenActivations : targetVal.map(() => 0)) : (setter === setOutputActivation ? outputActivation : 0);
      const loop = (t) => {
        const k = Math.min(1, (t - start) / ms);
        if (Array.isArray(targetVal)) {
          const next = targetVal.map((tv, i) => initial[i] + (tv - initial[i]) * easeInOutCubic(k));
          setter(next);
        } else {
          setter(initial + (targetVal - initial) * easeInOutCubic(k));
        }
        if (k < 1) requestAnimationFrame(loop);
        else res();
      };
      requestAnimationFrame(loop);
    });
  }

  // show proposed weight changes by flashing edges — short animation
  function animateWeightPulse(dHO, dIH, dB_O, dB_H) {
    return new Promise((res) => {
      // we simulate by setting a temporary ref state so UI renders with pulses
      setPulse({ dHO, dIH, dB_O, dB_H, on: true });
      setTimeout(() => {
        setPulse((p) => ({ ...p, on: false }));
        setTimeout(res, 300);
      }, 700);
    });
  }

  const [pulse, setPulse] = useState({ dHO: [0, 0], dIH: [[0, 0], [0, 0]], dB_O: 0, dB_H: [0, 0], on: false });

  // small animation easing
  const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // auto-run toggle
  const [autoRun, setAutoRun] = useState(false);
  useEffect(() => {
    let id;
    if (autoRun) {
      runningRef.current = true;
      const loop = async () => {
        if (!runningRef.current) return;
        await trainStep();
        if (runningRef.current) id = setTimeout(loop, 600);
      };
      loop();
    } else {
      runningRef.current = false;
      clearTimeout(id);
    }
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRun]);

  // layout helper to render nodes and connections
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-black text-white p-6 font-sans">
      <div className="max-w-4xl mx-auto bg-opacity-10 backdrop-blur rounded-2xl border border-gray-800 p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">2-2-1 MLP Visualizer</h1>
            <p className="text-sm text-gray-400">Animated forward & backward propagation — elegant, minimal UI</p>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400">LR</label>
            <input
              className="w-20 bg-gray-800 px-2 py-1 rounded text-sm"
              type="number"
              step="0.1"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
            />

            <button
              className={`px-3 py-2 rounded-xl text-sm font-medium ${autoRun ? "bg-green-500/30" : "bg-gray-800"}`}
              onClick={() => setAutoRun((v) => !v)}
            >
              {autoRun ? "Stop" : "Auto"}
            </button>

            <button className="px-3 py-2 rounded-xl bg-indigo-600 text-sm" onClick={trainStep}>
              Step
            </button>

            <button className="px-3 py-2 rounded-xl bg-gray-700 text-sm" onClick={randomize}>
              Randomize
            </button>
          </div>
        </div>

        <div className="flex gap-8">
          {/* Network Canvas */}
          <div className="flex-1">
            <div className="relative h-96">
              {/* draw connections as absolute positioned lines using simple layout math */}
              <NetworkCanvas
                input={input}
                hidden={hiddenActivations}
                output={outputActivation}
                weightsIH={weightsIH}
                weightsHO={weightsHO}
                biasH={biasH}
                biasO={biasO}
                pulse={pulse}
                phase={phase}
              />
            </div>

            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="bg-gray-800/30 p-4 rounded-lg">
                <h3 className="text-sm text-gray-300 mb-2">Inputs</h3>
                <div className="flex gap-2">
                  {input.map((v, i) => (
                    <div key={i} className="p-3 rounded-md bg-gray-900">
                      <div className="text-xs text-gray-400">x{i}</div>
                      <div className="text-lg font-mono">{v}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-800/30 p-4 rounded-lg">
                <h3 className="text-sm text-gray-300 mb-2">Target</h3>
                <div className="p-3 rounded-md bg-gray-900 inline-block">
                  <div className="text-xs text-gray-400">y</div>
                  <div className="text-lg font-mono">{target}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="w-80">
            <div className="bg-gray-800/30 p-4 rounded-lg mb-4">
              <h3 className="text-sm text-gray-300 mb-2">Phases</h3>
              <div className="flex flex-col gap-2">
                <PhaseBadge label="idle" active={phase === "idle"} />
                <PhaseBadge label="forward" active={phase === "forward"} />
                <PhaseBadge label="backward" active={phase === "backward"} />
                <PhaseBadge label="applied" active={phase === "applied"} />
              </div>
            </div>

            <div className="bg-gray-800/30 p-4 rounded-lg mb-4">
              <h3 className="text-sm text-gray-300 mb-2">Weights & Biases</h3>
              <div className="text-xs text-gray-400 mb-2">Hidden layer (weights IH)</div>
              <div className="grid grid-cols-2 gap-2 mb-2">
                {weightsIH.map((row, i) => (
                  <div key={i} className="bg-gray-900 p-2 rounded">
                    <div className="text-xxs text-gray-400">h{i}</div>
                    <div className="font-mono text-sm">[{fmt(row[0])}, {fmt(row[1])}]</div>
                    <div className="text-xs text-gray-400">b: {fmt(biasH[i])}</div>
                  </div>
                ))}
              </div>

              <div className="text-xs text-gray-400 mb-1">Output weights (HO)</div>
              <div className="bg-gray-900 p-2 rounded mb-2 font-mono">[{fmt(weightsHO[0])}, {fmt(weightsHO[1])}]</div>
              <div className="text-xs text-gray-400">bO: {fmt(biasO)}</div>
            </div>

            <div className="bg-gray-800/30 p-4 rounded-lg mb-4">
              <h3 className="text-sm text-gray-300 mb-2">Logs</h3>
              <div className="text-xs text-gray-400 space-y-1">
                {log.map((l, i) => (
                  <div key={i} className="text-sm text-gray-200">{l}</div>
                ))}
              </div>
            </div>

            <div className="bg-gray-800/30 p-4 rounded-lg">
              <h3 className="text-sm text-gray-300 mb-2">Controls</h3>
              <div className="flex flex-col gap-2">
                <label className="text-xs text-gray-400">Input</label>
                <div className="flex gap-2">
                  <input
                    className="bg-gray-900 p-2 rounded w-20"
                    value={input[0]}
                    onChange={(e) => setInput((p) => [Number(e.target.value), p[1]])}
                  />
                  <input
                    className="bg-gray-900 p-2 rounded w-20"
                    value={input[1]}
                    onChange={(e) => setInput((p) => [p[0], Number(e.target.value)])}
                  />
                </div>

                <label className="text-xs text-gray-400">Target</label>
                <input className="bg-gray-900 p-2 rounded w-40" value={target} onChange={(e) => setTarget(Number(e.target.value))} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Small visual components
function PhaseBadge({ label, active }) {
  return (
    <div className={`px-3 py-1 rounded-full text-sm ${active ? "bg-indigo-600 text-white" : "bg-gray-800 text-gray-400"}`}>{label}</div>
  );
}

// NetworkCanvas: draws nodes and connections; uses simple coordinates
function NetworkCanvas({ input, hidden, output, weightsIH, weightsHO, biasH, biasO, pulse, phase }) {
  // layout coordinates
  const W = 680;
  const H = 380;
  const leftX = 80;
  const midX = W / 2 - 20;
  const rightX = W - 80;

  const inputYs = [H * 0.3, H * 0.7];
  const hiddenYs = [H * 0.35, H * 0.65];
  const outputY = H / 2;

  // intensity helpers
  const colorFor = (val) => {
    // val range 0..1 -> map to bluish for low, warm for high
    const g = Math.round(200 * val + 30);
    const r = Math.round(50 + 170 * val);
    const b = Math.round(200 - 150 * val);
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full rounded-lg">
      {/* connections IH */}
      {weightsIH.map((row, i) => row.map((w, j) => {
        const x1 = leftX;
        const y1 = inputYs[j];
        const x2 = midX;
        const y2 = hiddenYs[i];
        const weightSign = Math.sign(w);
        const strokeW = Math.min(6, Math.abs(w) * 4 + 0.5);
        const strokeColor = weightSign >= 0 ? "rgba(99,102,241,0.9)" : "rgba(220,38,38,0.9)";
        const pulseActive = pulse.on && Math.abs(pulse.dIH[i][j]) > 1e-6;
        return (
          <g key={`ih-${i}-${j}`}>
            <line
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={strokeColor}
              strokeWidth={strokeW}
              strokeLinecap="round"
              opacity={pulseActive ? 1 : 0.7}
            />
            {/* weight label */}
            <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} fontSize={10} fill="#cbd5e1" textAnchor="middle">{fmt(w)}</text>
            {pulseActive && (
              <pulseRing cx={(x1 + x2) / 2} cy={(y1 + y2) / 2} val={pulse.dIH[i][j]} />
            )}
          </g>
        );
      }))}

      {/* connections HO */}
      {weightsHO.map((w, i) => {
        const x1 = midX + 20;
        const y1 = hiddenYs[i];
        const x2 = rightX - 10;
        const y2 = outputY;
        const strokeW = Math.min(6, Math.abs(w) * 4 + 0.5);
        const strokeColor = w >= 0 ? "rgba(99,102,241,0.95)" : "rgba(220,38,38,0.95)";
        const pulseActive = pulse.on && Math.abs(pulse.dHO[i]) > 1e-6;
        return (
          <g key={`ho-${i}`}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={strokeColor} strokeWidth={strokeW} strokeLinecap="round" opacity={pulseActive ? 1 : 0.85} />
            <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 8} fontSize={10} fill="#cbd5e1" textAnchor="middle">{fmt(w)}</text>
            {pulseActive && <pulseRing cx={(x1 + x2) / 2} cy={(y1 + y2) / 2} val={pulse.dHO[i]} />}
          </g>
        );
      })}

      {/* input nodes */}
      {input.map((v, i) => (
        <g key={`in-${i}`}>
          <circle cx={leftX} cy={inputYs[i]} r={26} fill={colorFor(v)} stroke="#111827" strokeWidth={2} />
          <text x={leftX} y={inputYs[i] + 6} fontSize={14} textAnchor="middle" fill="#0f172a" className="font-mono">{fmt(v)}</text>
          <text x={leftX} y={inputYs[i] + 32} fontSize={10} textAnchor="middle" fill="#94a3b8">x{i}</text>
        </g>
      ))}

      {/* hidden nodes */}
      {hidden.map((v, i) => (
        <g key={`hid-${i}`}>
          <motion.circle
            cx={midX}
            cy={hiddenYs[i]}
            r={28}
            fill={colorFor(v)}
            stroke="#0b1220"
            strokeWidth={2}
            animate={{ r: phase === "forward" ? 34 : 28 }}
            transition={{ duration: 0.4 }}
          />

          <text x={midX} y={hiddenYs[i] + 6} fontSize={14} textAnchor="middle" fill="#071024" className="font-mono">{fmt(v)}</text>
          <text x={midX} y={hiddenYs[i] + 34} fontSize={10} textAnchor="middle" fill="#94a3b8">h{i} b:{fmt(biasH[i])}</text>
        </g>
      ))}

      {/* output node */}
      <g>
        <motion.circle
          cx={rightX}
          cy={outputY}
          r={36}
          fill={colorFor(output)}
          stroke="#020617"
          strokeWidth={3}
          animate={{ r: phase === "forward" ? 44 : 36 }}
          transition={{ duration: 0.45 }}
        />
        <text x={rightX} y={outputY + 8} fontSize={16} textAnchor="middle" fill="#041226" className="font-mono">{fmt(output)}</text>
        <text x={rightX} y={outputY + 40} fontSize={10} textAnchor="middle" fill="#94a3b8">o b:{fmt(biasO)}</text>
      </g>

      {/* subtle glow */}
      <defs>
        <filter id="f1" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="coloredBlur" />
          <feBlend in="SourceGraphic" in2="coloredBlur" mode="screen" />
        </filter>
      </defs>
    </svg>
  );
}

// tiny component used inside svg to emulate a pulse ring (not a React component but a function returning element)
function pulseRing({ cx, cy, val }) {
  const sign = val >= 0 ? 1 : -1;
  const color = sign > 0 ? "rgba(34,197,94,0.95)" : "rgba(245,158,11,0.95)";
  const absv = Math.min(0.9, Math.abs(val));
  return (
    <g>
      <circle cx={cx} cy={cy} r={12 + absv * 18} fill="none" stroke={color} strokeWidth={2} opacity={0.9} />
      <circle cx={cx} cy={cy} r={6 + absv * 12} fill={color} opacity={0.08} />
    </g>
  );
}
