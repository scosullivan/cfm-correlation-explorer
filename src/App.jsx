import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell, ComposedChart, Line
} from "recharts";

// ============================================================
// MATH (same engine, hidden from user)
// ============================================================
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng) {
  let u, v, s;
  do { u = 2 * rng() - 1; v = 2 * rng() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
  return u * Math.sqrt((-2 * Math.log(s)) / s);
}

function generateData(T, N, rng) {
  const X = [];
  for (let i = 0; i < T; i++) { X[i] = []; for (let j = 0; j < N; j++) X[i][j] = gaussianRandom(rng); }
  const numF = 3;
  const factors = [];
  for (let i = 0; i < T; i++) { factors[i] = []; for (let f = 0; f < numF; f++) factors[i][f] = gaussianRandom(rng); }
  const loadings = Array.from({ length: N }, () => Array.from({ length: numF }, () => gaussianRandom(rng) * 0.6));
  for (let t = 0; t < T; t++) for (let j = 0; j < N; j++) for (let f = 0; f < numF; f++) X[t][j] += loadings[j][f] * factors[t][f];
  return X;
}

function corrMatrix(X) {
  const T = X.length, N = X[0].length;
  const m = Array(N).fill(0), s = Array(N).fill(0);
  for (let j = 0; j < N; j++) { for (let i = 0; i < T; i++) m[j] += X[i][j]; m[j] /= T; for (let i = 0; i < T; i++) s[j] += (X[i][j] - m[j]) ** 2; s[j] = Math.sqrt(s[j] / T); }
  const C = Array.from({ length: N }, () => Array(N).fill(0));
  for (let i = 0; i < N; i++) for (let j = i; j < N; j++) { let sum = 0; for (let t = 0; t < T; t++) sum += ((X[t][i] - m[i]) / s[i]) * ((X[t][j] - m[j]) / s[j]); C[i][j] = sum / T; C[j][i] = C[i][j]; }
  return C;
}

function getEigenvectors(matrix, numIter = 80) {
  const N = matrix.length; const vectors = []; let A = matrix.map(r => [...r]);
  for (let k = 0; k < N; k++) {
    let v = Array(A.length).fill(0).map(() => Math.random() - 0.5);
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)); v = v.map(x => x / norm);
    let ev = 0;
    for (let iter = 0; iter < numIter; iter++) {
      const Av = Array(A.length).fill(0);
      for (let i = 0; i < A.length; i++) for (let j = 0; j < A.length; j++) Av[i] += A[i][j] * v[j];
      ev = v.reduce((s, x, i) => s + x * Av[i], 0);
      norm = Math.sqrt(Av.reduce((s, x) => s + x * x, 0)); if (norm < 1e-10) break;
      v = Av.map(x => x / norm);
    }
    vectors.push({ eigenvalue: ev, vector: [...v] });
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A.length; j++) A[i][j] -= ev * v[i] * v[j];
  }
  return vectors.sort((a, b) => a.eigenvalue - b.eigenvalue);
}

function getEigenvalues(matrix) { return getEigenvectors(matrix).map(v => v.eigenvalue).sort((a, b) => a - b); }

function marchenkoPastur(x, q) {
  const lp = (1 + Math.sqrt(q)) ** 2, lm = (1 - Math.sqrt(q)) ** 2;
  if (x < lm || x > lp) return 0;
  return (1 / (2 * Math.PI * q * x)) * Math.sqrt((lp - x) * (x - lm));
}

function cleanMatrix(matrix, q) {
  const N = matrix.length; const evData = getEigenvectors(matrix);
  const lp = (1 + Math.sqrt(q)) ** 2;
  const noiseE = evData.filter(d => d.eigenvalue <= lp), signalE = evData.filter(d => d.eigenvalue > lp);
  const avgN = noiseE.length > 0 ? noiseE.reduce((s, d) => s + d.eigenvalue, 0) / noiseE.length : 1;
  const cleaned = Array.from({ length: N }, () => Array(N).fill(0));
  for (const ev of evData) { const l = ev.eigenvalue > lp ? ev.eigenvalue : avgN; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) cleaned[i][j] += l * ev.vector[i] * ev.vector[j]; }
  const diag = cleaned.map((r, i) => Math.sqrt(r[i]));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) cleaned[i][j] /= diag[i] * diag[j];
  return { cleaned, numSignal: signalE.length, numNoise: noiseE.length };
}

function minVarWeights(C) {
  const N = C.length; let w = Array(N).fill(1 / N);
  for (let iter = 0; iter < 50; iter++) {
    const Cw = Array(N).fill(0); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) Cw[i] += C[i][j] * w[j];
    const wCw = w.reduce((s, x, i) => s + x * Cw[i], 0); const lr = 0.3 / (1 + iter * 0.1);
    for (let i = 0; i < N; i++) w[i] -= lr * (2 * Cw[i] - 2 * wCw);
    const sumW = w.reduce((s, x) => s + x, 0); w = w.map(x => x / sumW);
  }
  return w;
}

function portVol(w, C) { let v = 0; for (let i = 0; i < w.length; i++) for (let j = 0; j < w.length; j++) v += w[i] * w[j] * C[i][j]; return Math.sqrt(Math.max(0, v)); }

// ============================================================
// DESIGN TOKENS
// ============================================================
const NAVY = "#0B1D3A";
const TEAL = "#00B4D8";
const GOLD = "#D4A843";
const RED = "#DC2626";
const GREEN = "#059669";
const MED = "#64748B";

// ============================================================
// STEP COMPONENTS
// ============================================================

function StepIndicator({ current, total }) {
  return (
    <div className="flex items-center gap-2 mb-6">
      {Array.from({ length: total }, (_, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-500 ${
            i < current ? "bg-cyan-500 text-white" : i === current ? "bg-cyan-500 text-white ring-4 ring-cyan-100" : "bg-slate-200 text-slate-400"
          }`}>
            {i + 1}
          </div>
          {i < total - 1 && <div className={`w-8 h-0.5 transition-all duration-500 ${i < current ? "bg-cyan-400" : "bg-slate-200"}`} />}
        </div>
      ))}
    </div>
  );
}

function NavButton({ onClick, children, primary = false, disabled = false }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-6 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 ${
        disabled ? "opacity-40 cursor-not-allowed" :
        primary
          ? "bg-cyan-500 text-white hover:bg-cyan-600 shadow-sm"
          : "bg-slate-100 text-slate-600 hover:bg-slate-200"
      }`}
    >
      {children}
    </button>
  );
}

// Custom bar label that only shows on extreme values
const ExtremeLabel = (props) => {
  const { x, y, width, value } = props;
  if (Math.abs(value) < 0.04) return null;
  return (
    <text x={x + width / 2} y={value >= 0 ? y - 4 : y + 14} textAnchor="middle" fill={MED} fontSize={8}>
      {(value * 100).toFixed(0)}%
    </text>
  );
};

// ============================================================
// MAIN
// ============================================================
export default function CFMExplorer() {
  const [step, setStep] = useState(0);
  const [animQ, setAnimQ] = useState(0.35);
  const [isAnimating, setIsAnimating] = useState(false);
  const animRef = useRef(null);
  const TOTAL_STEPS = 5;

  // Fixed scenario: 50 assets
  const N = 50;
  const seed = 42;

  // Compute data for the animated q value
  const data = useMemo(() => {
    const T = Math.round(N / animQ);
    const rng = mulberry32(seed);
    const X = generateData(T, N, rng);
    const C = corrMatrix(X);
    const eigenvalues = getEigenvalues(C);
    const q = animQ;
    const lp = (1 + Math.sqrt(q)) ** 2;
    const lm = (1 - Math.sqrt(q)) ** 2;

    const maxEig = Math.max(...eigenvalues, lp) * 1.15;
    const numBins = 25;
    const binW = maxEig / numBins;
    const histogram = Array.from({ length: numBins }, (_, i) => {
      const lo = i * binW, hi = (i + 1) * binW, mid = (lo + hi) / 2;
      const count = eigenvalues.filter(e => e >= lo && e < hi).length;
      return { bin: mid.toFixed(2), density: count / (N * binW), mp: marchenkoPastur(mid, q), isSignal: mid > lp };
    });

    const signalCount = eigenvalues.filter(e => e > lp).length;
    const noiseCount = N - signalCount;

    const { cleaned } = cleanMatrix(C, q);
    const wRaw = minVarWeights(C);
    const wClean = minVarWeights(cleaned);
    const volRaw = portVol(wRaw, C);
    const volClean = portVol(wClean, cleaned);
    const volTruth = portVol(wClean, C);

    const weightData = wRaw.map((w, i) => ({ asset: `${i + 1}`, raw: w, cleaned: wClean[i] })).sort((a, b) => a.raw - b.raw);
    const maxAbsRaw = Math.max(...wRaw.map(Math.abs));
    const maxAbsClean = Math.max(...wClean.map(Math.abs));
    const hhiRaw = wRaw.reduce((s, w) => s + w * w, 0);
    const hhiClean = wClean.reduce((s, w) => s + w * w, 0);

    return { histogram, lp, signalCount, noiseCount, q, T, weightData, volRaw, volClean, volTruth, maxAbsRaw, maxAbsClean, hhiRaw, hhiClean };
  }, [animQ]);

  // Step 2 animation: slowly increase q
  const startAnimation = useCallback(() => {
    if (isAnimating) return;
    setIsAnimating(true);
    setAnimQ(0.05);
    let current = 0.05;
    const tick = () => {
      current += 0.008;
      if (current >= 0.85) {
        setAnimQ(0.85);
        setIsAnimating(false);
        return;
      }
      setAnimQ(Math.round(current * 100) / 100);
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
  }, [isAnimating]);

  useEffect(() => {
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  const resetAnimation = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    setIsAnimating(false);
    setAnimQ(0.35);
  }, []);

  const next = () => { resetAnimation(); setStep(s => Math.min(s + 1, TOTAL_STEPS - 1)); };
  const prev = () => { resetAnimation(); setStep(s => Math.max(s - 1, 0)); };

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div style={{ backgroundColor: NAVY }} className="px-6 py-6">
        <div className="max-w-3xl mx-auto">
          <div className="text-cyan-400 text-xs font-semibold tracking-widest mb-1">CFM RESEARCH</div>
          <h1 className="text-white text-xl font-bold font-serif">
            Why Your Portfolio Is Built on Patterns That Won't Last
          </h1>
          <p className="text-slate-400 mt-1 text-xs">
            An interactive guide to correlation noise and what CFM does about it
          </p>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-6">
        <StepIndicator current={step} total={TOTAL_STEPS} />

        {/* ============ STEP 0: THE SETUP ============ */}
        {step === 0 && (
          <div>
            <h2 className="text-2xl font-bold text-slate-800 font-serif mb-4">
              Every portfolio manager relies on correlations
            </h2>
            <div className="bg-slate-50 rounded-xl p-6 border border-slate-200 mb-5">
              <p className="text-slate-600 leading-relaxed">
                To build a diversified portfolio, you need to know how assets move together.
                If two stocks always rise and fall in sync, holding both doesn't reduce your risk.
                If they move independently, it does.
              </p>
              <p className="text-slate-600 leading-relaxed mt-3">
                This is captured in a <span className="font-semibold text-slate-800">correlation matrix</span> —
                a table measuring the relationship between every pair of assets.
                For a 50-stock portfolio, that's <span className="font-semibold text-slate-800">1,225 correlations</span> to estimate.
              </p>
              <p className="text-slate-600 leading-relaxed mt-3">
                The problem? <span className="font-semibold" style={{ color: RED }}>Most of those measured correlations won't persist.</span> They're artefacts of the specific window of data you happened to use — patterns that showed up by chance and won't be there next month when your portfolio needs them.
              </p>
            </div>
            <div className="bg-cyan-50 border border-cyan-200 rounded-xl p-5">
              <div className="text-xs font-bold tracking-wide mb-2" style={{ color: TEAL }}>THE CORE INSIGHT</div>
              <p className="text-cyan-900 text-sm leading-relaxed">
                When you estimate 1,225 relationships from a limited amount of data — say, one year of daily prices —
                your matrix is contaminated by <span className="font-semibold">coincidental patterns that look exactly like genuine ones</span>.
                Your optimiser can't tell the difference. It treats accidental correlations as stable relationships, and builds
                your portfolio around them. When those patterns vanish — which they will — the portfolio behaves nothing like the model predicted.
              </p>
            </div>
          </div>
        )}

        {/* ============ STEP 1: THE MATHS (VISUAL) ============ */}
        {step === 1 && (
          <div>
            <h2 className="text-2xl font-bold text-slate-800 font-serif mb-2">
              There's a way to know exactly how much is noise
            </h2>
            <p className="text-sm text-slate-500 mb-4 leading-relaxed">
              In the 1960s, mathematicians proved that a matrix of <em>purely random</em> numbers produces a
              predictable statistical fingerprint — the <span className="font-semibold text-slate-700">Marchenko-Pastur distribution</span>.
              Anything in your real correlation matrix that fits inside this fingerprint is indistinguishable from noise.
              CFM's founders — Jean-Philippe Bouchaud and Marc Potters — developed a published, peer-reviewed method for separating the two.
            </p>
            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 mb-4">
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs font-bold text-slate-500 tracking-wide">50 ASSETS · {data.T} DAYS OF DATA · q = {data.q.toFixed(2)}</div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={data.histogram} margin={{ top: 10, right: 20, bottom: 25, left: 10 }}>
                  <XAxis dataKey="bin" tick={{ fontSize: 9, fill: MED }} label={{ value: "Strength of pattern (stronger →)", position: "bottom", fontSize: 10, fill: MED, offset: 0 }} />
                  <YAxis tick={{ fontSize: 9, fill: MED }} />
                  <ReferenceLine x={data.lp.toFixed(2)} stroke={RED} strokeDasharray="5 5" strokeWidth={2} />
                  <Bar dataKey="density" name="Your data" radius={[2, 2, 0, 0]}>
                    {data.histogram.map((e, i) => <Cell key={i} fill={e.isSignal ? RED : "#94A3B8"} fillOpacity={e.isSignal ? 0.85 : 0.35} />)}
                  </Bar>
                  <Line type="monotone" dataKey="mp" name="Pure randomness" stroke={TEAL} strokeWidth={2.5} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="bg-slate-100 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-slate-400">{N}</div>
                <div className="text-xs text-slate-500 mt-1">Patterns your model sees</div>
              </div>
              <div className="bg-green-50 rounded-lg p-4 text-center border border-green-200">
                <div className="text-2xl font-bold" style={{ color: GREEN }}>{data.signalCount}</div>
                <div className="text-xs text-green-700 mt-1">Genuine relationships</div>
              </div>
              <div className="bg-red-50 rounded-lg p-4 text-center border border-red-200">
                <div className="text-2xl font-bold" style={{ color: RED }}>{data.noiseCount}</div>
                <div className="text-xs text-red-700 mt-1">Coincidence</div>
              </div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <p className="text-sm text-amber-800">
                <span className="font-bold">Read this chart:</span> Each bar represents a pattern of co-movement in your portfolio — things like
                "the whole market rises and falls together" or "tech stocks move as a group." The{" "}
                <span className="text-slate-500 font-semibold">grey bars</span> are patterns weak enough to be pure coincidence — they fit inside what you'd find in
                completely random data (the{" "}
                <span style={{ color: TEAL }} className="font-semibold">teal curve</span>). Only the{" "}
                <span style={{ color: RED }} className="font-semibold">red bars</span> — the ones breaking through to the right of the dashed line —
                are strong enough to be genuine. Out of {N} apparent patterns, only {data.signalCount} are real.
              </p>
            </div>
          </div>
        )}

        {/* ============ STEP 2: WATCH IT GET WORSE ============ */}
        {step === 2 && (
          <div>
            <h2 className="text-2xl font-bold text-slate-800 font-serif mb-2">
              Now watch what happens as the problem gets harder
            </h2>
            <p className="text-sm text-slate-500 mb-4 leading-relaxed">
              The noise depends on one number: the ratio of assets to days of data.
              More assets with the same data? More noise.
              Press play and watch the noise take over in real time.
            </p>

            <div className="flex items-center gap-4 mb-4">
              <button
                onClick={isAnimating ? resetAnimation : startAnimation}
                className={`px-5 py-2 rounded-lg text-sm font-bold transition-all ${
                  isAnimating ? "bg-red-500 text-white hover:bg-red-600" : "bg-cyan-500 text-white hover:bg-cyan-600"
                }`}
              >
                {isAnimating ? "Reset" : "▶  Play"}
              </button>
              <div className="flex-1 bg-slate-100 rounded-full h-3 overflow-hidden">
                <div className="h-full rounded-full transition-all duration-100"
                  style={{ width: `${((animQ - 0.05) / 0.8) * 100}%`, backgroundColor: animQ > 0.6 ? RED : animQ > 0.3 ? GOLD : GREEN }} />
              </div>
              <div className="text-sm font-mono font-bold min-w-[100px]" style={{ color: animQ > 0.6 ? RED : animQ > 0.3 ? GOLD : GREEN }}>
                q = {animQ.toFixed(2)}
              </div>
            </div>

            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 mb-4">
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={data.histogram} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                  <XAxis dataKey="bin" tick={{ fontSize: 9, fill: MED }} />
                  <YAxis tick={{ fontSize: 9, fill: MED }} />
                  <ReferenceLine x={data.lp.toFixed(2)} stroke={RED} strokeDasharray="5 5" strokeWidth={2} />
                  <Bar dataKey="density" radius={[2, 2, 0, 0]}>
                    {data.histogram.map((e, i) => <Cell key={i} fill={e.isSignal ? RED : "#94A3B8"} fillOpacity={e.isSignal ? 0.85 : 0.35} />)}
                  </Bar>
                  <Line type="monotone" dataKey="mp" stroke={TEAL} strokeWidth={2.5} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="rounded-lg p-3 text-center" style={{ backgroundColor: animQ > 0.6 ? "#FEF2F2" : animQ > 0.3 ? "#FFFBEB" : "#ECFDF5" }}>
                <div className="text-lg font-bold" style={{ color: animQ > 0.6 ? RED : animQ > 0.3 ? GOLD : GREEN }}>
                  {((data.noiseCount / N) * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-600">of patterns are coincidence</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-3 text-center">
                <div className="text-lg font-bold text-slate-700">{data.signalCount}</div>
                <div className="text-xs text-slate-500">genuine relationships</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-3 text-center">
                <div className="text-lg font-bold text-slate-700">{(data.q).toFixed(2)}</div>
                <div className="text-xs text-slate-500">noise ratio (higher = worse)</div>
              </div>
            </div>

            <div className="bg-slate-800 rounded-xl p-5">
              <p className="text-sm text-slate-300 leading-relaxed">
                <span className="text-cyan-400 font-bold">What you just watched:</span> As the ratio increases,
                the noise distribution (teal curve) swells and swallows more of the bars. The genuine relationships
                get harder to distinguish. At q = 0.5, over 90% of apparent structure is noise.
                Most institutional portfolios operate in exactly this zone.
              </p>
            </div>
          </div>
        )}

        {/* ============ STEP 3: PORTFOLIO IMPACT ============ */}
        {step === 3 && (
          <div>
            <h2 className="text-2xl font-bold text-slate-800 font-serif mb-2">
              What does this do to your portfolio?
            </h2>
            <p className="text-sm text-slate-500 mb-4 leading-relaxed">
              This is where it matters for your capital. On the left: portfolio weights built on
              the raw, noisy correlation matrix. On the right: weights after CFM's cleaning methodology
              removes the noise. Same assets and data, but CFM reads the data differently.
            </p>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-red-50 rounded-xl p-4 border border-red-200">
                <div className="text-xs font-bold text-red-700 tracking-wide mb-1">STANDARD APPROACH</div>
                <div className="text-xs text-red-500 mb-3">Trusts the raw correlations</div>
                <div style={{ height: 200 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.weightData} margin={{ top: 15, right: 5, bottom: 5, left: 5 }}>
                      <XAxis dataKey="asset" tick={false} />
                      <YAxis tick={{ fontSize: 8, fill: MED }} domain={[Math.min(0, ...data.weightData.map(d => d.raw)) * 1.2, Math.max(...data.weightData.map(d => d.raw), ...data.weightData.map(d => d.cleaned)) * 1.1]} />
                      <ReferenceLine y={0} stroke="#CBD5E1" />
                      <Bar dataKey="raw" radius={[1, 1, 0, 0]}>
                        {data.weightData.map((d, i) => <Cell key={i} fill={d.raw >= 0 ? "#94A3B8" : RED} fillOpacity={0.6} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-center mt-3">
                  <span className="text-lg font-bold" style={{ color: RED }}>{Math.round(1 / data.hhiRaw)}</span>
                  <span className="text-xs text-red-600 ml-2">effective positions out of {N}</span>
                </div>
              </div>
              <div className="bg-green-50 rounded-xl p-4 border border-green-200">
                <div className="text-xs font-bold text-green-700 tracking-wide mb-1">CFM'S APPROACH</div>
                <div className="text-xs text-green-600 mb-3">Applies Bouchaud & Potters' cleaning</div>
                <div style={{ height: 200 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.weightData} margin={{ top: 15, right: 5, bottom: 5, left: 5 }}>
                      <XAxis dataKey="asset" tick={false} />
                      <YAxis tick={{ fontSize: 8, fill: MED }} domain={[Math.min(0, ...data.weightData.map(d => d.raw)) * 1.2, Math.max(...data.weightData.map(d => d.raw), ...data.weightData.map(d => d.cleaned)) * 1.1]} />
                      <ReferenceLine y={0} stroke="#CBD5E1" />
                      <Bar dataKey="cleaned" radius={[1, 1, 0, 0]}>
                        {data.weightData.map((d, i) => <Cell key={i} fill={d.cleaned >= 0 ? GREEN : RED} fillOpacity={0.6} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-center mt-3">
                  <span className="text-lg font-bold" style={{ color: GREEN }}>{Math.round(1 / data.hhiClean)}</span>
                  <span className="text-xs text-green-600 ml-2">effective positions out of {N}</span>
                </div>
              </div>
            </div>

            <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-4">
              <p className="text-sm text-cyan-800 leading-relaxed">
                <span className="font-bold">What you're seeing:</span> "Effective positions" measures how
                diversified a portfolio truly is. A 50-stock portfolio where one stock dominates might
                only have 15 effective positions — the rest are window dressing. The standard approach wastes
                diversification by fitting to noise. CFM's approach, developed by Bouchaud and Potters and
                published in <em>Risk</em> magazine, extracts more genuine diversification from the same assets.
              </p>
            </div>
          </div>
        )}

        {/* ============ STEP 4: THE PUNCHLINE ============ */}
        {step === 4 && (
          <div>
            <h2 className="text-2xl font-bold text-slate-800 font-serif mb-2">
              One approach gives you a false sense of precision. The other doesn't.
            </h2>
            <p className="text-sm text-slate-500 mb-5 leading-relaxed">
              Every optimiser produces a risk estimate. The question is whether that estimate
              reflects reality. Here's the difference between trusting noisy correlations and cleaning them first.
            </p>

            {/* STANDARD APPROACH */}
            <div className="bg-red-50 border border-red-200 rounded-xl p-5 mb-4">
              <div className="text-xs font-bold tracking-wide mb-4" style={{ color: RED }}>STANDARD APPROACH — TRUSTS RAW CORRELATIONS</div>
              <div className="flex items-center gap-4">
                <div className="text-center flex-1">
                  <div className="text-xs text-slate-500 mb-1">Model says</div>
                  <div className="text-3xl font-bold text-slate-400">{(data.volRaw * 100).toFixed(1)}%</div>
                  <div className="text-xs text-slate-400">volatility</div>
                </div>
                <div className="text-center px-3">
                  <div className="text-xs text-slate-400 mb-1">but</div>
                  <div className="text-2xl">→</div>
                </div>
                <div className="text-center flex-1">
                  <div className="text-xs text-red-600 mb-1">Reality is</div>
                  <div className="text-3xl font-bold" style={{ color: RED }}>{(data.volTruth * 100).toFixed(1)}%</div>
                  <div className="text-xs text-red-600">volatility</div>
                </div>
                <div className="text-center flex-1 bg-white rounded-lg p-3 border border-red-200">
                  <div className="text-xs text-red-600 mb-1">Risk underestimation</div>
                  <div className="text-2xl font-bold" style={{ color: RED }}>
                    {(((data.volTruth - data.volRaw) / data.volRaw) * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-red-500">worse than model predicted</div>
                </div>
              </div>
              <p className="text-xs text-red-700 mt-4 leading-relaxed">
                The optimiser fits to accidental correlations, producing a risk estimate that's lower than what you'll actually experience.
                You don't discover the gap until a drawdown hits harder than your model said it could.
              </p>
            </div>

            {/* CFM APPROACH */}
            <div className="bg-green-50 border border-green-200 rounded-xl p-5 mb-5">
              <div className="text-xs font-bold tracking-wide mb-4" style={{ color: GREEN }}>CFM'S APPROACH — CLEANS THE MATRIX FIRST</div>
              <div className="flex items-center gap-4">
                <div className="text-center flex-1">
                  <div className="text-xs text-slate-500 mb-1">Model says</div>
                  <div className="text-3xl font-bold" style={{ color: GREEN }}>{(data.volClean * 100).toFixed(1)}%</div>
                  <div className="text-xs text-green-600">volatility</div>
                </div>
                <div className="text-center px-3">
                  <div className="text-xs text-slate-400 mb-1">and</div>
                  <div className="text-2xl">→</div>
                </div>
                <div className="text-center flex-1">
                  <div className="text-xs text-green-600 mb-1">Reality is</div>
                  <div className="text-3xl font-bold" style={{ color: GREEN }}>~{(data.volClean * 100).toFixed(1)}%</div>
                  <div className="text-xs text-green-600">volatility</div>
                </div>
                <div className="text-center flex-1 bg-white rounded-lg p-3 border border-green-200">
                  <div className="text-xs text-green-600 mb-1">Surprise factor</div>
                  <div className="text-2xl font-bold" style={{ color: GREEN }}>
                    ~0%
                  </div>
                  <div className="text-xs text-green-600">what you see is what you get</div>
                </div>
              </div>
              <p className="text-xs text-green-700 mt-4 leading-relaxed">
                The cleaned portfolio may show a higher risk number — but it's an <span className="font-bold">honest</span> number.
                No hidden risk. No nasty surprises. The weights are more diversified, more stable, and the risk
                estimate reflects what you'll actually experience.
              </p>
            </div>

            {/* Dollar impact */}
            <div className="bg-slate-100 rounded-xl p-5 mb-5 border border-slate-200">
              <div className="text-xs font-bold text-slate-500 tracking-wide mb-2">WHAT THIS MEANS FOR A $500M PORTFOLIO</div>
              <p className="text-sm text-slate-700 leading-relaxed">
                The standard approach tells you to expect a worst-month loss of around{" "}
                <span className="font-bold">${((data.volRaw * 500)).toFixed(0)}M</span>.
                The actual loss could be{" "}
                <span className="font-bold" style={{ color: RED }}>${((data.volTruth * 500)).toFixed(0)}M</span> —
                {" "}that's <span className="font-bold">${(((data.volTruth - data.volRaw) * 500)).toFixed(0)}M you didn't know was at risk</span>.
                CFM's approach doesn't eliminate risk — it eliminates the gap between what you expect and what happens.
              </p>
            </div>

            <div style={{ backgroundColor: NAVY }} className="rounded-xl p-6">
              <div className="text-cyan-400 text-xs font-bold tracking-widest mb-3">WHY THIS MATTERS FOR MANAGER SELECTION</div>
              <p className="text-slate-300 text-sm leading-relaxed mb-3">
                This isn't a generic statistical technique available to anyone. The cleaning methodology
                was developed by CFM's own leadership — Jean-Philippe Bouchaud (Chairman, member of the
                French Academy of Sciences) and Marc Potters (CIO) — published in peer-reviewed journals,
                and is running in production across CFM's strategies today. The research <em className="text-white">is</em> the
                investment process.
              </p>
              <p className="text-slate-300 text-sm leading-relaxed mb-3">
                When evaluating systematic managers, ask: <em className="text-white">how do they handle the fact that
                most measured correlations are noise?</em> If the answer is
                "we use shrinkage" or "we use a shorter lookback window," those are
                rough approximations. CFM's approach is mathematically optimal — and you can read the
                published paper to verify it yourself.
              </p>
              <p className="text-slate-400 text-xs mt-4">
                Based on Bun, Bouchaud & Potters, <em>Cleaning Large Correlation Matrices:
                Tools from Random Matrix Theory</em>, Risk (2016). The interactive simulations above
                use a simplified version of CFM's methodology to illustrate the principle.
              </p>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex justify-between items-center mt-8 pt-6 border-t border-slate-200">
          <NavButton onClick={prev} disabled={step === 0}>
            ← Back
          </NavButton>
          <div className="text-xs text-slate-400">
            {step === 0 && "The problem"}
            {step === 1 && "The science"}
            {step === 2 && "The scale of it"}
            {step === 3 && "Portfolio impact"}
            {step === 4 && "The bottom line"}
          </div>
          <NavButton onClick={next} primary disabled={step === TOTAL_STEPS - 1}>
            Next →
          </NavButton>
        </div>
      </div>
    </div>
  );
}
