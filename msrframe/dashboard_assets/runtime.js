/* msrframe dashboard runtime: decode exported models and evaluate them in the browser.
 * Mirrors msrframe/export.py (predict_scalar / predict_trajectory) operation for operation,
 * so a model gives the same k_eff here as in Python (checked by tests/test_dashboard.py). */
'use strict';
const MSR = (() => {
  const atobSafe = typeof atob === 'function' ? atob : (s => Buffer.from(s, 'base64').toString('binary'));
  const TYPES = { f8: Float64Array, f4: Float32Array, i4: Int32Array, i2: Int16Array };

  function decodeArray(o) {
    const bin = atobSafe(o.data), n = bin.length, buf = new ArrayBuffer(n), u8 = new Uint8Array(buf);
    for (let i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
    const a = new TYPES[o.b64](buf);
    a.shape = o.shape;
    return a;
  }
  function decode(o) {
    if (Array.isArray(o)) return o.map(decode);
    if (o && typeof o === 'object') {
      if (typeof o.b64 === 'string' && typeof o.data === 'string') return decodeArray(o);
      const r = {};
      for (const k of Object.keys(o)) r[k] = decode(o[k]);
      return r;
    }
    return o;
  }

  // ---------------------------------------------------------------- parameter space
  function makeSpace(params) {
    const lo = params.map(p => (p.log ? Math.log10(p.low) : p.low));
    const hi = params.map(p => (p.log ? Math.log10(p.high) : p.high));
    return {
      params, keys: params.map(p => p.key), dim: params.length,
      nominal: params.map(p => p.nominal),
      toUnit(x) {
        const u = new Array(params.length);
        for (let j = 0; j < params.length; j++) {
          const t = params[j].log ? Math.log10(Math.max(x[j], 1e-300)) : x[j];
          u[j] = hi[j] > lo[j] ? (t - lo[j]) / (hi[j] - lo[j]) : t - lo[j];
        }
        return u;
      },
      fromUnit(u) {
        return u.map((v, j) => { const t = lo[j] + v * (hi[j] - lo[j]); return params[j].log ? Math.pow(10, t) : t; });
      },
    };
  }

  // ---------------------------------------------------------------- scalar models (standardised units)
  function matern(r, nu) {
    if (nu === 0.5) return Math.exp(-r);
    if (nu === 1.5) { const a = Math.sqrt(3) * r; return (1 + a) * Math.exp(-a); }
    if (nu === 2.5) { const a = Math.sqrt(5) * r; return (1 + a + 5 * r * r / 3) * Math.exp(-a); }
    return Math.exp(-0.5 * r * r);
  }
  function gp(s, u) {
    const X = s.X, n = s.n, d = s.d, ls = s.ls, alpha = s.alpha, k = new Float64Array(n);
    let mean = 0;
    for (let i = 0; i < n; i++) {
      let r2 = 0;
      const o = i * d;
      for (let j = 0; j < d; j++) { const t = (u[j] - X[o + j]) / ls[j]; r2 += t * t; }
      const kv = s.c * matern(Math.sqrt(r2), s.nu);
      k[i] = kv;
      mean += kv * alpha[i];
    }
    const L = s.L, v = new Float64Array(n);
    let vv = 0;
    for (let i = 0, off = 0; i < n; i++) {
      let acc = k[i];
      for (let j = 0; j < i; j++) acc -= L[off + j] * v[j];
      v[i] = acc / L[off + i];
      vv += v[i] * v[i];
      off += i + 1;
    }
    const tot = Math.max(s.c + s.noise_var - vv, 0);
    return [mean, Math.sqrt(Math.max(tot - s.noise_var, 0))];
  }
  function act(name, z) {
    if (name === 'tanh') return Math.tanh(z);
    if (name === 'relu') return z > 0 ? z : 0;
    if (name === 'logistic') return 1 / (1 + Math.exp(-z));
    return z;
  }
  function mlpForward(mem, u) {
    let h = u;
    const nl = mem.layers.length;
    for (let li = 0; li < nl; li++) {
      const ly = mem.layers[li], W = ly.W, b = ly.b, nin = ly.nin, nout = ly.nout, z = new Float64Array(nout);
      for (let o = 0; o < nout; o++) z[o] = b[o];
      for (let i = 0; i < nin; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const off = i * nout;
        for (let o = 0; o < nout; o++) z[o] += hi * W[off + o];
      }
      if (li < nl - 1) for (let o = 0; o < nout; o++) z[o] = act(mem.act, z[o]);
      h = z;
    }
    return h[0];
  }
  function mlp(s, u) {
    const P = s.members.map(m => mlpForward(m, u)), n = P.length;
    const mu = P.reduce((a, b) => a + b, 0) / n;
    if (!s.spread) return [mu, NaN];
    return [mu, Math.sqrt(P.reduce((a, b) => a + (b - mu) * (b - mu), 0) / n)];
  }
  function trees(s, u) {
    const u32 = u.map(Math.fround), R = s.roots, L = s.left, Rr = s.right, F = s.feat, T = s.thr, V = s.val, nt = R.length;
    const vals = new Float64Array(nt);
    for (let t = 0; t < nt; t++) {
      const r = R[t];
      let nd = r;
      while (L[nd] !== -1) nd = r + (u32[F[nd]] <= T[nd] ? L[nd] : Rr[nd]);
      vals[t] = V[nd];
    }
    let sum = 0;
    for (let t = 0; t < nt; t++) sum += vals[t];
    if (s.kind === 'gbt') return [s.init + s.lr * sum, NaN];
    const m = sum / nt;
    let v = 0;
    for (let t = 0; t < nt; t++) v += (vals[t] - m) * (vals[t] - m);
    return [m, Math.sqrt(v / nt)];
  }
  function pce(s, u) {
    const d = s.d, p = s.p, P = new Array(d);
    for (let j = 0; j < d; j++) {
      const x = 2 * u[j] - 1, row = new Float64Array(p + 1);
      row[0] = 1;
      if (p >= 1) row[1] = x;
      for (let k = 1; k < p; k++) row[k + 1] = ((2 * k + 1) * x * row[k] - k * row[k - 1]) / (k + 1);
      for (let k = 0; k <= p; k++) row[k] *= Math.sqrt(2 * k + 1);
      P[j] = row;
    }
    const A = s.A, coef = s.coef;
    let mu = 0;
    for (let t = 0; t < s.nt; t++) {
      let b = 1;
      const off = t * d;
      for (let j = 0; j < d; j++) { const a = A[off + j]; if (a) b *= P[j][a]; }
      mu += b * coef[t];
    }
    return [mu, s.res_sd];
  }
  function poly(s, u) {
    const nf = s.nf, d = s.d, Pw = s.powers, f = new Float64Array(nf);
    for (let i = 0; i < nf; i++) {
      let v = 1;
      for (let j = 0; j < d; j++) { const e = Pw[i * d + j]; if (e) v *= Math.pow(u[j], e); }
      f[i] = v;
    }
    let mu = s.intercept;
    for (let i = 0; i < nf; i++) mu += f[i] * s.coef[i];
    const S = s.sigma, fc = new Float64Array(nf);
    for (let i = 0; i < nf; i++) fc[i] = f[i] - s.offset[i];
    let q = 0;
    for (let i = 0; i < nf; i++) {
      let r = 0;
      for (let j = 0; j < nf; j++) r += fc[j] * S[j * nf + i];
      q += r * fc[i];
    }
    const tot = Math.sqrt(q + s.inv_alpha);
    return [mu, Math.sqrt(Math.max(tot * tot - s.inv_alpha, 0))];
  }
  function knn(s, u) {
    const n = s.n, d = s.d, X = s.X, dist = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let j = 0; j < d; j++) { const t = u[j] - X[i * d + j]; r += t * t; }
      dist[i] = Math.sqrt(r);
    }
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => dist[a] - dist[b] || a - b).slice(0, s.k);
    const anyZero = idx.some(i => dist[i] === 0);
    let sw = 0, sy = 0;
    for (const i of idx) { const w = anyZero ? (dist[i] === 0 ? 1 : 0) : 1 / dist[i]; sw += w; sy += w * s.y[i]; }
    return [sy / sw, NaN];
  }
  function rbfExpansion(s, u) {
    let f = s.b;
    for (let i = 0; i < s.n; i++) {
      let r = 0;
      for (let j = 0; j < s.d; j++) { const t = u[j] - s.X[i * s.d + j]; r += t * t; }
      f += s.w[i] * Math.exp(-s.gamma * r);
    }
    return [f, NaN];
  }

  /** Predict one point u (unit cube) -> [mean, sd] in output units; sd is the model's own
   *  uncertainty about the mean (Monte Carlo label noise excluded), NaN if it has none. */
  function predictScalar(s, u) {
    if (s.type === 'ensemble') {
      const R = s.members.map(m => predictScalar(m, u)), n = R.length;
      const mu = R.reduce((a, r) => a + r[0], 0) / n;
      const al = R.reduce((a, r) => a + (Number.isFinite(r[1]) ? r[1] * r[1] : 0), 0) / n;
      const va = R.reduce((a, r) => a + (r[0] - mu) * (r[0] - mu), 0) / n;
      return [mu, Math.sqrt(al + va)];
    }
    let r;
    switch (s.type) {
      case 'gp': r = gp(s, u); break;
      case 'mlp': r = mlp(s, u); break;
      case 'trees': r = trees(s, u); break;
      case 'pce': r = pce(s, u); break;
      case 'poly': r = poly(s, u); break;
      case 'knn': r = knn(s, u); break;
      case 'rbf_expansion': r = rbfExpansion(s, u); break;
      default: throw new Error('unknown exported model type ' + s.type);
    }
    const mu = r[0] * s.ys + s.ym;
    if (s.sd_model) return [mu, Math.exp(predictScalar(s.sd_model, u)[0])];
    return [mu, (!s.std || !Number.isFinite(r[1])) ? NaN : r[1] * s.ys];
  }

  // ---------------------------------------------------------------- trajectory models
  /** Predict trajectories at one point u -> {output: {mu: Float64Array(T), sd: Float64Array(T)}}.
   *  noise=false gives the uncertainty about the mean (used by the optimiser and the bands). */
  function predictTrajectory(s, u, noise = false, outputs = null) {
    const res = {}, outs = outputs || Object.keys(s.outputs);
    if (s.type === 'pca') {
      for (const o of outs) {
        const O = s.outputs[o], T = O.mu.length, mu = new Float64Array(T), va = new Float64Array(T).fill(O.trunc);
        let has = true;
        O.models.forEach((m, c) => {
          const [a, sd] = predictScalar(m, u);
          if (!m.std) has = false;
          const s2 = (Number.isFinite(sd) ? sd * sd : 0) + (noise ? m.noise * m.noise : 0);
          for (let t = 0; t < T; t++) { const v = O.V[c * T + t]; mu[t] += a * v; va[t] += s2 * v * v; }
        });
        const outMu = new Float64Array(T), outSd = new Float64Array(T);
        for (let t = 0; t < T; t++) { outMu[t] = O.mu[t] + mu[t] * O.sc; outSd[t] = has ? Math.sqrt(va[t]) * O.sc : NaN; }
        res[o] = { mu: outMu, sd: outSd };
      }
    } else if (s.type === 'time_input') {
      const tn = s.tn, T = tn.length, uu = Array.from(u).concat([0]);
      for (const o of outs) {
        const O = s.outputs[o], mu = new Float64Array(T), sd = new Float64Array(T);
        for (let t = 0; t < T; t++) {
          uu[uu.length - 1] = tn[t];
          const [a, sg] = predictScalar(O.model, uu);
          mu[t] = O.mu[t] + a * O.sc;
          sd[t] = (noise ? Math.sqrt(sg * sg + O.model.noise * O.model.noise) : sg) * O.sc;
        }
        res[o] = { mu, sd };
      }
    } else if (s.type === 'per_time') {
      for (const o of outs) {
        const ms = s.outputs[o], T = ms.length, mu = new Float64Array(T), sd = new Float64Array(T);
        ms.forEach((m, t) => {
          const [a, sg] = predictScalar(m, u);
          mu[t] = a;
          sd[t] = Math.sqrt(sg * sg + (noise ? m.noise * m.noise : 0));
        });
        res[o] = { mu, sd };
      }
    } else throw new Error('unknown trajectory type ' + s.type);
    return res;
  }
  /** Average of several trajectory models; sd combines their sd and their spread. */
  function predictTrajectoryEnsemble(specs, u, noise = false, outputs = null) {
    const parts = specs.map(s => predictTrajectory(s, u, noise, outputs)), res = {}, n = parts.length;
    for (const o of Object.keys(parts[0])) {
      const T = parts[0][o].mu.length, mu = new Float64Array(T), sd = new Float64Array(T);
      for (let t = 0; t < T; t++) {
        let m = 0, al = 0, va = 0;
        for (const p of parts) m += p[o].mu[t];
        m /= n;
        for (const p of parts) { const s = p[o].sd[t]; al += Number.isFinite(s) ? s * s : 0; va += (p[o].mu[t] - m) ** 2; }
        mu[t] = m;
        sd[t] = Math.sqrt(al / n + va / n);
      }
      res[o] = { mu, sd };
    }
    return res;
  }

  return { decode, makeSpace, predictScalar, predictTrajectory, predictTrajectoryEnsemble };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = MSR;
