/* msrframe dashboard application: one tab per stage, built lazily from the embedded DATA. */
'use strict';
const DATA = JSON.parse(document.getElementById('msr-data').textContent);
const $ = (s, r = document) => r.querySelector(s);
const fmt = Charts.fmtSig;
const isF = Number.isFinite;
const rho = k => (k - 1) / k * 1e5;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const pcmTxt = v => (isF(v) ? `${v < 0 ? '−' : v > 0 ? '+' : ''}${Math.abs(Math.round(v)).toLocaleString('en-US')}` : '–');
const unitTxt = u => (!u || u === '-' ? '' : ' ' + u);
const debounce = (f, ms = 60) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const geomspace = (a, b, n) => linspace(Math.log10(a), Math.log10(b), n).map(v => Math.pow(10, v));

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat(Infinity)) if (k !== null && k !== undefined && k !== false) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
function panel(title, sub, ...kids) {
  return el('div', { class: 'panel' }, title ? el('h3', { text: title }) : null, sub ? el('p', { class: 'psub', text: sub }) : null, ...kids);
}
function tile(label, value, unit, detail, slot) {
  return el('div', { class: 'tile' },
    el('span', { class: 'tl' }, slot !== undefined ? Charts.swatch(slot, { line: true }) : null, label),
    el('span', { class: 'v' }, String(value), unit ? el('small', { text: unit }) : null),
    detail ? el('span', { class: 'd', text: detail }) : null);
}
function table(headers, rows, opts = {}) {
  const t = el('table', {},
    el('thead', {}, el('tr', {}, headers.map(h => el('th', { class: h.num ? 'num' : null, text: h.t })))),
    el('tbody', {}, rows.map((r, i) => el('tr', { class: opts.hl && opts.hl(i) ? 'hl' : null },
      r.map((c, j) => el('td', { class: headers[j] && headers[j].num ? 'num' : null }, c))))));
  return el('div', { class: 'tablewrap' + (opts.scroll ? ' scroll' : '') }, t);
}
function segmented(options, current, onPick) {
  const wrap = el('div', { class: 'seg', role: 'group' });
  const btns = options.map(o => {
    const b = el('button', { type: 'button', 'aria-pressed': String(o.value === current), text: o.label });
    b.addEventListener('click', () => { btns.forEach(x => x.setAttribute('aria-pressed', String(x === b))); onPick(o.value); });
    return b;
  });
  wrap.append(...btns);
  wrap.set = v => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(options[i].value === v)));
  return wrap;
}
function download(name, text, type = 'text/csv') {
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type })), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
const csvCell = v => (typeof v === 'string' && /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : (v === null || v === undefined ? '' : String(v)));
const toCSV = (cols, rows) => [cols.join(','), ...rows.map(r => cols.map(c => csvCell(r[c])).join(','))].join('\n');

// ---------------------------------------------------------------- model names & colour slots
const BASE_NAME = { gp: 'Gaussian process', dnn: 'Neural-network ensemble', gbt: 'Gradient-boosted trees', random_forest: 'Random forest',
  extra_trees: 'Extra trees', pce: 'Polynomial chaos', svr: 'Support-vector regression', krr: 'Kernel ridge', poly_ridge: 'Quadratic ridge',
  linear: 'Bayesian linear', knn: 'k-nearest neighbours', ensemble: 'Ensemble', xgboost: 'XGBoost' };
const TRAJ_NAME = { pca: 'PCA + ', time_input: 'time-as-input ', per_time: 'per-time-point ' };
const baseOf = l => (l.includes(':') ? l.split(':').pop() : l);
const lowerFirst = n => (n.startsWith('Gaussian') ? n : n.charAt(0).toLowerCase() + n.slice(1));
function longName(label) {
  if (!label.includes(':')) return BASE_NAME[label] || label;
  const [a, b] = label.split(':');
  const base = BASE_NAME[b] || b;
  return a === 'pca' ? 'PCA + ' + base : (TRAJ_NAME[a] || a + ' + ') + lowerFirst(base);
}
const BASE_SLOT = {};
let nextSlot = 1;
/** Colour slots unique within a group, and the same for the same learner across groups. */
function assignSlots(labels) {
  const out = {}, used = new Set();
  const fresh = () => {
    for (let k = 0; k < 64; k++) { const s = ((nextSlot - 1) % 8) + 1; nextSlot++; if (!used.has(s)) return s; }
    return 1;
  };
  labels.forEach(l => {
    const b = baseOf(l);
    let s = BASE_SLOT[b];
    if (!s || used.has(s)) { s = fresh(); if (!BASE_SLOT[b]) BASE_SLOT[b] = s; }
    out[l] = s;
    used.add(s);
  });
  return out;
}
function keyName(label, slot) {
  return el('span', { class: 'kname' }, Charts.swatch(slot, { line: true }), el('span', { text: label }));
}

// ---------------------------------------------------------------- slider for one parameter (log-aware)
function slider(p, value, onInput) {
  const toS = v => (p.log ? Math.log10(v) : v), fromS = s => (p.log ? Math.pow(10, s) : s);
  const lo = toS(p.low), hi = toS(p.high);
  const inp = el('input', { type: 'range', min: lo, max: hi, step: (hi - lo) / 1000, value: toS(value), 'aria-label': p.name || p.key });
  const val = el('span', { class: 'val' });
  const show = v => { val.textContent = fmt(v, 4) + unitTxt(p.unit); };
  inp.addEventListener('input', () => { const v = fromS(+inp.value); show(v); onInput(v); });
  show(value);
  const wrap = el('div', { class: 'ctrl' },
    el('div', { class: 'row' }, el('span', { class: 'name' }, p.name || p.key, ' ', el('span', { class: 'key', text: p.name && p.name !== p.key ? p.key : '' })), val),
    inp,
    el('div', { class: 'ticks' }, el('span', { text: fmt(p.low, 3) + (p.log ? ' (log scale)' : '') }), el('span', { text: fmt(p.high, 3) })));
  wrap.setValue = v => { inp.value = toS(v); show(v); };
  return wrap;
}
function numField(label, value, unit, hint, step = 'any') {
  const inp = el('input', { type: 'number', value: value === null || value === undefined ? '' : value, step, 'aria-label': label });
  const row = el('div', { class: 'field' }, el('label', { class: 'name', text: label }),
    el('span', { class: 'numunit' }, inp, el('span', { class: 'unit', text: unit || '' })), hint ? el('span', { class: 'hint', text: hint }) : null);
  row.input = inp;
  row.num = () => (inp.value === '' ? null : +inp.value);
  return row;
}

// ================================================================ Stage 1: screening
function viewScreening(root) {
  const S = DATA.screening, methods = S.methods, slots = S._slots;
  const final = new Set(S.consensus), forced = S.forced || {};
  root.append(el('div', { class: 'lede' },
    el('div', { class: 'stack' }, el('h2', { text: 'Which parameters matter for k_eff?' }), el('p', { text: S.verdict })),
    el('div', { class: 'tiles' },
      tile('inputs screened', S.n_inputs, '', `${S.n_runs.toLocaleString('en-US')} transport runs`),
      tile('methods compared', methods.length, '', methods.join(' · ')),
      tile('kept for the surrogates', S.consensus.length, '', `≥ ${S.min_votes} votes${Object.keys(forced).length ? ' + safety-forced' : ''}`),
      isF(S.final_cv_rmse_pcm) ? tile('GP error, kept inputs only', Math.round(S.final_cv_rmse_pcm), 'pcm', '5-fold cross-validated') : null)));
  root.append(panel('Screening methods', 'Threshold = 2 × the largest score of an independent dummy input (the noise floor). Separation = weakest kept input ÷ strongest dummy: larger means a cleaner cut.',
    table([{ t: 'method' }, { t: 'index' }, { t: 'kept', num: true }, { t: 'threshold', num: true }, { t: 'dummy max', num: true }, { t: 'separation', num: true }, { t: 'time (s)', num: true }],
      S.methods_table.map(r => [keyName(r.method, slots[r.method]), r.units, r.kept, fmt(r.threshold, 3), fmt(r.dummy_max, 3),
        isF(r.dummy_separation) ? fmt(r.dummy_separation, 3) + '×' : '∞', fmt(r.time_s, 2)]))));
  const host = el('div');
  let showAll = false;
  const seg = segmented([{ value: 'top', label: 'Top 25' }, { value: 'all', label: `All ${S.n_inputs}` }], 'top', v => { showAll = v === 'all'; drawDots(); });
  root.append(el('div', { class: 'panel' }, el('div', { class: 'phead' }, el('h3', { text: 'Importance of each parameter' }), seg),
    el('p', { class: 'psub', text: 'Share of each method’s total importance (log scale). Bold rows form the reduced set the k_eff and burnup surrogates use; dummies show the noise floor.' }), host));
  function drawDots() {
    const rows = (showAll ? S.table : S.table.slice(0, 25)).map(r => ({
      label: r.name || r.key, full: r.name && r.name !== r.key ? `${r.name} (${r.key})` : r.key, bold: final.has(r.key),
      cells: Object.fromEntries(methods.map(m => [m, { v: r.share[m], keep: r.keep[m], rank: r.rank[m] }])) }));
    S.dummies.forEach(d => rows.push({ label: `${d.key} (dummy)`, muted: true, cells: Object.fromEntries(methods.map(m => [m, { v: d.share[m], keep: false, rank: d.rank[m] }])) }));
    Charts.dotPlot(host, { rows, methods: methods.map(m => ({ name: m, slot: slots[m] })), thresholds: S.thr_share });
  }
  const rHost = el('div');
  const consensus = el('div', { class: 'chips' }, S.consensus.map(k => el('span', { class: 'chip pass' }, el('span', { class: 'dot' }), k + (forced[k] ? ' (forced)' : ''))));
  const forcedNotes = Object.entries(forced).map(([k, why]) => el('p', { class: 'hint', text: `${k}: ${why}` }));
  root.append(el('div', { class: 'grid2' },
    panel('Retention curves', 'Cross-validated GP error using only each method’s top-m inputs: a better ranking gets low error with fewer inputs.', rHost),
    el('div', { class: 'stack' },
      panel('Agreement between methods', 'Jaccard = overlap of the kept sets; Spearman = rank correlation over the leading inputs.',
        table([{ t: 'pair' }, { t: 'Jaccard', num: true }, { t: 'Spearman (top)', num: true }],
          S.agreement.map(r => [r.pair, fmt(r.jaccard, 3), fmt(r.spearman_top, 3)]))),
      panel('Reduced set', null, consensus, ...forcedNotes))));
  function drawRet() {
    if (!S.retention.length) { rHost.replaceChildren(el('p', { class: 'muted', text: 'Retention curves were not computed (screening.validation.enabled = false).' })); return; }
    Charts.lineChart(rHost, { height: 260, xLabel: 'number of top-ranked inputs kept', yLabel: 'CV RMSE (pcm)', yLog: true, xName: 'inputs kept',
      series: methods.map(m => ({ name: m, slot: slots[m], markers: true, pts: S.retention.filter(r => r.method === m).map(r => [r.m, r.rmse]) })),
      tipFmt: v => Math.round(v).toLocaleString('en-US') + ' pcm' });
  }
  if (DATA.catalogue) root.append(catalogueExplorer(DATA.catalogue, S));
  drawDots();
  drawRet();
  return () => { drawDots(); drawRet(); };
}
function catalogueExplorer(cat, S) {
  const votes = {}, kept = new Set(S ? S.consensus : []), forced = (S && S.forced) || {};
  if (S) S.table.forEach(r => { votes[r.key] = r.votes; });
  const classes = [...new Set(cat.map(r => r.cls))];
  let cls = null, q = '';
  const body = el('div');
  const chips = el('div', { class: 'chips' }, [null, ...classes].map(c => {
    const b = el('button', { type: 'button', class: 'fchip', 'aria-pressed': String(c === null), text: c === null ? 'All classes' : c });
    b.addEventListener('click', () => { cls = c; chips.querySelectorAll('.fchip').forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw(); });
    return b;
  }));
  const search = el('input', { type: 'search', placeholder: 'Search parameters…', 'aria-label': 'search parameters' });
  search.addEventListener('input', () => { q = search.value.toLowerCase(); draw(); });
  function draw() {
    const rows = cat.filter(r => (!cls || r.cls === cls) && (!q || `${r.key} ${r.name} ${r.note || ''}`.toLowerCase().includes(q)));
    body.replaceChildren(table([{ t: 'parameter' }, { t: 'class' }, { t: 'nominal', num: true }, { t: 'range', num: true }, { t: 'unit' }, { t: 'stage' }, { t: 'screening' }],
      rows.map(r => [el('span', {}, r.name, el('br'), el('span', { class: 'hint', text: r.key + (r.note ? ' · ' + r.note : '') })), r.cls, fmt(r.nominal, 4),
        `${fmt(r.low, 3)} – ${fmt(r.high, 3)}`, r.unit || '', r.stage || '',
        kept.has(r.key) ? (forced[r.key] ? 'kept (forced)' : `kept (${votes[r.key]} votes)`) : (r.key in votes ? `dropped (${votes[r.key]} votes)` : 'not screened')]),
      { scroll: true, hl: i => kept.has(rows[i].key) }));
  }
  draw();
  return el('div', { class: 'panel' }, el('div', { class: 'phead' }, el('h3', { text: `Parameter catalogue (${cat.length})` }), search),
    el('p', { class: 'psub', text: 'Every parameter the framework knows, with the range used for sampling. Highlighted rows are in the reduced set.' }), chips, el('div', { style: 'margin-top:10px' }, body));
}

// ================================================================ Stage 2: k_eff predictor
function viewKeff(root) {
  const K = DATA.keff, space = MSR.makeSpace(K.space), keys = space.keys, P = K.space;
  const models = K.order, specs = K._specs, slots = K._slots;
  const M = Object.fromEntries(K.metrics.map(r => [r.model, r]));
  const x = space.nominal.slice();
  let loaded = null;                                   // index of a loaded OpenMC run, while unchanged
  if (!K.runs.U) K.runs.U = K.runs.X.map(r => space.toUnit(r));
  const pred = (lab, xx) => MSR.predictScalar(specs[lab], space.toUnit(xx));
  const stdModel = models.find(l => specs[l].std);
  const noise = isF(K.noise_pcm) ? K.noise_pcm : null;
  root.append(el('div', { class: 'lede' },
    el('div', { class: 'stack' }, el('h2', { text: 'k_eff without running OpenMC' }),
      el('p', { text: `${models.length} surrogate${models.length > 1 ? 's' : ''} trained on ${K.n_runs} OpenMC k-eigenvalue runs over ${keys.length} inputs. Move a slider and every model answers instantly in your browser. The errors quoted are 5-fold cross-validated: each run is predicted by a model that never saw it.` }),
      el('p', { class: 'hint', text: K.verdict })),
    el('div', { class: 'tiles' }, models.map(l => {
      const r = M[l] || {};
      return tile(l, isF(r.rmse) ? Math.round(r.rmse) : '–', 'pcm', `${longName(l)} · R² ${fmt(r.r2, 5)}${isF(r.coverage95) ? ` · 95% interval covers ${Math.round(100 * r.coverage95)}% of runs` : ''}`, slots[l]);
    }))));

  // ---- controls
  const sliders = {};
  keys.forEach((k, i) => { sliders[k] = slider(P[i], x[i], v => { x[i] = v; loaded = null; update(); }); });
  const runSel = el('select', { 'aria-label': 'load an OpenMC run' }, el('option', { value: '', text: 'Load an OpenMC run…' }),
    K.runs.ids.map((id, i) => el('option', { value: i, text: `${id} · k = ${K.runs.y[i].toFixed(5)}` })));
  runSel.addEventListener('change', () => {
    if (runSel.value === '') return;
    const i = +runSel.value;
    K.runs.X[i].forEach((v, j) => { x[j] = v; sliders[keys[j]].setValue(v); });
    loaded = i;
    update();
  });
  const critSel = el('select', { 'aria-label': 'input to solve for criticality' }, keys.map((k, i) => el('option', { value: k, text: P[i].name || k, selected: k === (K.derived_cfg.critical || keys[0]) })));
  const critOut = el('div', { class: 'stack', style: 'gap:4px;margin-top:6px' });
  const ctrls = el('div', { class: 'panel' }, el('h3', { text: 'Design inputs' }), ...keys.map(k => sliders[k]),
    el('div', { class: 'btnrow', style: 'margin-top:10px' },
      el('button', { class: 'btn', type: 'button', text: 'Reset to nominal', onclick: () => { keys.forEach((k, i) => { x[i] = P[i].nominal; sliders[k].setValue(x[i]); }); loaded = null; runSel.value = ''; update(); } }),
      runSel),
    el('div', { class: 'section-label label', text: 'Criticality search' }),
    el('div', { class: 'btnrow' }, 'Solve k_eff = 1 for', critSel, el('button', { class: 'btn primary', type: 'button', text: 'Solve', onclick: () => findCritical() })),
    critOut);
  function findCritical() {
    const j = keys.indexOf(critSel.value), p = P[j], rows = [];
    models.forEach(l => {
      const f = v => { const xx = x.slice(); xx[j] = v; return pred(l, xx)[0] - 1; };
      const tr = v => (p.log ? Math.log10(v) : v), un = t => (p.log ? Math.pow(10, t) : t);
      let a = tr(p.low), b = tr(p.high), fa = f(un(a)), fb = f(un(b));
      if (fa * fb > 0) { rows.push([keyName(l, slots[l]), `k = 1 not reached in ${fmt(p.low, 3)}–${fmt(p.high, 3)}${unitTxt(p.unit)}`]); return; }
      for (let it = 0; it < 60; it++) { const c = (a + b) / 2, fc = f(un(c)); if (fa * fc <= 0) { b = c; fb = fc; } else { a = c; fa = fc; } }
      rows.push([keyName(l, slots[l]), `${fmt(un((a + b) / 2), 5)}${unitTxt(p.unit)}`]);
    });
    critOut.replaceChildren(el('p', { class: 'hint', text: `${p.name || p.key} giving k_eff = 1 with the other inputs as set:` }), table([{ t: 'model' }, { t: 'critical value', num: true }], rows));
  }

  // ---- outputs
  const readouts = el('div', { class: 'readouts' }), agree = el('p', { class: 'note' }), near = el('p', { class: 'hint' });
  const sweepHost = el('div'), sensHost = el('div');
  let sweepKey = K.derived_cfg.critical && keys.includes(K.derived_cfg.critical) ? K.derived_cfg.critical : keys[0];
  const sweepSel = el('select', { 'aria-label': 'input to sweep' }, keys.map((k, i) => el('option', { value: k, text: P[i].name || k, selected: k === sweepKey })));
  sweepSel.addEventListener('change', () => { sweepKey = sweepSel.value; drawSweep(); });
  root.append(el('div', { class: 'work' }, ctrls, el('div', { class: 'stack' },
    el('div', { class: 'panel' }, el('h3', { text: 'Predicted k_eff at these inputs' }), readouts, el('div', { class: 'stack', style: 'gap:6px;margin-top:10px' }, agree, near)),
    el('div', { class: 'panel' }, el('div', { class: 'phead' }, el('h3', { text: 'Sweep through the current point' }), sweepSel),
      el('p', { class: 'psub', text: `Other inputs held at their current values; shaded: ${stdModel || 'best model'} ±2σ (uncertainty about the mean).` }), sweepHost),
    panel('Local sensitivities at this point', 'Change of reactivity per unit of each input (central difference over 2% of its range; per decade for log-scaled inputs).', sensHost))));

  function update() {
    const res = models.map(l => [l, pred(l, x)]);
    const truth = loaded !== null ? { y: K.runs.y[loaded], sd: K.runs.y_std ? K.runs.y_std[loaded] : null, id: K.runs.ids[loaded] } : null;
    readouts.replaceChildren(
      ...(truth ? [el('div', { class: 'ro truth' }, el('span', { class: 'tl' }, Charts.swatch(0, { line: false }), ` OpenMC ${truth.id}`),
        el('span', { class: 'k num', text: truth.y.toFixed(5) }),
        el('span', { class: 'pm', text: `ρ ${pcmTxt(rho(truth.y))} pcm${isF(truth.sd) ? ` · ±${Math.round(truth.sd * 1e5)} pcm Monte Carlo` : ''}` }))] : []),
      ...res.map(([l, [k, s]]) => {
        const cv = truth && K.runs.cv[l] ? K.runs.cv[l][loaded] : null;
        return el('div', { class: 'ro' }, el('span', { class: 'tl' }, Charts.swatch(slots[l], { line: true }), ' ', l),
          el('span', { class: 'k num', text: k.toFixed(5) }),
          el('span', { class: 'pm', text: `ρ ${pcmTxt(rho(k))} pcm · ${isF(s) ? `±${Math.round(s * 1e5)} pcm (1σ)` : 'no uncertainty estimate'}` }),
          isF(cv) ? el('span', { class: 'cmp', text: `model that never saw this run: ${cv.toFixed(5)} (${pcmTxt((cv - truth.y) * 1e5)} pcm)` }) : null);
      }));
    // agreement among the credible models only (CV error within 2x of the best); weak models are named, not counted
    const bestR = Math.min(...models.map(l => (M[l] && isF(M[l].rmse) ? M[l].rmse : Infinity)));
    const cred = res.filter(([l]) => !(M[l] && isF(M[l].rmse)) || M[l].rmse <= 2 * bestR);
    const weak = res.filter(r => !cred.includes(r)).map(([l]) => `${l} (${Math.round(M[l].rmse)} pcm)`);
    const ks = cred.map(r => r[1][0]), spread = (Math.max(...ks) - Math.min(...ks)) * 1e5;
    const weakTxt = weak.length ? ` Not counted: ${weak.join(', ')}, whose cross-validated error is more than twice the best.` : '';
    if (cred.length > 1) {
      const ok = noise === null || spread < noise;
      agree.textContent = (ok ? `The ${cred.length > 2 ? 'credible models' : 'models'} agree within ${Math.round(spread)} pcm${noise ? `, inside the OpenMC noise of the training data (≈${Math.round(noise)} pcm)` : ''}.`
        : `The ${cred.length > 2 ? 'credible models' : 'models'} disagree by ${Math.round(spread)} pcm, more than the OpenMC noise (≈${Math.round(noise)} pcm): this region is thinly sampled. Treat the prediction with care and consider running OpenMC here ("msrframe suggest" finds such points).`) + weakTxt;
      agree.className = 'note' + (ok ? '' : ' alert');
    } else agree.textContent = weakTxt.trim();
    // nearest training run, in the unit cube
    const u = space.toUnit(x);
    let best = -1, bd = Infinity;
    K.runs.U.forEach((ur, i) => { let d = 0; for (let j = 0; j < u.length; j++) d += (u[j] - ur[j]) ** 2; if (d < bd) { bd = d; best = i; } });
    const dist = Math.sqrt(bd / u.length), outside = u.some(v => v < -1e-9 || v > 1 + 1e-9);
    near.textContent = `Nearest OpenMC run: ${K.runs.ids[best]} (k = ${K.runs.y[best].toFixed(5)}), ${(100 * dist).toFixed(0)}% of the typical input range away.` +
      (outside ? ' Some inputs are outside the trained range: the models extrapolate.' : dist > 0.25 ? ' That is far: treat the predictions as extrapolation.' : '');
    drawSweep();
    drawSens();
  }
  const drawSweep = debounce(() => {
    const j = keys.indexOf(sweepKey), p = P[j], grid = p.log ? geomspace(p.low, p.high, 61) : linspace(p.low, p.high, 61);
    const series = models.map(l => {
      const pts = [], band = [];
      grid.forEach(g => { const xx = x.slice(); xx[j] = g; const [k, s] = pred(l, xx); pts.push([g, k]); if (l === stdModel && isF(s)) band.push([g, k - 2 * s, k + 2 * s]); });
      return { name: l, slot: slots[l], pts, band: band.length ? band : null };
    });
    series.push({ name: 'current value', slot: 0, pts: [[x[j], pred(models[0], x)[0]]], line: false, r: 5, noLegend: true, noHover: true });
    Charts.lineChart(sweepHost, { height: 270, xLabel: `${p.name || p.key}${unitTxt(p.unit) ? ` (${p.unit})` : ''}`, yLabel: 'k_eff', xLog: p.log, xName: p.name || p.key,
      series, hlines: [{ y: 1, label: 'critical', fit: false }], tipFmt: v => v.toFixed(5) });
  }, 50);
  function drawSens() {
    const rows = keys.map((k, j) => {
      const p = P[j], tr = v => (p.log ? Math.log10(v) : v), un = t => (p.log ? Math.pow(10, t) : t);
      const t0 = tr(x[j]), h = 0.01 * (tr(p.high) - tr(p.low));
      const a = clamp(t0 - h, tr(p.low), tr(p.high)), b = clamp(t0 + h, tr(p.low), tr(p.high));
      const cells = models.map(l => {
        const xa = x.slice(), xb = x.slice();
        xa[j] = un(a); xb[j] = un(b);
        const d = (rho(pred(l, xb)[0]) - rho(pred(l, xa)[0])) / (b - a);
        return fmt(d, 3);
      });
      const coeff = (K.derived_cfg.coefficients || []).includes(k);
      return [el('span', { style: coeff ? 'font-weight:700' : null, text: p.name || k }), p.log ? `pcm per decade` : `pcm per ${p.unit && p.unit !== '-' ? p.unit : 'unit'}`, ...cells];
    });
    sensHost.replaceChildren(table([{ t: 'input' }, { t: 'unit' }, ...models.map(l => ({ t: l, num: true }))], rows,
      { hl: i => (K.derived_cfg.coefficients || []).includes(keys[i]) }));
  }

  // ---- accuracy
  const parHost = el('div'), lcHost = el('div');
  let parModel = models[0];
  const parSel = el('select', { 'aria-label': 'model for the parity plot' }, models.filter(l => K.runs.cv[l]).map(l => el('option', { value: l, text: l })));
  parSel.addEventListener('change', () => { parModel = parSel.value; drawPar(); });
  root.append(el('h2', { text: 'How accurate is each model?', style: 'margin-top:8px' }));
  root.append(el('div', { class: 'grid2' },
    el('div', { class: 'panel' }, el('div', { class: 'phead' }, el('h3', { text: 'Predicted vs OpenMC (cross-validated)' }), parSel), parHost),
    panel('Learning curves', 'Hold-out error against the number of training runs: the slope shows what more OpenMC runs would buy.', lcHost)));
  const best = Math.min(...K.metrics.map(r => r.rmse));
  root.append(panel('Cross-validated accuracy', noise ? `OpenMC noise of the targets ≈ ${Math.round(noise)} pcm rms: part of every error is Monte Carlo noise in the training data, not model error.` : null,
    table([{ t: 'model' }, { t: 'RMSE (pcm)', num: true }, { t: 'MAE', num: true }, { t: 'max |error|', num: true }, { t: 'bias', num: true }, { t: 'R²', num: true },
      { t: 'noise-corrected', num: true }, { t: '95% interval covers', num: true }, { t: 'fit time (s)', num: true }],
    K.metrics.map(r => [keyName(r.model, slots[r.model] || 1), el('span', {}, Math.round(r.rmse).toLocaleString('en-US'), el('span', { class: 'cellbar', style: `width:${Math.round(60 * best / r.rmse)}px` })),
      Math.round(r.mae), Math.round(r.max_abs), pcmTxt(r.bias), fmt(r.r2, 5), isF(r.rmse_noise_corrected) ? Math.round(r.rmse_noise_corrected) : '–',
      isF(r.coverage95) ? `${Math.round(100 * r.coverage95)}%` : '–', fmt(r.fit_time_s, 2)]))));
  if (K.holdout && K.holdout.length) root.append(panel('Fixed hold-out (rows marked split = test)', 'Models trained on the training split only, as in the pilot study.',
    table([{ t: 'model' }, { t: 'RMSE (pcm)', num: true }, { t: 'R²', num: true }, { t: '95% interval covers', num: true }],
      K.holdout.map(r => [r.model, Math.round(r.rmse), fmt(r.r2, 5), isF(r.coverage95) ? `${Math.round(100 * r.coverage95)}%` : '–']))));
  function drawPar() {
    const cv = K.runs.cv[parModel];
    if (!cv) { parHost.replaceChildren(el('p', { class: 'muted', text: 'No cross-validated predictions stored for this model.' })); return; }
    Charts.scatterChart(parHost, { height: 300, diag: true, xLabel: 'OpenMC k_eff', yLabel: `${parModel} prediction`,
      points: K.runs.y.map((y, i) => ({ x: y, y: cv[i], slot: slots[parModel], label: K.runs.ids[i] })),
      tip: p => [el('div', { class: 'th', text: p.label }), el('div', { class: 'trow', text: `OpenMC ${p.x.toFixed(5)}` }),
        el('div', { class: 'trow', text: `${parModel} ${p.y.toFixed(5)} (${pcmTxt((p.y - p.x) * 1e5)} pcm)` })] });
  }
  function drawLC() {
    if (!K.learning.length) { lcHost.replaceChildren(el('p', { class: 'muted', text: 'No learning curves stored.' })); return; }
    Charts.lineChart(lcHost, { height: 280, xLabel: 'training runs', yLabel: 'hold-out RMSE (pcm)', xLog: true, yLog: true, xName: 'training runs',
      series: [...new Set(K.learning.map(r => r.model))].map(l => ({ name: l, slot: slots[l] || assignSlots([l])[l], markers: true,
        pts: K.learning.filter(r => r.model === l).map(r => [r.n_train, r.rmse]) })),
      tipFmt: v => Math.round(v).toLocaleString('en-US') + ' pcm' });
  }
  update();
  drawPar();
  drawLC();
  return () => { drawSweep(); drawPar(); drawLC(); };
}

// ================================================================ Stage 3: burnup predictor
function viewBurnup(root) {
  const B = DATA.burnup, space = MSR.makeSpace(B.space), keys = space.keys, P = B.space, T = B.times;
  const models = B.order, specs = B._specs, slots = B._slots;
  const M = Object.fromEntries(B.metrics.map(r => [r.model, r]));
  const x = space.nominal.slice();
  let years = T[T.length - 1], outSel = B.outputs.find(o => o !== 'keff') || B.outputs[0], loaded = null;
  const stdModel = models.find(l => B._has_std[l]);
  const firstOther = B.outputs.find(o => o !== 'keff');
  root.append(el('div', { class: 'lede' },
    el('div', { class: 'stack' }, el('h2', { text: 'Burnup history for any operating choice' }),
      el('p', { text: `Each training history is a ${fmt(T[T.length - 1], 3)}-year OpenMC depletion. ${B.n_histories} histories were run; accuracy is ${B.folds}-fold cross-validated over histories (each history predicted by models that never saw it) and the models shown are refit on all of them. Every model returns the whole history at once.` }),
      el('p', { class: 'hint', text: B.verdict })),
    el('div', { class: 'tiles' }, models.map(l => {
      const r = M[l] || {};
      return tile(l, isF(r.keff_rmse_pcm) ? Math.round(r.keff_rmse_pcm) : '–', 'pcm k_eff', `${longName(l)}${firstOther && isF(r[firstOther + '_rel_rmse_%']) ? ` · ${B.labels[firstOther] || firstOther} ${fmt(r[firstOther + '_rel_rmse_%'], 2)}%` : ''}`, slots[l]);
    }))));
  const yearsCtl = slider({ key: 'horizon', name: 'Horizon', unit: 'years', low: T[1] || T[0], high: T[T.length - 1], log: false }, years, v => { years = v; draw(); });
  const sliders = keys.map((k, i) => slider(P[i], x[i], v => { x[i] = v; loaded = null; draw(); }));
  const histSel = el('select', { 'aria-label': 'load an OpenMC history' }, el('option', { value: '', text: 'Load an OpenMC history…' }),
    B.histories.ids.map((id, i) => el('option', { value: i, text: id })));
  histSel.addEventListener('change', () => {
    if (histSel.value === '') return;
    loaded = +histSel.value;
    B.histories.X[loaded].forEach((v, j) => { x[j] = v; sliders[j].setValue(v); });
    draw();
  });
  const ctrls = el('div', { class: 'panel' }, el('h3', { text: 'Operating inputs' }), yearsCtl, ...sliders,
    el('div', { class: 'btnrow', style: 'margin-top:10px' },
      el('button', { class: 'btn', type: 'button', text: 'Reset to nominal', onclick: () => { keys.forEach((k, i) => { x[i] = P[i].nominal; sliders[i].setValue(x[i]); }); loaded = null; histSel.value = ''; draw(); } }),
      histSel));
  const kHost = el('div'), oHost = el('div'), atHost = el('div');
  const outOpts = B.outputs.filter(o => o !== 'keff').map(o => ({ value: o, label: B.labels[o] || o }));
  const seg = outOpts.length ? segmented(outOpts, outSel, v => { outSel = v; draw(); }) : null;
  root.append(el('div', { class: 'work' }, ctrls, el('div', { class: 'stack' },
    panel('k_eff over the horizon', `shaded: ${stdModel || 'model'} ±2σ (uncertainty about the mean)${B.histories.ids.length ? ' · load an OpenMC history to compare' : ''}`, kHost),
    seg ? el('div', { class: 'panel' }, el('div', { class: 'phead' }, el('h3', { text: 'Inventories and composition' })), el('div', { style: 'overflow-x:auto;margin-bottom:6px' }, seg), oHost) : null,
    panel('Values at the end of the horizon', null, atHost))));
  const draw = debounce(() => {
    const u = space.toUnit(x), idx = T.map((t, i) => i).filter(i => T[i] <= years + 1e-9);
    const preds = Object.fromEntries(models.map(l => [l, MSR.predictTrajectory(specs[l], u)]));
    const truth = loaded !== null ? B.histories : null;
    const mk = (o, isK) => {
      const series = models.map(l => ({ name: l, slot: slots[l], pts: idx.map(i => [T[i], preds[l][o].mu[i]]),
        band: l === stdModel ? idx.map(i => [T[i], preds[l][o].mu[i] - 2 * preds[l][o].sd[i], preds[l][o].mu[i] + 2 * preds[l][o].sd[i]]) : null }));
      if (truth) {
        series.push({ name: `OpenMC ${truth.ids[loaded]}`, slot: 0, width: 2.6, markers: true, r: 3, pts: idx.map(i => [T[i], truth.Y[o][loaded][i]]) });
        const cvl = B.cv && B.cv[models[0]];
        if (cvl && cvl[o]) series.push({ name: `${models[0]} never trained on it`, slot: slots[models[0]], dash: '1.5 3', width: 1.6, pts: idx.map(i => [T[i], cvl[o][loaded][i]]) });
      }
      return series;
    };
    Charts.lineChart(kHost, { height: 260, xLabel: 'time (years)', yLabel: 'k_eff', xName: 'year', xDomain: [0, years], series: mk('keff', true),
      hlines: [{ y: 1, label: 'critical', fit: false }], tipFmt: v => v.toFixed(5) });
    if (seg) Charts.lineChart(oHost, { height: 240, xLabel: 'time (years)', yLabel: `${B.labels[outSel] || outSel}${B.units[outSel] ? ` (${B.units[outSel]})` : ''}`,
      xName: 'year', xDomain: [0, years], series: mk(outSel), tipFmt: v => fmt(v, 5) });
    const ie = idx[idx.length - 1];
    const rows = B.outputs.map(o => [`${B.labels[o] || o}${B.units[o] ? ` (${B.units[o]})` : ''}`,
      ...(truth ? [o === 'keff' ? truth.Y[o][loaded][ie].toFixed(5) : fmt(truth.Y[o][loaded][ie], 5)] : []),
      ...models.map(l => (o === 'keff' ? preds[l][o].mu[ie].toFixed(5) : fmt(preds[l][o].mu[ie], 5)))]);
    atHost.replaceChildren(table([{ t: `at t = ${fmt(T[ie], 3)} y` }, ...(truth ? [{ t: 'OpenMC', num: true }] : []), ...models.map(l => ({ t: l, num: true }))], rows));
  }, 40);
  // ---- accuracy
  const tHost = el('div');
  root.append(el('h2', { text: 'How accurate is each model?', style: 'margin-top:8px' }));
  const relCols = B.outputs.filter(o => o !== 'keff');
  root.append(panel('k_eff error along the history', `${B.folds}-fold cross-validation over histories: the error at each time point of a history the model never saw.`, tHost));
  root.append(panel('Cross-validated accuracy', 'k_eff in pcm; inventories as relative RMSE (%).',
    table([{ t: 'model' }, { t: 'k_eff RMSE (pcm)', num: true }, { t: '95% interval covers', num: true }, ...relCols.map(o => ({ t: B.labels[o] || o, num: true }))],
      B.metrics.map(r => [keyName(r.model, slots[r.model] || 1), Math.round(r.keff_rmse_pcm), isF(r.keff_coverage95) ? `${Math.round(100 * r.keff_coverage95)}%` : '–',
        ...relCols.map(o => (isF(r[o + '_rel_rmse_%']) ? fmt(r[o + '_rel_rmse_%'], 3) + '%' : '–'))]))));
  function drawT() {
    const labs = [...new Set(B.rmse_vs_time.map(r => r.model))];
    Charts.lineChart(tHost, { height: 260, xLabel: 'time (years)', yLabel: 'k_eff RMSE (pcm)', xName: 'year',
      series: labs.map(l => ({ name: l, slot: slots[l] || assignSlots([l])[l], markers: true, r: 2.8, pts: B.rmse_vs_time.filter(r => r.model === l).map(r => [r.time_y, r.rmse_pcm]) })),
      tipFmt: v => Math.round(v) + ' pcm' });
  }
  draw();
  drawT();
  return () => { draw(); drawT(); };
}

// ================================================================ Stage 4: fuel-cycle optimiser
const W_BAND = 20, W_HN = 1e4, W_FIS = 10;            // same safety-first weights as msrframe/stages/optimize.py
function viewOptimize(root) {
  const B = DATA.burnup, O = DATA.optimize, space = MSR.makeSpace(B.space), keys = space.keys, P = B.space, T = B.times;
  const models = B.order, specs = B._specs, slots = B._slots;
  const has = o => B.outputs.includes(o);
  const scenarios = O.scenarios || [];
  // ---- form
  const presetSel = el('select', { 'aria-label': 'scenario preset' }, scenarios.map((s, i) => el('option', { value: i, text: s.title || s.name })), el('option', { value: 'custom', text: 'Custom' }));
  const fT = numField('Target reactivity', 0, 'pcm', 'ρ = (k − 1)/k to hold; 0 = exactly critical.');
  const fTol = numField('Allowed deviation', 400, 'pcm', 'Half-width of the band; the temperature limits below can narrow it.');
  const fT0 = numField('Hold from year', 1, 'y', 'Start of the checked window (year 1 skips the ²³³Pa/Xe/Sm start-up transient).');
  const fH = numField('Horizon', Math.min(18, T[T.length - 1]), 'y', 'End of the window; controls are held constant until then.');
  const surr = el('select', { 'aria-label': 'burnup surrogate' }, models.map(l => el('option', { value: l, text: l })), models.length > 1 ? el('option', { value: 'ensemble', text: 'ensemble of all shown' }) : null);
  surr.value = O.surrogate && models.includes(O.surrogate) ? O.surrogate : models[0];
  const fK = numField('Uncertainty multiplier k', 1, '× σ', 'The band must hold for prediction ± k·σ (σ: the surrogate’s uncertainty about the mean).');
  const fA = numField('Temperature coefficient α_T', -6.9, 'pcm/K', 'Turns a reactivity excess into the salt-temperature change that would absorb it.');
  const fUp = numField('Allowed salt heat-up', 50, 'K', 'Upper band edge ≤ |α_T| × this.');
  const fDn = numField('Allowed salt cool-down', 50, 'K', 'Lower band edge ≥ −|α_T| × this.');
  const fHN = has('HN_molpct') ? numField('Heavy-nuclide limit', 25, 'mol%', 'ThF₄ + UF₄ solubility cap; checked on the mean prediction.') : null;
  const fF = has('fissile_total_kg') ? numField('Fissile inventory cap', 7000, 'kg', '²³³U + ²³³Pa in fuel and blanket; mean prediction.') : null;
  const fW = keys.includes('u233_feed_kgd') ? numField('Reward for ²³³U extraction', 0, '(0 = off)', 'Objective weight, not a limit: favours extracting bred ²³³U.') : null;
  const fG = numField('Generations', 120, '', 'Differential-evolution generations (population 40–90).', 10);
  const ctl = {};
  const ctlRows = keys.map((k, i) => {
    const p = P[i];
    const st = ctl[k] = { mode: 'fix', lo: p.low, hi: p.high, value: p.nominal };
    const lo = el('input', { type: 'number', step: 'any', 'aria-label': `${k} minimum` }), hi = el('input', { type: 'number', step: 'any', 'aria-label': `${k} maximum` });
    const fx = el('input', { type: 'number', step: 'any', 'aria-label': `${k} fixed value` });
    const bounds = el('div', { class: 'bounds' });
    const render = () => {
      lo.value = fmt(st.lo, 6).replace(/,/g, ''); hi.value = fmt(st.hi, 6).replace(/,/g, ''); fx.value = fmt(st.value, 6).replace(/,/g, '');
      bounds.replaceChildren(...(st.mode === 'opt' ? ['between ', lo, ' and ', hi, unitTxt(p.unit)] : st.mode === 'fix' ? ['fixed at ', fx, unitTxt(p.unit)]
        : [`slowest removal the surrogate knows: ${fmt(p.high, 3)}${unitTxt(p.unit)}`]));
    };
    [lo, hi, fx].forEach(inp => inp.addEventListener('change', () => { st.lo = +lo.value; st.hi = +hi.value; st.value = +fx.value; }));
    const opts = [{ value: 'opt', label: 'Optimise' }, { value: 'fix', label: 'Fixed' }].concat(p.log ? [{ value: 'off', label: 'Off' }] : []);
    const seg = segmented(opts, st.mode, v => { st.mode = v; render(); });
    const row = el('div', { class: 'cfield' }, el('div', { class: 'top' }, el('span', {}, p.name || k, ' ', el('span', { class: 'hint', style: 'display:inline', text: p.name !== k ? k : '' })), seg), bounds);
    row.sync = () => { seg.set(st.mode); render(); };
    render();
    return row;
  });
  function loadPreset(i) {
    const s = scenarios[i];
    if (!s) return;
    fT.input.value = s.rho_target_pcm ?? 0; fTol.input.value = s.rho_tol_pcm ?? 400; fT0.input.value = s.eval_start_y ?? 0;
    fH.input.value = Math.min(s.horizon_y ?? T[T.length - 1], T[T.length - 1]); fK.input.value = s.uncertainty_k ?? 1;
    const lim = s.limits || {};
    fA.input.value = lim.alpha_T_pcm_per_K ?? ''; fUp.input.value = lim.dT_up_K ?? ''; fDn.input.value = lim.dT_down_K ?? '';
    if (fHN) fHN.input.value = lim.hn_max_molpct ?? ''; if (fF) fF.input.value = lim.fissile_max_kg ?? '';
    if (fW) fW.input.value = (s.weights || {}).u233_extraction ?? 0;
    keys.forEach((k, j) => {
      const p = P[j], v = (s.controls || {})[k], st = ctl[k];
      if (v === 'off') st.mode = 'off';
      else if (Array.isArray(v)) { st.mode = 'opt'; st.lo = Math.max(v[0], p.low); st.hi = Math.min(v[1], p.high); }
      else { st.mode = 'fix'; st.value = v === undefined || v === null ? p.nominal : clamp(+v, p.low, p.high); }
      ctlRows[j].sync();
    });
    pyBtn.hidden = !pyResult(s.name);
  }
  const pyResult = name => (O.python_results || []).find(r => r.scenario === name);
  presetSel.addEventListener('change', () => { if (presetSel.value !== 'custom') loadPreset(+presetSel.value); else pyBtn.hidden = true; });
  const go = el('button', { class: 'btn primary', type: 'button', text: 'Find controls' });
  const pyBtn = el('button', { class: 'btn', type: 'button', text: 'Show the Python optimum', hidden: true });
  const prog = el('div', { class: 'progress' }, el('i'));
  const form = el('div', { class: 'panel' }, el('h3', { text: 'Plan' }),
    el('div', { class: 'cfield' }, el('label', { class: 'name', text: 'Scenario (from the config)' }), presetSel),
    fT, fTol, fT0, fH, el('div', { class: 'field' }, el('label', { class: 'name', text: 'Burnup model' }), surr), fK,
    el('div', { class: 'section-label label', text: 'Controls' }),
    el('p', { class: 'hint', text: 'Optimise: the controller chooses a value in the range. Fixed: you set it. Off: removal disabled (cycle set to the slowest value in the training data).' }),
    ...ctlRows,
    el('div', { class: 'section-label label', text: 'Safety limits' }), fA, fUp, fDn, fHN, fF, fW,
    el('div', { class: 'section-label label', text: 'Search' }), fG,
    el('div', { class: 'btnrow', style: 'margin-top:12px' }, go, pyBtn), prog);
  const out = el('div', { class: 'stack' }, panel('Result', null, el('p', { class: 'muted', text: 'Choose a scenario (or edit the plan) and press Find controls. The search runs in your browser on the burnup surrogate.' })));
  root.append(el('div', { class: 'lede' }, el('div', { class: 'stack' }, el('h2', { text: 'Fuel-cycle controller' }),
    el('p', { text: `Finds start-up loading, feeds and removal cycles that keep the reactivity inside a band over a planning window, using the ${models.length > 1 ? 'chosen' : ''} burnup surrogate. The objective is the one in msrframe/stages/optimize.py: RMS tracking error plus heavy penalties for leaving the band at ±kσ or exceeding the heavy-nuclide and fissile limits.` })),
    el('div', { class: 'tiles' }, tile('scenarios in config', scenarios.length, '', scenarios.map(s => s.name).join(' · ')),
      tile('burnup models', models.length, '', models.join(' · ')))));
  root.append(el('div', { class: 'work' }, form, out));

  function buildReq() {
    const num = (f, d) => (f && f.num() !== null ? f.num() : d);
    return {
      rho_target_pcm: num(fT, 0), rho_tol_pcm: num(fTol, 400), eval_start_y: num(fT0, 0), horizon_y: Math.min(num(fH, T[T.length - 1]), T[T.length - 1]),
      uncertainty_k: num(fK, 1), surrogate: surr.value, generations: Math.max(5, Math.round(num(fG, 120))),
      limits: { alpha_T_pcm_per_K: num(fA, 1e9), dT_up_K: num(fUp, 1e9), dT_down_K: num(fDn, 1e9), hn_max_molpct: num(fHN, Infinity), fissile_max_kg: num(fF, Infinity) },
      weights: { u233_extraction: num(fW, 0) },
      controls: Object.fromEntries(keys.map(k => [k, { ...ctl[k] }])),
    };
  }
  function decode(req) {
    const free = [], fixed = {};
    keys.forEach((k, j) => {
      const p = P[j], c = req.controls[k];
      if (c.mode === 'off') fixed[k] = p.high;
      else if (c.mode === 'opt') free.push({ k, lo: Math.max(Math.min(c.lo, c.hi), p.low), hi: Math.min(Math.max(c.lo, c.hi), p.high), log: p.log });
      else fixed[k] = clamp(c.value, p.low, p.high);
    });
    return { free, fixed };
  }
  function paramsFrom(u, dec) {
    const p = Object.assign({}, dec.fixed);
    dec.free.forEach((f, i) => { p[f.k] = f.log ? Math.pow(10, Math.log10(f.lo) + u[i] * (Math.log10(f.hi) - Math.log10(f.lo))) : f.lo + u[i] * (f.hi - f.lo); });
    return p;
  }
  const OUTS = ['keff', 'HN_molpct', 'fissile_total_kg'].filter(has);
  function predictPlan(req, p, outs = OUTS) {
    const u = space.toUnit(keys.map(k => p[k]));
    return req.surrogate === 'ensemble' ? MSR.predictTrajectoryEnsemble(models.map(l => specs[l]), u, false, outs) : MSR.predictTrajectory(specs[req.surrogate], u, false, outs);
  }
  function evaluate(req, pred) {
    const tEnd = T.find(t => t >= req.horizon_y - 1e-9) ?? T[T.length - 1], lim = req.limits, aT = Math.abs(lim.alpha_T_pcm_per_K);
    const lo = Math.max(req.rho_target_pcm - req.rho_tol_pcm, -aT * lim.dT_down_K), hi = Math.min(req.rho_target_pcm + req.rho_tol_pcm, aT * lim.dT_up_K);
    let n = 0, se = 0, v2 = 0, viol = 0, meanIn = true, hn = -Infinity, fis = -Infinity, up = -Infinity, dn = Infinity, smax = 0;
    for (let i = 0; i < T.length; i++) {
      if (T[i] < req.eval_start_y - 1e-9 || T[i] > tEnd + 1e-9) continue;
      const k = clamp(pred.keff.mu[i], 0.5, 1.5), r = rho(k), sd = isF(pred.keff.sd[i]) ? pred.keff.sd[i] : 0, s = sd / (k * k) * 1e5;
      const a = Math.max(0, r + req.uncertainty_k * s - hi), b = Math.max(0, lo - (r - req.uncertainty_k * s));
      se += (r - req.rho_target_pcm) ** 2; v2 += a * a + b * b; n++;
      viol = Math.max(viol, a, b); smax = Math.max(smax, s);
      up = Math.max(up, r + req.uncertainty_k * s); dn = Math.min(dn, r - req.uncertainty_k * s);
      if (r < lo || r > hi) meanIn = false;
      if (pred.HN_molpct) hn = Math.max(hn, pred.HN_molpct.mu[i]);
      if (pred.fissile_total_kg) fis = Math.max(fis, pred.fissile_total_kg.mu[i]);
    }
    const track = Math.sqrt(se / Math.max(n, 1));
    let pen = W_BAND * Math.sqrt(v2 / Math.max(n, 1));
    if (isF(hn) && isF(lim.hn_max_molpct)) pen += W_HN * Math.max(0, hn - lim.hn_max_molpct);
    if (isF(fis) && isF(lim.fissile_max_kg)) pen += W_FIS * Math.max(0, fis - lim.fissile_max_kg);
    const status = viol <= 1e-6 ? 'feasible' : meanIn ? `mean inside band, not certified at ${req.uncertainty_k} sigma` : 'infeasible';
    return { cost: track + pen, track, pen, band: [lo, hi], viol, meanIn, status, hn, fis, smax, tEnd, dTup: Math.max(up, 0) / aT, dTdn: Math.max(-dn, 0) / aT };
  }
  function cost(req, p) {
    const o = evaluate(req, predictPlan(req, p)), f = p.u233_feed_kgd;
    return o.cost - (isF(f) ? req.weights.u233_extraction * Math.max(-f, 0) * 100 : 0);
  }
  function run(req, onProgress, onDone) {
    const dec = decode(req), D = dec.free.length;
    if (!D) { onDone({ p: paramsFrom([], dec), dec, evals: 1 }); return; }
    const NP = Math.min(90, Math.max(40, 10 * D)), G = req.generations;
    let seed = 12345;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
    let pop = Array.from({ length: NP }, () => Array.from({ length: D }, rnd));
    let fit = pop.map(u => cost(req, paramsFrom(u, dec))), g = 0, evals = NP;
    const step = () => {
      const until = performance.now() + 30;
      while (g < G && performance.now() < until) {
        const F = 0.5 + 0.5 * rnd(), bi = fit.indexOf(Math.min(...fit));
        for (let i = 0; i < NP; i++) {
          let a, b, c;
          do { a = Math.floor(rnd() * NP); } while (a === i);
          do { b = Math.floor(rnd() * NP); } while (b === i || b === a);
          do { c = Math.floor(rnd() * NP); } while (c === i || c === a || c === b);
          const jr = Math.floor(rnd() * D), trial = pop[i].slice();
          for (let j = 0; j < D; j++) if (rnd() < 0.7 || j === jr) {
            let v = (rnd() < 0.5 ? pop[bi][j] : pop[a][j]) + F * (pop[b][j] - pop[c][j]);
            if (v < 0) v = rnd() * pop[i][j];
            if (v > 1) v = pop[i][j] + rnd() * (1 - pop[i][j]);
            trial[j] = v;
          }
          const ft = cost(req, paramsFrom(trial, dec));
          evals++;
          if (ft <= fit[i]) { pop[i] = trial; fit[i] = ft; }
        }
        g++;
      }
      onProgress(g / G);
      if (g < G) setTimeout(step, 0);
      else { const bi = fit.indexOf(Math.min(...fit)); onDone({ p: paramsFrom(pop[bi], dec), dec, evals, cost: fit[bi] }); }
    };
    setTimeout(step, 0);
  }
  function showResult(req, res, source) {
    const p = res.p, outs = ['keff', 'HN_molpct', 'fissile_total_kg', 'U233_fuel_kg', 'FP_fuel_kg'].filter(has);
    const pred = predictPlan(req, p, outs), o = evaluate(req, pred);
    const idx = T.map((t, i) => i).filter(i => T[i] <= o.tEnd + 1e-9);
    const pass = o.viol <= 1e-6;
    const chip = (ok, text) => el('span', { class: 'chip ' + (ok ? 'pass' : 'fail') }, el('span', { class: 'dot' }), (ok ? 'Pass · ' : 'Fail · ') + text);
    const freeKeys = new Set(res.dec.free.map(f => f.k));
    const rows = keys.map((k, j) => [P[j].name || k, `${fmt(p[k], 4)}${unitTxt(P[j].unit)}`, freeKeys.has(k) ? 'chosen by the controller' : req.controls[k].mode === 'off' ? 'removal off' : 'fixed']);
    const rhoHost = el('div'), limHost = el('div'), invHost = el('div');
    const lab = req.surrogate === 'ensemble' ? 'ensemble' : req.surrogate, slot = req.surrogate === 'ensemble' ? 0 : slots[req.surrogate];
    const verdict = pass ? `Feasible: the ${req.uncertainty_k}σ envelope stays inside ${Math.round(o.band[0])} … ${Math.round(o.band[1])} pcm at every time point from year ${req.eval_start_y} to year ${fmt(o.tEnd, 3)} (largest σ ${Math.round(o.smax)} pcm).`
      : `${o.status[0].toUpperCase()}${o.status.slice(1)}: the ${req.uncertainty_k}σ envelope leaves the band by up to ${Math.round(o.viol)} pcm. Absorbing that with temperature feedback would need +${Math.round(o.dTup)} K / −${Math.round(o.dTdn)} K of salt temperature. Try a shorter horizon, a wider band, or more OpenMC histories to shrink σ.`;
    const csv = toCSV(['parameter', 'value', 'unit', 'how'], keys.map((k, j) => ({ parameter: k, value: p[k], unit: P[j].unit, how: rows[j][2] })));
    out.replaceChildren(...[
      el('div', { class: 'panel' }, el('h3', { text: source === 'python' ? 'Python optimum (re-evaluated here)' : 'Controller decision' }),
        el('p', { style: 'margin-bottom:8px', text: `RMS deviation from the target ${Math.round(o.track)} pcm${res.evals ? ` (${res.evals.toLocaleString('en-US')} surrogate histories evaluated with ${lab})` : ` with ${lab}`}.` }),
        el('p', { class: 'note' + (pass ? '' : ' alert'), style: 'margin-bottom:10px', text: verdict }),
        el('div', { class: 'chips', style: 'margin-bottom:10px' }, chip(pass, `reactivity band at ${req.uncertainty_k}σ`),
          isF(o.hn) && isF(req.limits.hn_max_molpct) ? chip(o.hn <= req.limits.hn_max_molpct, `HN ≤ ${req.limits.hn_max_molpct} mol% (max ${o.hn.toFixed(2)})`) : null,
          isF(o.fis) && isF(req.limits.fissile_max_kg) ? chip(o.fis <= req.limits.fissile_max_kg, `fissile ≤ ${req.limits.fissile_max_kg} kg (max ${Math.round(o.fis)})`) : null),
        table([{ t: 'control' }, { t: 'value', num: true }, { t: '' }], rows),
        el('div', { class: 'btnrow', style: 'margin-top:10px' }, el('button', { class: 'btn', type: 'button', text: 'Download controls (CSV) for OpenMC verification', onclick: () => download('controls_for_openmc.csv', csv) }))),
      panel('Reactivity under these controls', `grey band = allowed range; shaded = ${lab} ±${req.uncertainty_k}σ; dashed = target`, rhoHost),
      (pred.HN_molpct || pred.fissile_total_kg) ? panel('Limits', null, limHost) : null,
      (pred.U233_fuel_kg || pred.FP_fuel_kg) ? panel('²³³U and fission products in the fuel salt', null, invHost) : null].filter(Boolean));
    const k = i => clamp(pred.keff.mu[i], 0.5, 1.5);
    Charts.lineChart(rhoHost, { height: 270, xLabel: 'time (years)', yLabel: 'reactivity (pcm)', xName: 'year', xDomain: [0, o.tEnd],
      hbands: [{ y0: o.band[0], y1: o.band[1], label: 'allowed' }], hlines: [{ y: req.rho_target_pcm, label: 'target' }],
      vbands: req.eval_start_y > 0 ? [{ x0: 0, x1: Math.min(req.eval_start_y, o.tEnd), label: 'not checked' }] : [],
      yDomain: (() => { const v = idx.filter(i => T[i] >= req.eval_start_y - 1e-9).map(i => rho(k(i))); return [Math.min(o.band[0] - 300, ...v.map(a => a - 300)), Math.max(o.band[1] + 300, ...v.map(a => a + 300))]; })(),
      series: [{ name: lab, slot, markers: true, r: 3, pts: idx.map(i => [T[i], rho(k(i))]),
        band: idx.map(i => { const s = (isF(pred.keff.sd[i]) ? pred.keff.sd[i] : 0) / (k(i) * k(i)) * 1e5; return [T[i], rho(k(i)) - req.uncertainty_k * s, rho(k(i)) + req.uncertainty_k * s]; }) }],
      tipFmt: v => Math.round(v).toLocaleString('en-US') + ' pcm' });
    if (pred.HN_molpct || pred.fissile_total_kg) {
      const host2 = el('div'), host3 = el('div');
      limHost.replaceChildren(el('div', { class: 'grid2' }, host2, host3));
      if (pred.HN_molpct) Charts.lineChart(host2, { height: 220, xLabel: 'time (years)', yLabel: 'heavy nuclides (mol%)', xName: 'year', xDomain: [0, o.tEnd],
        series: [{ name: 'HN', slot: 2, pts: idx.map(i => [T[i], pred.HN_molpct.mu[i]]), noLegend: true }],
        hlines: isF(req.limits.hn_max_molpct) ? [{ y: req.limits.hn_max_molpct, label: 'limit' }] : [], tipFmt: v => v.toFixed(3) + ' mol%' });
      if (pred.fissile_total_kg) Charts.lineChart(host3, { height: 220, xLabel: 'time (years)', yLabel: 'fissile inventory (kg)', xName: 'year', xDomain: [0, o.tEnd],
        series: [{ name: 'fissile', slot: 3, pts: idx.map(i => [T[i], pred.fissile_total_kg.mu[i]]), noLegend: true }],
        hlines: isF(req.limits.fissile_max_kg) ? [{ y: req.limits.fissile_max_kg, label: 'cap' }] : [], tipFmt: v => Math.round(v) + ' kg' });
    }
    if (pred.U233_fuel_kg || pred.FP_fuel_kg) Charts.lineChart(invHost, { height: 220, xLabel: 'time (years)', yLabel: 'kg', xName: 'year', xDomain: [0, o.tEnd],
      series: [pred.U233_fuel_kg ? { name: B.labels.U233_fuel_kg || '233U in fuel salt', slot: 1, pts: idx.map(i => [T[i], pred.U233_fuel_kg.mu[i]]) } : null,
        pred.FP_fuel_kg ? { name: B.labels.FP_fuel_kg || 'fission products', slot: 2, pts: idx.map(i => [T[i], pred.FP_fuel_kg.mu[i]]) } : null].filter(Boolean),
      tipFmt: v => Math.round(v).toLocaleString('en-US') + ' kg' });
  }
  go.addEventListener('click', () => {
    const req = buildReq();
    go.disabled = true; go.textContent = 'Searching…';
    run(req, f => { prog.firstChild.style.width = (100 * f).toFixed(0) + '%'; }, res => {
      go.disabled = false; go.textContent = 'Find controls';
      showResult(req, res, 'js');
      out.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
  pyBtn.addEventListener('click', () => {
    const s = scenarios[+presetSel.value], r = s && pyResult(s.name);
    if (!r) return;
    const req = buildReq();
    if (r.surrogate && models.includes(r.surrogate)) { req.surrogate = r.surrogate; surr.value = r.surrogate; }
    const dec = decode(req);
    showResult(req, { p: Object.assign({}, dec.fixed, r.controls), dec, evals: 0 }, 'python');
  });
  if (scenarios.length) loadPreset(0);
  return () => {};
}

// ================================================================ Data & training
function viewData(root) {
  const D = DATA.data;
  const counts = D.counts || {};
  root.append(el('div', { class: 'lede' },
    el('div', { class: 'stack' }, el('h2', { text: 'Your data make the models more accurate' }),
      el('p', { text: 'Every number in this dashboard comes from the CSV files below. Add OpenMC runs, re-train, and the dashboard is rebuilt with the new models; the history charts record how the error fell as the database grew.' })),
    el('div', { class: 'tiles' }, Object.entries(counts).map(([k, v]) => tile(k, v.toLocaleString('en-US'), '', null)))));
  root.append(panel('Training data files', 'Files matched by the data: patterns of each stage in the config. Files under data/user/ are runs you added.',
    table([{ t: 'stage' }, { t: 'file' }, { t: 'rows', num: true }, { t: 'kind' }], D.files.map(f => [f.stage, f.file, f.rows.toLocaleString('en-US'), f.kind]),
      { hl: i => D.files[i].kind === 'added by you' })));
  const hk = el('div'), hb = el('div');
  const hist = (host, rows, ycol, label) => {
    const labs = [...new Set(rows.map(r => r.model))], ns = [...new Set(rows.map(r => r.n_data))];
    if (ns.length < 2) {
      host.replaceChildren(...[el('p', { class: 'note', text: `One training session so far (${ns[0] ?? 0} ${label}). Each re-training after adding data adds a point here.` }),
        rows.length ? table([{ t: 'model' }, { t: 'error', num: true }, { t: 'when' }], rows.slice(-labs.length).map(r => [r.model, `${Math.round(r[ycol])} pcm`, (r.timestamp || '').replace('T', ' ')])) : null].filter(Boolean));
      return;
    }
    const slotsH = assignSlots(labs);
    Charts.lineChart(host, { height: 250, xLabel: `${label} in the database`, yLabel: 'CV RMSE (pcm)', xName: label,
      series: labs.map(l => {
        const by = {};
        rows.filter(r => r.model === l).forEach(r => { by[r.n_data] = r[ycol]; });
        return { name: l, slot: slotsH[l], markers: true, pts: Object.entries(by).map(([n, v]) => [+n, v]).sort((a, b) => a[0] - b[0]) };
      }), tipFmt: v => Math.round(v) + ' pcm' });
  };
  const hasK = (D.history.keff || []).length > 0, hasB = (D.history.burnup || []).length > 0;
  const hp = [hasK ? panel('k_eff accuracy as the data grew', 'From results/history/keff_history.csv (one point per training session).', hk) : null,
    hasB ? panel('Burnup accuracy as the data grew', 'From results/history/burnup_history.csv.', hb) : null].filter(Boolean);
  if (hp.length) root.append(hp.length === 2 ? el('div', { class: 'grid2' }, ...hp) : hp[0]);
  if (D.active_learning && D.active_learning.length) {
    const ah = el('div');
    root.append(panel('Active learning: runs chosen where the models are least certain vs at random', 'RMSE on an independent test set after each batch of new runs (msrframe loop).', ah));
    const labs = [...new Set(D.active_learning.map(r => r.model))], sl = assignSlots(labs);
    Charts.lineChart(ah, { height: 260, xLabel: 'training runs', yLabel: 'test RMSE (pcm)', yLog: true, xName: 'runs',
      series: labs.flatMap(l => ['active', 'random'].map(arm => ({ name: `${l} · ${arm}`, slot: sl[l], dash: arm === 'active' ? '' : '2 3', markers: arm === 'active',
        pts: D.active_learning.filter(r => r.model === l && r.arm === arm).map(r => [r.n, r.rmse_pcm]) }))).filter(s => s.pts.length),
      tipFmt: v => Math.round(v) + ' pcm' });
  }
  Object.entries(D.suggested || {}).forEach(([stage, rows]) => {
    if (!rows || !rows.length) return;
    const cols = Object.keys(rows[0]);
    root.append(el('div', { class: 'panel' }, el('div', { class: 'phead' }, el('h3', { text: `Suggested next OpenMC runs (${stage})` }),
      el('button', { class: 'btn', type: 'button', text: 'Download CSV', onclick: () => download(`suggested_runs_${stage}.csv`, toCSV(cols, rows)) })),
      el('p', { class: 'psub', text: 'Points where the models are least certain or disagree most, spread out over the design space (msrframe suggest). Run them, fill the output column and add them.' }),
      table(cols.map(c => ({ t: c, num: c !== 'case_id' })), rows.map(r => cols.map(c => (typeof r[c] === 'number' ? fmt(r[c], 5) : (r[c] ?? '')))), { scroll: true })));
  });
  const cfgPath = D.config || 'config/msfr_default.yaml';
  root.append(el('div', { class: 'panel guide' }, el('h3', { text: 'How to add your own data' }),
    el('ol', {},
      el('li', {}, 'Run OpenMC at the design points you need (or at the suggested ones):', el('pre', { text: `python -m msrframe suggest -c ${cfgPath} --stage keff -n 20` })),
      el('li', {}, 'Add the results. Columns: the stage inputs plus keff (and keff_std); burnup files also need case_id and time_y.', el('pre', { text: `python -m msrframe add-data -c ${cfgPath} --stage keff --file my_runs.csv --retrain` })),
      el('li', {}, 'Rebuild this page (it is also rebuilt by every msrframe run):', el('pre', { text: `python -m msrframe dashboard -c ${cfgPath}` })),
      el('li', {}, 'Swap a method by editing one word in the config (e.g. morris → random_forest, or keff models: [gp, random_forest, svr]) and re-run.'))));
  root.append(panel('Models in this dashboard', 'Exact: the browser reproduces the Python model to round-off. Distilled: an unknown plug-in re-learned by a small student model; its error is shown.',
    table([{ t: 'stage' }, { t: 'model' }, { t: 'export' }, { t: 'max deviation', num: true }, { t: 'size (KB)', num: true }],
      D.models.map(m => [m.stage, m.label, m.method, m.max_dev || '–', fmt(m.size_kb, 3)]))));
  const draw = () => { if (hasK) hist(hk, D.history.keff, 'cv_rmse_pcm', 'runs'); if (hasB) hist(hb, D.history.burnup, 'keff_rmse_pcm', 'histories'); };
  draw();
  return draw;
}

// ================================================================ boot
(function boot() {
  const meta = DATA.meta;
  $('#title').textContent = meta.project;
  document.title = meta.project;
  $('#sub').textContent = meta.subtitle;
  $('#refcard').replaceChildren(...meta.refcard.map(([k, v]) => el('span', {}, k + ' ', el('b', { text: v }))));
  // decode models once and fix colour slots (colour follows the learner across tabs)
  if (DATA.keff) { DATA.keff._specs = MSR.decode(DATA.keff.models); DATA.keff._slots = assignSlots([...new Set(DATA.keff.order.concat(DATA.keff.ranking))]); }
  if (DATA.burnup) {
    DATA.burnup._specs = MSR.decode(DATA.burnup.models);
    DATA.burnup._slots = assignSlots([...new Set(DATA.burnup.order.concat(DATA.burnup.ranking))]);
    const hasStd = spec => Object.values(spec.outputs || {}).every(o => (Array.isArray(o) ? o.every(m => m.std) : o.models ? o.models.every(m => m.std) : !!(o.model && o.model.std)));
    DATA.burnup._has_std = Object.fromEntries(DATA.burnup.order.map(l => [l, hasStd(DATA.burnup.models[l])]));
  }
  if (DATA.screening) DATA.screening._slots = assignSlots(DATA.screening.methods);
  const views = [];
  if (DATA.screening) views.push({ id: 'scr', stage: 'Stage 1', title: 'Parameter screening', fn: viewScreening });
  if (DATA.keff) views.push({ id: 'keff', stage: 'Stage 2', title: 'k_eff predictor', fn: viewKeff });
  if (DATA.burnup) views.push({ id: 'burn', stage: 'Stage 3', title: 'Burnup predictor', fn: viewBurnup });
  if (DATA.burnup && DATA.optimize) views.push({ id: 'opt', stage: 'Stage 4', title: 'Fuel-cycle optimiser', fn: viewOptimize });
  views.push({ id: 'data', stage: 'Training', title: 'Data & accuracy', fn: viewData });
  const tabs = $('#tabs'), host = $('#views'), redraws = {};
  views.forEach(v => {
    const b = el('button', { class: 'tab', role: 'tab', id: 'tab_' + v.id, 'aria-selected': 'false', 'aria-controls': 'view_' + v.id },
      el('span', { class: 'stage', text: v.stage }), el('span', { class: 't', text: v.title }));
    tabs.append(b);
    host.append(el('section', { class: 'view', id: 'view_' + v.id, role: 'tabpanel', hidden: true }));
    b.addEventListener('click', () => show(v.id));
  });
  function show(id) {
    views.forEach(v => { const on = v.id === id; $('#tab_' + v.id).setAttribute('aria-selected', String(on)); $('#view_' + v.id).hidden = !on; });
    const v = views.find(w => w.id === id), sec = $('#view_' + id);
    if (!redraws[id]) {
      try { redraws[id] = v.fn(sec) || (() => {}); } catch (e) { console.error(e); sec.append(el('p', { class: 'note alert', text: 'This view could not be built: ' + e.message })); redraws[id] = () => {}; }
    } else requestAnimationFrame(redraws[id]);
    try { localStorage.setItem('msrframe_tab', id); } catch (e) { /* storage unavailable */ }
  }
  let saved = null;
  try { saved = localStorage.getItem('msrframe_tab'); } catch (e) { saved = null; }
  show(views.some(v => v.id === saved) ? saved : views[0].id);
  // palette toggle (colour / black & white), remembered per browser
  let pal = 'colour';
  try { pal = localStorage.getItem('msrframe_palette') || 'colour'; } catch (e) { /* ignore */ }
  const applyPal = p => { if (p === 'mono') document.documentElement.dataset.palette = 'mono'; else delete document.documentElement.dataset.palette; };
  applyPal(pal);
  $('#tools').append(el('span', { text: 'Series:' }), segmented([{ value: 'colour', label: 'Colour' }, { value: 'mono', label: 'Black & white' }], pal, p => {
    applyPal(p);
    try { localStorage.setItem('msrframe_palette', p); } catch (e) { /* ignore */ }
  }));
  let w = window.innerWidth;
  window.addEventListener('resize', debounce(() => {
    if (Math.abs(window.innerWidth - w) < 30) return;
    w = window.innerWidth;
    const cur = views.find(v => $('#tab_' + v.id).getAttribute('aria-selected') === 'true');
    if (cur && redraws[cur.id]) redraws[cur.id]();
  }, 200));
})();
