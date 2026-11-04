/* msrframe dashboard charts: small dependency-free SVG charts with hover tooltips.
 * Colours come from CSS tokens (--s1 ... --s8); every series also has its own dash pattern
 * and marker shape, so the charts stay readable in black & white and for colour-blind readers. */
'use strict';
const Charts = (() => {
  const NS = 'http://www.w3.org/2000/svg';
  const DASH = ['', '7 3', '2 3', '9 3 2 3', '4 2', '12 3', '1 2', '6 2 1 2'];
  let uid = 0;
  const isF = Number.isFinite;

  function sv(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function hel(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  // slot 0 = the reference / OpenMC series, drawn in ink
  const colour = slot => (slot === 0 ? 'var(--ink)' : `var(--s${((slot - 1) % 8) + 1})`);
  const dash = slot => (slot === 0 ? '' : DASH[(slot - 1) % 8]);

  /** Marker path of size r centred at (x, y); shape follows the series slot. */
  function markerPath(slot, x, y, r) {
    const k = slot === 0 ? 0 : (slot - 1) % 8;
    if (k === 0) return `M${x - r},${y}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0`;
    if (k === 1) return `M${x - r},${y - r}h${2 * r}v${2 * r}h${-2 * r}z`;
    if (k === 2) return `M${x},${y - r * 1.2}L${x + r * 1.1},${y + r * 0.9}L${x - r * 1.1},${y + r * 0.9}z`;
    if (k === 3) return `M${x},${y - r * 1.3}L${x + r * 1.1},${y}L${x},${y + r * 1.3}L${x - r * 1.1},${y}z`;
    if (k === 4) return `M${x},${y + r * 1.2}L${x + r * 1.1},${y - r * 0.9}L${x - r * 1.1},${y - r * 0.9}z`;
    if (k === 5) { const a = r * 0.45; return `M${x - a},${y - r}h${2 * a}v${r - a}h${r - a}v${2 * a}h${a - r}v${r - a}h${-2 * a}v${a - r}h${a - r}v${-2 * a}h${r - a}z`; }
    if (k === 6) { const a = r * 0.9; return `M${x - a},${y - a}L${x + a},${y + a}M${x + a},${y - a}L${x - a},${y + a}`; }
    let p = '';
    for (let i = 0; i < 10; i++) { const ang = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.5 : r * 1.25; p += `${i ? 'L' : 'M'}${x + rr * Math.cos(ang)},${y + rr * Math.sin(ang)}`; }
    return p + 'z';
  }
  /** shape (optional) picks the marker shape independently of the colour slot. */
  function marker(g, slot, x, y, r, filled = true, shape = null) {
    const sh = shape === null ? slot : shape;
    const open = sh !== 0 && (sh - 1) % 8 === 6;
    return sv('path', { d: markerPath(sh, x, y, r), class: 'mk', style: `stroke:${colour(slot)};fill:${filled && !open ? colour(slot) : 'var(--panel)'};stroke-width:${open ? 2 : 1.5}` }, g);
  }
  /** Small legend swatch: line with dash pattern + marker. */
  function swatch(slot, opts = {}) {
    const s = sv('svg', { width: 30, height: 12, viewBox: '0 0 30 12', class: 'lkey', 'aria-hidden': 'true' });
    if (opts.band) sv('rect', { x: 0, y: 1, width: 30, height: 10, style: `fill:${colour(slot)};opacity:.18` }, s);
    if (opts.line !== false) sv('line', { x1: 1, y1: 6, x2: 29, y2: 6, style: `stroke:${colour(slot)};stroke-width:2;stroke-dasharray:${dash(slot)}` }, s);
    if (opts.marker !== false) marker(s, slot, 15, 6, 3.6, opts.filled !== false, opts.shape ?? null);
    return s;
  }

  // ---------------------------------------------------------------- ticks & formatting
  function niceStep(span, n) {
    const raw = span / Math.max(n, 1), mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  }
  function niceTicks(lo, hi, n = 5) {
    if (!(hi > lo)) return [lo];
    const st = niceStep(hi - lo, n), out = [];
    for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-9; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
    out.step = st;
    return out;
  }
  function logTicks(lo, hi) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      const v = Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
    if (out.length < 3) {
      for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
        for (const m of [2, 5]) { const v = m * Math.pow(10, e); if (v >= lo && v <= hi) out.push(v); }
      out.sort((a, b) => a - b);
    }
    return out;
  }
  function tickLabel(v, ticks) {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) {
      const e = Math.floor(Math.log10(a)), m = v / Math.pow(10, e);
      return (Math.abs(m - 1) < 1e-9 ? '' : `${+m.toFixed(1)}×`) + `10${sup(e)}`;
    }
    const st = ticks && ticks.step ? ticks.step : a;
    const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(st) + 1e-9)));
    return v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
  }
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const sup = e => String(e).split('').map(c => SUP[c]).join('');
  function fmtSig(v, s = 4) {
    if (!isF(v)) return '–';
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e6 || a < 1e-4) return v.toExponential(Math.max(s - 1, 0));
    return Number(v.toPrecision(s)).toLocaleString('en-US', { maximumFractionDigits: 8 });
  }

  function makeScale(d0, d1, r0, r1, log) {
    if (log) {
      const a = Math.log10(d0), b = Math.log10(d1);
      const f = v => r0 + (Math.log10(Math.max(v, 1e-300)) - a) / (b - a || 1) * (r1 - r0);
      f.inv = p => Math.pow(10, a + (p - r0) / (r1 - r0) * (b - a));
      return f;
    }
    const f = v => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0);
    f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
    return f;
  }
  function extent(vals, log) {
    const v = vals.filter(x => isF(x) && (!log || x > 0));
    if (!v.length) return [0, 1];
    let lo = Math.min(...v), hi = Math.max(...v);
    if (lo === hi) { const d = Math.abs(lo) * 0.05 || 1; lo -= d; hi += d; }
    return [lo, hi];
  }

  // ---------------------------------------------------------------- frame shared by line & scatter charts
  function frame(host, cfg, xd, yd) {
    host.replaceChildren();
    const wrap = hel('div', 'chart');
    host.appendChild(wrap);
    if (cfg.legend && cfg.legend.length) {
      const lg = hel('div', 'legend');
      cfg.legend.forEach(it => { const s = hel('span', 'litem'); s.append(swatch(it.slot, it), document.createTextNode(it.name)); lg.append(s); });
      wrap.appendChild(lg);
    }
    const W = Math.max(260, Math.round(host.clientWidth || 640)), H = cfg.height || 260;
    const m = { l: cfg.ml || 62, r: cfg.mr || 16, t: 10, b: cfg.xLabel ? 44 : 26 };
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': cfg.aria || cfg.yLabel || 'chart' });
    wrap.appendChild(svg);
    const px0 = m.l, px1 = W - m.r, py0 = H - m.b, py1 = m.t;
    const sx = makeScale(xd[0], xd[1], px0, px1, cfg.xLog), sy = makeScale(yd[0], yd[1], py0, py1, cfg.yLog);
    const xt = cfg.xLog ? logTicks(xd[0], xd[1]) : niceTicks(xd[0], xd[1], Math.max(3, Math.round((px1 - px0) / 90)));
    const yt = cfg.yLog ? logTicks(yd[0], yd[1]) : niceTicks(yd[0], yd[1], Math.max(3, Math.round((py0 - py1) / 50)));
    const g = sv('g', {}, svg);
    yt.forEach(v => {
      const y = sy(v);
      if (y < py1 - 0.5 || y > py0 + 0.5) return;
      sv('line', { x1: px0, x2: px1, y1: y, y2: y, class: 'gridline' }, g);
      sv('text', { x: px0 - 6, y: y + 4, 'text-anchor': 'end', class: 'tick' }, g).textContent = cfg.yTick ? cfg.yTick(v) : tickLabel(v, yt);
    });
    xt.forEach(v => {
      const x = sx(v);
      if (x < px0 - 0.5 || x > px1 + 0.5) return;
      sv('line', { x1: x, x2: x, y1: py0, y2: py0 + 4, class: 'axisline' }, g);
      sv('text', { x, y: py0 + 17, 'text-anchor': 'middle', class: 'tick' }, g).textContent = cfg.xTick ? cfg.xTick(v) : tickLabel(v, xt);
    });
    sv('line', { x1: px0, x2: px1, y1: py0, y2: py0, class: 'axisline' }, g);
    sv('line', { x1: px0, x2: px0, y1: py1, y2: py0, class: 'axisline' }, g);
    if (cfg.xLabel) sv('text', { x: (px0 + px1) / 2, y: H - 6, 'text-anchor': 'middle', class: 'axis-title' }, g).textContent = cfg.xLabel;
    if (cfg.yLabel) sv('text', { x: 12, y: (py0 + py1) / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${(py0 + py1) / 2})`, class: 'axis-title' }, g).textContent = cfg.yLabel;
    const cid = 'clip' + (++uid);
    const defs = sv('defs', {}, svg);
    sv('rect', { x: px0, y: py1 - 2, width: px1 - px0, height: py0 - py1 + 4 }, sv('clipPath', { id: cid }, defs));
    const plot = sv('g', { 'clip-path': `url(#${cid})` }, svg);
    const tip = hel('div', 'tip');
    tip.hidden = true;
    wrap.appendChild(tip);
    return { wrap, svg, plot, sx, sy, W, H, px0, px1, py0, py1, tip };
  }
  function placeTip(F, tip, px, py) {
    tip.hidden = false;
    const ww = F.wrap.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
    const svgTop = F.svg.getBoundingClientRect().top - F.wrap.getBoundingClientRect().top;
    const scale = F.svg.getBoundingClientRect().width / F.W || 1;
    let left = px * scale + 14;
    if (left + tw > ww) left = px * scale - tw - 14;
    tip.style.left = Math.max(0, left) + 'px';
    tip.style.top = Math.max(0, svgTop + py * scale - th / 2) + 'px';
  }

  // ---------------------------------------------------------------- line chart
  /** cfg: {series:[{name, slot, pts:[[x,y]], band:[[x,lo,hi]], line, markers, width, noLegend, noHover, filled}],
   *   xLabel, yLabel, xDomain, yDomain, xLog, yLog, hlines:[{y,label}], vlines:[{x,label}], hbands:[{y0,y1,label}],
   *   vbands:[{x0,x1,label}], tipFmt(v), xTipFmt(v), height} */
  function lineChart(host, cfg) {
    const series = cfg.series.filter(s => s && ((s.pts && s.pts.length) || (s.band && s.band.length)));
    const xs = [], ys = [];
    series.forEach(s => {
      (s.pts || []).forEach(p => { xs.push(p[0]); ys.push(p[1]); });
      (s.band || []).forEach(b => { xs.push(b[0]); ys.push(b[1], b[2]); });
    });
    (cfg.hlines || []).forEach(h => { if (h.fit !== false) ys.push(h.y); });
    (cfg.hbands || []).forEach(b => ys.push(b.y0, b.y1));
    const xd = cfg.xDomain || extent(xs, cfg.xLog);
    let yd = cfg.yDomain || extent(ys, cfg.yLog);
    if (!cfg.yDomain && !cfg.yLog) { const pad = (yd[1] - yd[0]) * 0.06; yd = [yd[0] - pad, yd[1] + pad]; }
    if (!cfg.yDomain && cfg.yLog) yd = [yd[0] / 1.15, yd[1] * 1.15];
    cfg.legend = cfg.legend || series.filter(s => !s.noLegend).map(s => ({ name: s.name, slot: s.slot, band: !!s.band && !s.pts, line: s.line !== false, marker: !!s.markers || s.line === false }));
    const F = frame(host, cfg, xd, yd), { plot, sx, sy } = F;
    (cfg.hbands || []).forEach(b => {
      const y0 = sy(Math.max(b.y0, yd[0])), y1 = sy(Math.min(b.y1, yd[1]));
      sv('rect', { x: F.px0, width: F.px1 - F.px0, y: Math.min(y0, y1), height: Math.abs(y1 - y0), class: 'hband' }, plot);
      if (b.label) sv('text', { x: F.px1 - 6, y: Math.min(y0, y1) + 13, 'text-anchor': 'end', class: 'band-label' }, plot).textContent = b.label;
    });
    (cfg.vbands || []).forEach(b => {
      const x0 = sx(b.x0), x1 = sx(b.x1);
      sv('rect', { x: Math.min(x0, x1), width: Math.abs(x1 - x0), y: F.py1, height: F.py0 - F.py1, class: 'vband' }, plot);
      if (b.label) sv('text', { x: Math.min(x0, x1) + 4, y: F.py1 + 12, class: 'band-label' }, plot).textContent = b.label;
    });
    series.forEach(s => {
      if (!s.band || !s.band.length) return;
      const b = s.band.filter(p => isF(p[1]) && isF(p[2]));
      if (!b.length) return;
      const d = b.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0])},${sy(p[2])}`).join('') + b.slice().reverse().map(p => `L${sx(p[0])},${sy(p[1])}`).join('') + 'Z';
      sv('path', { d, class: 'area', style: `fill:${colour(s.slot)}` }, plot);
    });
    (cfg.hlines || []).forEach(h => {
      const y = sy(h.y);
      if (y < F.py1 || y > F.py0) return;
      sv('line', { x1: F.px0, x2: F.px1, y1: y, y2: y, class: 'refline' }, plot);
      if (h.label) sv('text', { x: F.px1 - 4, y: y - 4, 'text-anchor': 'end', class: 'ref-label' }, plot).textContent = h.label;
    });
    (cfg.vlines || []).forEach(v => {
      const x = sx(v.x);
      sv('line', { x1: x, x2: x, y1: F.py1, y2: F.py0, class: 'refline' }, plot);
      if (v.label) sv('text', { x: x + 4, y: F.py1 + 12, class: 'ref-label' }, plot).textContent = v.label;
    });
    series.forEach(s => {
      const pts = (s.pts || []).filter(p => isF(p[0]) && isF(p[1]) && (!cfg.yLog || p[1] > 0));
      if (s.line !== false && pts.length > 1) {
        let d = '', pen = false;
        (s.pts || []).forEach(p => {
          if (!isF(p[1]) || (cfg.yLog && p[1] <= 0)) { pen = false; return; }
          d += `${pen ? 'L' : 'M'}${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`;
          pen = true;
        });
        sv('path', { d, class: 'line', style: `stroke:${colour(s.slot)};stroke-width:${s.width || 2};stroke-dasharray:${s.dash === undefined ? dash(s.slot) : s.dash}` }, plot);
      }
      if (s.markers || s.line === false) pts.forEach(p => marker(plot, s.slot, sx(p[0]), sy(p[1]), s.r || 3.6, s.filled !== false));
    });
    // ---- hover: crosshair + values of every series at the nearest x
    const hov = series.filter(s => !s.noHover && s.pts && s.pts.length);
    if (!hov.length) return F;
    const xsAll = [...new Set(hov.flatMap(s => s.pts.map(p => p[0])).filter(isF))].sort((a, b) => a - b);
    const cross = sv('line', { y1: F.py1, y2: F.py0, class: 'crosshair', visibility: 'hidden' }, F.svg);
    const dots = sv('g', {}, F.svg);
    const overlay = sv('rect', { x: F.px0, y: F.py1, width: F.px1 - F.px0, height: F.py0 - F.py1, class: 'overlay' }, F.svg);
    const tipFmt = cfg.tipFmt || (v => fmtSig(v, 5)), xFmt = cfg.xTipFmt || (v => fmtSig(v, 4));
    const move = ev => {
      const r = F.svg.getBoundingClientRect(), px = (ev.clientX - r.left) * F.W / r.width;
      let best = xsAll[0], bd = Infinity;
      for (const x of xsAll) { const dd = Math.abs(sx(x) - px); if (dd < bd) { bd = dd; best = x; } }
      const X = sx(best);
      cross.setAttribute('x1', X); cross.setAttribute('x2', X); cross.setAttribute('visibility', 'visible');
      dots.replaceChildren();
      F.tip.replaceChildren(hel('div', 'th', `${cfg.xName || cfg.xLabel || 'x'}: ${xFmt(best)}`));
      let yMid = 0, cnt = 0;
      hov.forEach(s => {
        let p = null, pd = Infinity;
        for (const q of s.pts) { const dd = Math.abs(q[0] - best); if (dd < pd) { pd = dd; p = q; } }
        if (!p || !isF(p[1]) || Math.abs(sx(p[0]) - X) > 3) return;
        marker(dots, s.slot, X, sy(p[1]), 4.2, true);
        yMid += sy(p[1]); cnt++;
        let extra = '';
        if (s.band) { const b = s.band.find(q => Math.abs(q[0] - p[0]) < 1e-12); if (b && isF(b[1])) extra = ` (${tipFmt(b[1])} … ${tipFmt(b[2])})`; }
        if (s.tipExtra) extra += s.tipExtra(p);
        const row = hel('div', 'trow');
        row.append(swatch(s.slot, { band: false, marker: true, line: s.line !== false }), hel('span', 'tl', s.name + ' '), hel('span', 'tv', tipFmt(p[1], s)), hel('span', 'tl', extra));
        F.tip.append(row);
      });
      placeTip(F, F.tip, X, cnt ? yMid / cnt : (F.py0 + F.py1) / 2);
    };
    const leave = () => { cross.setAttribute('visibility', 'hidden'); dots.replaceChildren(); F.tip.hidden = true; };
    overlay.addEventListener('pointermove', move);
    overlay.addEventListener('pointerdown', move);
    overlay.addEventListener('pointerleave', leave);
    return F;
  }

  // ---------------------------------------------------------------- scatter chart (parity plots)
  /** cfg: {points:[{x,y,slot,label,extra}], diag:true, xLabel, yLabel, tip(p), height, legend} */
  function scatterChart(host, cfg) {
    const pts = cfg.points.filter(p => isF(p.x) && isF(p.y));
    const all = pts.flatMap(p => [p.x, p.y]);
    let xd = extent(cfg.diag ? all : pts.map(p => p.x)), yd = extent(cfg.diag ? all : pts.map(p => p.y));
    const pad = (xd[1] - xd[0]) * 0.04, pady = (yd[1] - yd[0]) * 0.04;
    xd = [xd[0] - pad, xd[1] + pad]; yd = [yd[0] - pady, yd[1] + pady];
    const F = frame(host, cfg, cfg.xDomain || xd, cfg.yDomain || yd), { plot, sx, sy } = F;
    if (cfg.diag) {
      const a = Math.max(xd[0], yd[0]), b = Math.min(xd[1], yd[1]);
      sv('line', { x1: sx(a), y1: sy(a), x2: sx(b), y2: sy(b), class: 'refline' }, plot);
    }
    pts.forEach(p => {
      if (p.err && isF(p.err)) sv('line', { x1: sx(p.x), x2: sx(p.x), y1: sy(p.y - p.err), y2: sy(p.y + p.err), class: 'errbar' }, plot);
    });
    pts.forEach(p => marker(plot, p.slot || 1, sx(p.x), sy(p.y), p.r || 3.4, p.filled !== false));
    const ring = sv('g', {}, F.svg);
    const overlay = sv('rect', { x: F.px0, y: F.py1, width: F.px1 - F.px0, height: F.py0 - F.py1, class: 'overlay' }, F.svg);
    const move = ev => {
      const r = F.svg.getBoundingClientRect(), px = (ev.clientX - r.left) * F.W / r.width, py = (ev.clientY - r.top) * F.W / r.width;
      let best = null, bd = 24 * 24;
      for (const p of pts) { const dx = sx(p.x) - px, dy = sy(p.y) - py, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = p; } }
      ring.replaceChildren();
      if (!best) { F.tip.hidden = true; return; }
      sv('circle', { cx: sx(best.x), cy: sy(best.y), r: 7, class: 'ring' }, ring);
      F.tip.replaceChildren(...(cfg.tip ? cfg.tip(best) : [hel('div', 'th', best.label || ''), hel('div', 'trow', `${fmtSig(best.x, 5)}, ${fmtSig(best.y, 5)}`)]));
      placeTip(F, F.tip, sx(best.x), sy(best.y));
    };
    overlay.addEventListener('pointermove', move);
    overlay.addEventListener('pointerdown', move);
    overlay.addEventListener('pointerleave', () => { ring.replaceChildren(); F.tip.hidden = true; });
    return F;
  }

  // ---------------------------------------------------------------- importance dot plot (log x)
  /** cfg: {rows:[{label, sub, bold, muted, cells:{method:{v, keep, rank}}}], methods:[{name, slot}], thresholds:{method:v}, xLabel} */
  function dotPlot(host, cfg) {
    host.replaceChildren();
    const wrap = hel('div', 'chart');
    host.appendChild(wrap);
    const lg = hel('div', 'legend');
    const SHAPES = [1, 2, 3, 4, 5];                      // shapes that show filled vs hollow
    cfg.methods.forEach((m, i) => { m.shape = SHAPES[i % SHAPES.length]; const s = hel('span', 'litem'); s.append(swatch(m.slot, { line: false, shape: m.shape }), document.createTextNode(m.name)); lg.append(s); });
    const s2 = hel('span', 'litem muted', 'filled = kept · hollow = dropped · dashed ticks = each method’s noise-calibrated threshold');
    lg.append(s2);
    wrap.appendChild(lg);
    const rowH = 19, W = Math.max(300, Math.round(host.clientWidth || 640));
    const labW = Math.min(300, Math.max(120, W * 0.36));
    const m = { l: labW, r: 14, t: 6, b: 40 }, H = m.t + m.b + rowH * cfg.rows.length;
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': cfg.aria || 'importance by method' });
    wrap.appendChild(svg);
    const vals = cfg.rows.flatMap(r => Object.values(r.cells).map(c => c.v)).filter(v => isF(v) && v > 0);
    const lo = Math.max(Math.min(...vals, ...Object.values(cfg.thresholds || {}).filter(v => v > 0)) / 2, 1e-7), hi = 1;
    const sx = makeScale(lo, hi, m.l, W - m.r, true);
    const ticks = logTicks(lo, hi);
    ticks.forEach(v => {
      const x = sx(v);
      sv('line', { x1: x, x2: x, y1: m.t, y2: H - m.b, class: 'gridline' }, svg);
      sv('text', { x, y: H - m.b + 16, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = tickLabel(v);
    });
    sv('text', { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'axis-title' }, svg).textContent = cfg.xLabel || 'share of importance (log)';
    const nM = cfg.methods.length, off = i => (i - (nM - 1) / 2) * Math.min(4.5, 12 / nM);
    cfg.methods.forEach((mm, i) => {
      const t = (cfg.thresholds || {})[mm.name];
      if (!isF(t) || t <= 0) return;
      cfg.rows.forEach((r, ri) => {
        const y = m.t + ri * rowH + rowH / 2 + off(i);
        sv('line', { x1: sx(t), x2: sx(t), y1: y - 4, y2: y + 4, style: `stroke:${colour(mm.slot)};stroke-width:1.2;opacity:.7` }, svg);
      });
    });
    const hits = [];
    cfg.rows.forEach((r, ri) => {
      const y = m.t + ri * rowH + rowH / 2;
      if (ri % 2 === 0) sv('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, class: 'zebra' }, svg);
      const tx = sv('text', { x: m.l - 8, y: y + 4, 'text-anchor': 'end', class: 'rowlabel' + (r.bold ? ' strong' : '') + (r.muted ? ' muted' : '') }, svg);
      const maxc = Math.max(14, Math.floor(labW / 6.4));
      tx.textContent = r.label.length > maxc ? r.label.slice(0, maxc - 1) + '…' : r.label;
      cfg.methods.forEach((mm, i) => {
        const c = r.cells[mm.name];
        if (!c || !isF(c.v) || c.v <= 0) return;
        const px = sx(Math.max(c.v, lo)), py = y + off(i);
        marker(svg, mm.slot, px, py, 3.8, !!c.keep, mm.shape);
        hits.push({ px, py, r, c, mm });
      });
    });
    const tip = hel('div', 'tip');
    tip.hidden = true;
    wrap.appendChild(tip);
    const F = { wrap, svg, W, tip };
    const ring = sv('g', {}, svg);
    svg.addEventListener('pointermove', ev => {
      const rr = svg.getBoundingClientRect(), px = (ev.clientX - rr.left) * W / rr.width, py = (ev.clientY - rr.top) * W / rr.width;
      let best = null, bd = 18 * 18;
      for (const h of hits) { const dx = h.px - px, dy = h.py - py, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = h; } }
      ring.replaceChildren();
      if (!best) { tip.hidden = true; return; }
      sv('circle', { cx: best.px, cy: best.py, r: 7, class: 'ring' }, ring);
      tip.replaceChildren(hel('div', 'th', best.r.full || best.r.label), hel('div', 'trow', `${best.mm.name}: share ${fmtSig(best.c.v, 3)} · rank ${best.c.rank}`), hel('div', 'trow', best.c.keep ? 'kept' : 'dropped'));
      placeTip(F, tip, best.px, best.py);
    });
    svg.addEventListener('pointerleave', () => { ring.replaceChildren(); tip.hidden = true; });
    return F;
  }

  return { lineChart, scatterChart, dotPlot, swatch, niceTicks, fmtSig, colour, dash };
})();
