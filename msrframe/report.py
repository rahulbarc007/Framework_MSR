"""Self-contained HTML report (figures embedded) with one tab per stage."""
from __future__ import annotations

import base64
import datetime as _dt
import html
from pathlib import Path

import numpy as np
import pandas as pd

CSS = """
:root{--ground:#fcfcfb;--panel:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#d9d8d2;--accent:#2a78d6}
*{box-sizing:border-box}body{margin:0;padding:0 16px 40px;background:var(--ground);color:var(--ink);
font:14px/1.5 "Times New Roman",Times,"Liberation Serif",serif}.wrap{max-width:1280px;margin:0 auto}
header{padding:22px 0 14px;border-bottom:1px solid var(--ink)}h1{font-size:28px;margin:0}h2{font-size:21px;margin:18px 0 6px}
h3{font-size:15px;margin:16px 0 6px}.sub{color:var(--ink2);margin-top:6px;max-width:80ch}
.tabs{display:flex;gap:4px;overflow-x:auto;padding:12px 0;position:sticky;top:0;background:var(--ground);z-index:2}
.tab{border:1px solid var(--line);background:var(--panel);padding:7px 12px;cursor:pointer;white-space:nowrap;font:inherit}
.tab[aria-selected=true]{background:var(--ink);color:#fff;border-color:var(--ink)}
.verdict{border-left:3px solid var(--ink);padding:10px 14px;background:var(--panel);margin:10px 0 16px;max-width:110ch}
figure{margin:14px 0 22px;background:var(--panel);border:1px solid var(--line);padding:10px}figure img{max-width:100%;height:auto;display:block}
figcaption{color:var(--ink2);font-size:13px;margin-top:6px}.tw{overflow-x:auto;border:1px solid var(--line);background:var(--panel);margin:6px 0 18px}
table{border-collapse:collapse;font-size:12.5px;min-width:100%}th,td{padding:4px 8px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap}
th{background:#f3f2ee;position:sticky;top:0}td:first-child,th:first-child{text-align:left}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0}
.tile{border:1px solid var(--line);background:var(--panel);padding:10px 12px}.tile b{font-size:22px;display:block}.tile span{color:var(--ink2);font-size:12px}
.note{color:var(--ink2);font-size:13px}
"""

JS = """
const tabs=[...document.querySelectorAll('.tab')],views=[...document.querySelectorAll('.view')];
function show(i){tabs.forEach((t,j)=>t.setAttribute('aria-selected',String(i===j)));views.forEach((v,j)=>v.hidden=i!==j);
try{localStorage.setItem('msrframe_tab',i)}catch(e){}}
tabs.forEach((t,i)=>t.onclick=()=>show(i));let s=0;try{s=+localStorage.getItem('msrframe_tab')||0}catch(e){};show(Math.min(s,tabs.length-1));
"""


def _img(path):
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{b}"


def _table(df: pd.DataFrame, n=40):
    df = df.head(n)

    def f(v):
        if isinstance(v, (bool, np.bool_)):
            return "✓" if v else ""
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "–"
            a = abs(v)
            return f"{v:,.0f}" if a >= 1000 else (f"{v:.4g}")
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{f(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def write_report(cfg, results, extra_figs=None, path=None):
    path = Path(path or cfg.out / "report.html")
    tabs, views = [], []
    tiles = []
    for R in results:
        s = R["summary"]
        if R["stage"] == "screening":
            tiles.append((s["n_runs"], "screening runs", f"{s['n_inputs']} inputs -> {len(s['consensus'])} kept"))
        elif R["stage"] == "keff":
            tiles.append((s["n_runs"], "k_eff runs", f"best: {s['ranking'][0]} ({s['cv'][s['ranking'][0]]['rmse']:.0f} pcm)"))
        elif R["stage"] == "burnup":
            tiles.append((s["n_histories"], "burnup histories", f"best: {s['ranking'][0]}"))
        elif R["stage"] == "optimize":
            tiles.append((len(s["results"]), "scenarios optimised", ", ".join(sorted({r['status'].split(',')[0] for r in s['results']}))))
    ov = ['<h2>Overview</h2><div class="tiles">' + "".join(
        f'<div class="tile"><b>{v}</b>{html.escape(l)}<br><span>{html.escape(d)}</span></div>' for v, l, d in tiles) + "</div>"]
    for R in results:
        ov.append(f"<h3>{html.escape(R['title'])}</h3><div class='verdict'>{html.escape(R['verdict'])}</div>")
    ov.append("<h3>Plug-in configuration used</h3>" + _table(pd.DataFrame(
        [dict(stage=st, component=", ".join(str(x) for x in cfg[st].get(k, [])) or "-", key=k)
         for st, k in (("screening", "methods"), ("keff", "models"), ("burnup", "models"), ("optimize", "optimizers"))
         if cfg[st].get("enabled", True)])))
    ov.append("<p class='note'>Swap any component by editing the YAML (e.g. <code>morris</code> → <code>random_forest</code>), "
              "add OpenMC runs to <code>data/user/</code> and re-run <code>python -m msrframe run</code>.</p>")
    tabs.append("Overview")
    views.append("".join(ov))
    for R in results:
        body = [f"<h2>{html.escape(R['title'])}</h2><div class='verdict'>{html.escape(R['verdict'])}</div>"]
        for p, cap in R["figures"]:
            body.append(f"<figure><img alt='{html.escape(cap)}' src='{_img(p)}'><figcaption>{html.escape(cap)}</figcaption></figure>")
        for name, df in R["tables"].items():
            body.append(f"<h3>{html.escape(name.replace('_', ' '))}</h3>{_table(df)}")
        tabs.append(R["title"].split(" - ")[-1])
        views.append("".join(body))
    for title, figs in (extra_figs or {}).items():
        tabs.append(title)
        views.append(f"<h2>{html.escape(title)}</h2>" + "".join(
            f"<figure><img alt='{html.escape(c)}' src='{_img(p)}'><figcaption>{html.escape(c)}</figcaption></figure>" for p, c in figs))
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(cfg['project'])}</title><style>{CSS}</style></head>
<body><div class="wrap"><header><div class="note">msrframe · trainable MSR neutronics design framework</div><h1>{html.escape(cfg['project'])}</h1>
<div class="sub">Generated {now}. Every stage's methods are plug-ins chosen in the YAML config; all numbers come from the data files listed there.</div></header>
<nav class="tabs">{''.join(f'<button class="tab" role="tab">{html.escape(t)}</button>' for t in tabs)}</nav>
{''.join(f'<section class="view" hidden>{v}</section>' for v in views)}</div><script>{JS}</script></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path
