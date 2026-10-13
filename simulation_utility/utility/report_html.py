"""Tabelle di riepilogo (le stesse della console) come HTML, da embeddare nel Gantt: SIMULATION
INFO, dettaglio per missione, riepilogo per pipeline, totale costellazione."""
import csv as _csv
import os

# Prefisso operazione -> nome missione (ordine di esecuzione).
_MISSIONS = [
    ("IS", "imagery_slicing"),
    ("CD", "cloud_detection"),
    ("GC", "geometric_correction"),
    ("CF", "classification"),
]

_CSS = """
<style>
.rep-wrap { font-family: system-ui, sans-serif; margin: 24px 40px; color: #222; }
.rep-wrap h3 { margin: 22px 0 6px; font-size: 15px; }
.rep-wrap table { border-collapse: collapse; font-size: 13px; }
.rep-wrap .info td { padding: 2px 14px 2px 0; }
.rep-wrap .info td:first-child { color: #666; }
.rep-wrap .rep { min-width: 720px; }
.rep-wrap .rep th, .rep-wrap .rep td { border: 1px solid #ddd; padding: 4px 10px; }
.rep-wrap .rep th { background: #f4f5f6; text-align: center; font-weight: 600; }
.rep-wrap .rep td.num { text-align: right; font-variant-numeric: tabular-nums; }
.rep-wrap .rep td.name { font-weight: 600; }
.rep-wrap .rep tr.pgroup td { border-top: 2px solid #bbb; }
</style>
"""


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ms(v) -> str:
    return f"{v:.4f}ms" if v is not None else "N/A"


def _j(v) -> str:
    return f"{v:.4f}J"


def _read(tasks_csv):
    """Legge le righe del CSV in dizionari con float; ritorna gli id pipeline in
    ordine di prima apparizione."""
    rows, order = [], []
    with open(tasks_csv, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            dag = r.get("dag_id", "")
            if dag not in order:
                order.append(dag)
            rows.append({
                "dag": dag,
                "op":  r.get("operation", ""),
                "start": _f(r.get("start_time_ms")),
                "end":   _f(r.get("end_time_ms")),
                "energy": (_f(r.get("energy_cpu_J")) or 0.0) + (_f(r.get("energy_net_J")) or 0.0),
                "dtype": r.get("device_type", ""),
            })
    return rows, order


def _span(chunks):
    """(inizio, fine, energia) di un insieme di chunk; None se tempi mancanti."""
    energy = sum(c["energy"] for c in chunks)
    ok = [c for c in chunks if c["start"] is not None and c["end"] is not None]
    if not ok:
        return None, None, energy
    return min(c["start"] for c in ok), max(c["end"] for c in ok), energy


def _sim_info(config, order):
    seen = {}
    for dag in order:
        pid = dag.split("_", 1)[1] if "_" in dag else dag        # "SAT-1_P-4096" -> "P-4096"
        px  = pid.split("-", 1)[1] if pid.startswith("P-") else "?"
        seen[pid] = px
    sim_config = config.get("SIMULATION", {})
    policy = sim_config.get("SCHEDULER", {}).get("policy_scheduler", {}).get("selected", "?")
    info = [
        ("Seed",              config.get("seed", "?")),
        ("Scheduling policy", f"{policy} (+ EFT)"),
        ("Pipelines (istanze)", len(order)),          # una per generatore, es. SAT-1_P-4096
        ("Tipi pipeline",     ", ".join(seen.keys())),
        ("Pipeline image px", ", ".join(f"{v}px" for v in seen.values())),
        ("Scenario",          sim_config.get("SCENARIO", {}).get("scenario", "?")),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in info)
    return f'<h3>Simulation Info</h3><table class="info">{body}</table>'


def _mission_detail(rows, order):
    trs = ""
    for dag in order:
        chunks = [c for c in rows if c["dag"] == dag]
        first = True
        for prefix, name in _MISSIONS:
            grp = [c for c in chunks if c["op"].split("-")[0] == prefix]
            if not grp:
                continue
            s, e, energy = _span(grp)
            d = (e - s) if (s is not None and e is not None) else None
            pid = dag if first else ""
            cls = ' class="pgroup"' if first else ""
            trs += (f'<tr{cls}><td class="name">{pid}</td><td>{name}</td>'
                    f'<td class="num">{_ms(s)}</td><td class="num">{_ms(e)}</td>'
                    f'<td class="num">{_ms(d)}</td><td class="num">{_j(energy)}</td></tr>')
            first = False
    head = ("<tr><th>Pipeline</th><th>Mission</th><th>Inizio</th><th>Fine</th>"
            "<th>Durata</th><th>Energia</th></tr>")
    return (f'<h3>Dettaglio per missione</h3>'
            f'<table class="rep"><thead>{head}</thead><tbody>{trs}</tbody></table>')


def _pipeline_summary(rows, order):
    trs = ""
    for dag in order:
        chunks = [c for c in rows if c["dag"] == dag]
        s, e, energy = _span(chunks)
        d = (e - s) if (s is not None and e is not None) else None
        trs += (f'<tr><td class="name">{dag}</td>'
                f'<td class="num">{_ms(s)}</td><td class="num">{_ms(e)}</td>'
                f'<td class="num">{_ms(d)}</td><td class="num">{_j(energy)}</td></tr>')
    head = "<tr><th>Pipeline</th><th>Inizio</th><th>Fine</th><th>Durata</th><th>Energia</th></tr>"
    return (f'<h3>Riepilogo per pipeline</h3>'
            f'<table class="rep"><thead>{head}</thead><tbody>{trs}</tbody></table>')


def _constellation(rows):
    s, e, e_tot = _span(rows)
    d = (e - s) if (s is not None and e is not None) else None
    e_sat = sum(c["energy"] for c in rows if c["dtype"] != "GND")
    head = ("<tr><th>Inizio</th><th>Fine</th><th>Durata</th>"
            "<th>Energia (SAT+SAS)</th><th>Energia tot</th></tr>")
    row  = (f'<tr><td class="num">{_ms(s)}</td><td class="num">{_ms(e)}</td>'
            f'<td class="num">{_ms(d)}</td>'
            f'<td class="num">{_j(e_sat)}</td><td class="num">{_j(e_tot)}</td></tr>')
    return (f'<h3>Totale costellazione</h3>'
            f'<table class="rep"><thead>{head}</thead><tbody>{row}</tbody></table>')


def build_report_html(config: dict, tasks_csv: str) -> str:
    """HTML autocontenuto (CSS incluso) con info + le tre tabelle, dal CSV dei task."""
    if not tasks_csv or not os.path.isfile(tasks_csv):
        return ""
    rows, order = _read(tasks_csv)
    if not rows:
        return ""
    return (_CSS + '<div class="rep-wrap">'
            + _sim_info(config, order)
            + _mission_detail(rows, order)
            + _pipeline_summary(rows, order)
            + _constellation(rows)
            + '</div>')


def build_sim_info_html(config: dict, tasks_csv: str) -> str:
    """Solo il blocco Simulation Info (CSS incluso), per le pagine di confronto
    dove le info di più run vanno affiancate."""
    if not tasks_csv or not os.path.isfile(tasks_csv):
        return ""
    rows, order = _read(tasks_csv)
    if not rows:
        return ""
    return _CSS + '<div class="rep-wrap">' + _sim_info(config, order) + '</div>'
