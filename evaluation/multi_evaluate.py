"""Confronto degli stessi candidati su più scenari di test."""
import csv
from pathlib import Path

import json5

from utility.file_utility import RESULTS_PATH

_EVAL_ROOT  = Path(RESULTS_PATH) / "evaluation"
_MULTI_ROOT = Path(RESULTS_PATH) / "multi_evaluation"

# Stile dei candidati noti, lo stesso delle figure della tesi: nome esteso, colore, simbolo del
# marcatore e tratto della linea.
_STYLES = {
    "baseline":       ("Baseline",             "#7f7f7f", "diamond",       "dot"),
    "specialist":     ("Specialista",          "#1f77b4", "circle",        "solid"),
    "fedavg":         ("FedAvg",               "#ff7f0e", "square",        "dash"),
    "greedy_soup":    ("Greedy Soup",          "#2ca02c", "triangle-up",   "solid"),
    "lambda_scaling": ("Lambda Scaling",       "#d62728", "triangle-down", "dash"),
    "random_domain":  ("Domain Randomization", "#9467bd", "cross",         "dashdot"),
}
# Candidati non elencati sopra: colori in ordine di prima comparsa.
_PALETTE = ["#8c564b", "#e377c2", "#bcbd22", "#17becf"]
_MASK_SUFFIX = "-MASK"


#######################################################################################################
# LETTURA
#######################################################################################################

def _scenario_of(eval_folder: str) -> str:
    """Scenario di test del run (uguale per tutti i candidati della cartella)."""
    folder = _EVAL_ROOT / eval_folder
    candidates = [p for p in folder.iterdir() if p.is_dir()] if folder.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"Nessun run trovato in {folder} (EVALUATION non eseguita?)")
    with open(candidates[0] / "run_config.json5", encoding="utf-8") as f:
        run_config = json5.load(f)
    return run_config["SIMULATION"]["SCENARIO"]["scenario"]


def _read_comparison(eval_folder: str) -> list:
    path = _EVAL_ROOT / eval_folder / "comparison.csv"
    if not path.is_file():
        raise FileNotFoundError(f"comparison.csv mancante in {eval_folder} (EVALUATION non eseguita?)")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect(eval_folders: list) -> list:
    """Tabella lunga: una riga per (scenario, candidato) con le metriche di comparison.csv."""
    rows = []
    for folder in eval_folders:
        scenario = _scenario_of(folder)
        for r in _read_comparison(folder):
            rows.append({
                "scenario":              scenario,
                "eval_folder":           folder,
                "name":                  r["name"],
                "policy":                r["policy"],
                "model_folder":          r["model_folder"],
                "n_pipelines":           r.get("n_pipelines", ""),
                "n_completed":           r.get("n_completed", ""),
                "constellation_time_ms": _to_float(r["constellation_time_ms"]),
                "energy_satsas_J":       _to_float(r["energy_satsas_J"]),
                "energy_total_J":        _to_float(r["energy_total_J"]),
            })
    return rows


#######################################################################################################
# OUTPUT - CSV
#######################################################################################################

def _save_csv(path: Path, rows: list) -> None:
    fields = ["scenario", "eval_folder", "name", "policy", "model_folder",
              "n_pipelines", "n_completed",
              "constellation_time_ms", "energy_satsas_J", "energy_total_J"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[MULTI-EVAL] tabella combinata salvata: {path}")


#######################################################################################################
# OUTPUT - GRAFICI
#######################################################################################################

def _short(scenario: str) -> str:
    """Etichetta breve dello scenario: solo la foglia (scenario_0), non il path
    completo - il prefisso comune è identico per tutti e ruba spazio all'asse."""
    return scenario.rsplit("/", 1)[-1]


def _scenario_key(scenario: str):
    """Ordine numerico degli scenari (scenario_10 dopo scenario_9)."""
    tick = _tick(scenario)
    return (0, int(tick), "") if tick.isdigit() else (1, 0, tick)


def _tick(scenario: str) -> str:
    """Etichetta dell'asse x: il numero dello scenario, o la foglia senza trattini bassi."""
    leaf = _short(scenario)
    return leaf[len("scenario_"):] if leaf.startswith("scenario_") else leaf.replace("_", " ")


def _style(name: str, names: list) -> dict:
    """Etichetta, colore, marcatore e tratto del candidato."""
    mask = name.endswith(_MASK_SUFFIX)
    base = name[:-len(_MASK_SUFFIX)] if mask else name
    if base in _STYLES:
        label, color, symbol, dash = _STYLES[base]
    else:
        others = [n for n in names if n.removesuffix(_MASK_SUFFIX) not in _STYLES]
        label = base.replace("_", " ")
        color = _PALETTE[others.index(name) % len(_PALETTE)]
        symbol, dash = "circle", "solid"
    if mask:
        label, color, dash = label + " con maschera", _darker(color), "dash"
    elif base + _MASK_SUFFIX in names:
        dash = "solid"
    return {"label": label, "color": color, "symbol": symbol, "dash": dash}


def _darker(color: str, amount: float = .45) -> str:
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c * (1 - amount)) for c in (r, g, b))


def _rows_of(rows: list, name: str) -> dict:
    """Righe del candidato `name` indicizzate per scenario."""
    return {r["scenario"]: r for r in rows if r["name"] == name}


def _hover(label: str, ytitle: str) -> str:
    """Tooltip ricco: candidato, scenario, valore, pipeline completate, modello."""
    return (f"<b>{label}</b><br>Scenario: %{{customdata[3]}}<br>{ytitle}: %{{y:,.1f}}"
            "<br>Pipeline completate: %{customdata[0]}/%{customdata[1]}"
            "<br>Modello: %{customdata[2]}<extra></extra>")


def _traces_data(rows: list, scenarios: list, name: str, metric: str):
    """(posizioni, y, customdata) del candidato per il metric, ordinati come scenarios."""
    by_scenario = _rows_of(rows, name)
    x  = list(range(len(scenarios)))
    y  = [by_scenario.get(sc, {}).get(metric) for sc in scenarios]
    cd = [[by_scenario.get(sc, {}).get("n_completed", ""),
           by_scenario.get(sc, {}).get("n_pipelines", ""),
           by_scenario.get(sc, {}).get("model_folder") or by_scenario.get(sc, {}).get("policy", ""),
           _short(sc)]
          for sc in scenarios]
    return x, y, cd


_LEGEND_COLUMNS = 3
_WIDTH          = 1100


def _layout(fig, scenarios: list, names: list, title: str, ytitle: str, xrange: list) -> None:
    """Impaginazione comune, come nelle figure della tesi: grandezza nel titolo in alto, asse y con
    la sola dimensione e l'unita' di misura, legenda in alto a destra su tre colonne, migliaia
    separate da uno spazio, nessun bordo superiore o destro."""
    # Legenda su tre colonne riempite dall'alto verso il basso nell'ordine dei candidati, come nelle
    # figure della tesi, qualunque sia l'ordine di disegno.
    order  = [_style(n, names)["label"] for n in names]
    n_rows = -(-len(order) // _LEGEND_COLUMNS)
    for trace in fig.data:
        i = order.index(trace.name)
        trace.legendrank = (i % n_rows) * _LEGEND_COLUMNS + i // n_rows
    axis = dict(showline=True, linecolor="#000000", ticks="outside", zeroline=False)
    # Dimensioni fisse, nelle proporzioni delle figure della tesi (8,1 x 3,7 pollici): adattato alla
    # larghezza della finestra, il grafico si allungherebbe.
    fig.update_layout(
        autosize=False, width=_WIDTH, height=round(_WIDTH * 3.7 / 8.1),
        template="plotly_white", separators=", ",
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", xref="paper"),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom",
                    entrywidth=0.33, entrywidthmode="fraction"),
        margin=dict(l=80, r=30, t=110, b=60),
    )
    fig.update_xaxes(axis, title="Scenario di test", tickmode="array",
                     tickvals=list(range(len(scenarios))),
                     ticktext=[_tick(sc) for sc in scenarios], range=xrange)
    low, high = _yrange(fig)
    ticks = _ticks(low, high)
    # Lo zero compare una sola volta, nell'angolo: se l'asse y parte da zero, il suo zero non si
    # scrive.
    fig.update_yaxes(axis, title=dict(text=ytitle, standoff=18), automargin=True,
                     range=[low, high], tickmode="array", tickvals=ticks,
                     ticktext=["" if v == 0 else _number(v) for v in ticks])


def _yrange(fig) -> tuple:
    """Estremi dell'asse y: margine del 5% attorno ai dati, come nelle figure della tesi."""
    values = [v for trace in fig.data for v in (trace.y or ()) if v is not None]
    low, high = min(values), max(values)
    bars = any(trace.type == "bar" for trace in fig.data)
    labelled = any(trace.type == "bar" and trace.text is not None for trace in fig.data)
    pad = 0.05 * ((high - low) or abs(high) or 1)
    if bars:
        return 0, high + (0.12 * high if labelled else pad)
    start = low - pad
    return (0 if low >= 0 and start < 0 else start), high + pad


def _ticks(low: float, high: float, target: int = 8) -> list:
    """Tacche a passo regolare (1, 2, 2,5 o 5 per una potenza di dieci)."""
    import math
    raw  = (high - low) / target
    base = 10 ** math.floor(math.log10(raw))
    step = next(k * base for k in (1, 2, 2.5, 5, 10) if k * base >= raw)
    first = math.ceil(low / step) * step
    return [round(first + i * step, 9) for i in range(int((high - first) // step) + 1)]


def _number(value: float) -> str:
    """Numero con le migliaia separate da uno spazio e la virgola decimale."""
    text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return text.replace(",", " ").replace(".", ",")


def _bar_chart(rows: list, scenarios: list, names: list, metric: str, title: str, ytitle: str):
    """Barre raggruppate: per ciascuno scenario, una barra per candidato; il valore compare nel
    tooltip."""
    import plotly.graph_objects as go

    labelled = len(names) <= 2
    fig = go.Figure()
    for name in names:
        style = _style(name, names)
        x, y, cd = _traces_data(rows, scenarios, name, metric)
        text = [f"{v:,.0f}".replace(",", " ") if v is not None else "" for v in y] if labelled else None
        fig.add_bar(name=style["label"], x=x, y=y, marker_color=style["color"],
                    text=text, textposition="outside", textfont=dict(size=10), cliponaxis=False,
                    customdata=cd, hovertemplate=_hover(style["label"], ytitle))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0)
    _layout(fig, scenarios, names, title, ytitle, [-0.5, len(scenarios) - 0.5])
    return fig


def _line_chart(rows: list, scenarios: list, names: list, metric: str, title: str, ytitle: str):
    """Le "funzioni": ogni candidato è una curva che attraversa gli scenari, con il proprio
    marcatore e tratto."""
    import plotly.graph_objects as go

    # Le linee non continue stanno sopra quelle continue, con marcatori piu' piccoli; la baseline,
    # punteggiata, sta sopra tutte.
    order = {"solid": 0, "dash": 1, "dashdot": 1, "dot": 2}
    size  = {"solid": 9, "dash": 7, "dashdot": 7, "dot": 6}
    fig = go.Figure()
    for name in sorted(names, key=lambda n: order[_style(n, names)["dash"]]):
        style = _style(name, names)
        x, y, cd = _traces_data(rows, scenarios, name, metric)
        fig.add_scatter(name=style["label"], x=x, y=y, mode="lines+markers", cliponaxis=False,
                        line=dict(color=style["color"], width=2, dash=style["dash"]),
                        marker=dict(color=style["color"], size=size[style["dash"]], symbol=style["symbol"]),
                        customdata=cd, hovertemplate=_hover(style["label"], ytitle))
    _layout(fig, scenarios, names, title, ytitle, [0, len(scenarios) - 1])
    return fig


# C_E comprende il calcolo orbitale e tutte le trasmissioni, incluse quelle con GND.
_METRICS = [
    ("constellation_time_ms", "Tempo totale di completamento C<sub>T</sub>", "Tempo (ms)"),
    ("energy_satsas_J",       "Costo energetico C<sub>E</sub>",              "Energia (J)"),
]


def _save_html(path: Path, rows: list) -> None:
    scenarios = sorted({r["scenario"] for r in rows}, key=_scenario_key)
    names     = list(dict.fromkeys(r["name"] for r in rows))   # ordine di prima comparsa

    # Per ogni metrica: barre ("i grafici") e linee ("le funzioni"). plotly.js incluso una sola
    # volta, sul primo grafico della pagina.
    blocks, first = [], True
    for metric, title, ytitle in _METRICS:
        bar  = _bar_chart(rows, scenarios, names, metric, title, ytitle)
        line = _line_chart(rows, scenarios, names, metric, title, ytitle)
        blocks.append(bar.to_html(include_plotlyjs=first, full_html=False));  first = False
        blocks.append(line.to_html(include_plotlyjs=False, full_html=False))

    page = ('<!doctype html><html><head><meta charset="utf-8">'
            '<title>Confronto multi-scenario</title></head><body>'
            '<div style="font-family: system-ui, sans-serif; margin: 24px 40px; color: #222;">'
            '<h1 style="font-size:22px; margin: 10px 0 4px;">Confronto multi-scenario</h1>'
            f'<p style="color:#666; margin: 0 0 14px;">Scenari di test: {", ".join(_tick(s) for s in scenarios)} '
            f'&nbsp;·&nbsp; metodi: {", ".join(_style(n, names)["label"] for n in names)}</p>'
            + "".join(blocks) +
            '</div></body></html>')

    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[MULTI-EVAL] report salvato: {path}")


#######################################################################################################
# ENTRY POINT
#######################################################################################################

def aggrega(eval_folders: list, out_root) -> None:
    """Panoramica dei confronti gia' prodotti: tabella combinata e grafici
    tempo/energia per scenario, una serie per candidato."""
    from pathlib import Path as _P
    out_root = _P(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print()
    print(f"[PANORAMICA] {len(eval_folders)} scenari")
    rows = _collect(eval_folders)
    _save_csv(out_root / "multi_comparison.csv", rows)
    _save_html(out_root / "comparison_scenarios.html", rows)
