"""Gantt interattivo (HTML) dello scheduling, generato dal CSV."""
import csv as _csv
import os

from utility.log_colors import hex_for_device

#######################################################################################################
# CONFIGURAZIONE - ritocca qui l'aspetto del Gantt
#######################################################################################################

# Colori e label per missione.
_MISSION_COLORS = {
    "IS": "#9467bd",   # imagery slicing      - viola
    "CD": "#1f77b4",   # cloud detection      - blu
    "GC": "#2ca02c",   # geometric correction - verde
    "CF": "#ff7f0e",   # classification       - arancione
}
_MISSION_LABELS = {
    "IS": "Imagery slicing",
    "CD": "Cloud detection",
    "GC": "Geometric correction",
    "CF": "Classification",
}
_DEFAULT_COLOR = "#7c8081"                 # missione sconosciuta

# Ordine delle righe (in alto i satelliti comuni).
_TYPE_ORDER = {"SAT": 0, "SAS": 1, "GND": 2}


def _row_label(device_id: str, device_name: str) -> str:
    """Label di riga del Gantt: tag [scenario_id] colorato + nome reale del satellite."""
    color = hex_for_device(device_id)
    return f'<span style="color:{color}">[{device_id}]</span> {device_name}'

# Dimensioni del grafico.
ROW_HEIGHT_PX  = 40                        # altezza per riga-device
HEIGHT_PADDING = 170                       # spazio fisso per titolo/legenda/assi
MIN_HEIGHT_PX  = 400
X_RANGE_PAD    = 1.02                      # margine a destra sull'asse tempo
MARGIN         = dict(l=90, r=30, t=70, b=50)
BAR_GAP        = 0.25

# Bordo nero delle barre-task: pieno per i task "normali", assente per quelli molto piccoli (sotto
# il pixel il contorno diventerebbe una lineetta-fantasma).
BORDER_COLOR    = "black"
BORDER_WIDTH    = 0.2                       # task "normali"
SMALL_TASK_FRAC = 0.02                      # "molto piccola" se durata < SMALL_TASK_FRAC · durata max
SMALL_BORDER    = 0.00                      # bordo per i task molto piccoli (0 = nessun contorno)

# Finestre offline (fasce di sfondo per-riga).
OFFLINE_COLOR   = "#5a5f63"                 # grigio scuro, ben visibile dietro le barre
OFFLINE_OPACITY = 0.45
OFFLINE_HALF_H  = 0.45                      # mezza altezza della fascia (in unità di riga)


#######################################################################################################
# LETTURA CSV
#######################################################################################################

def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mission_key(node_id, operation, mission_type):
    """Prefisso missione (IS/CD/GC/CF) dal node_id, con fallback su operation/mission_type."""
    base = node_id or operation or mission_type or ""
    return base.split("-")[0]


def _read_tasks(tasks_csv):
    """Legge results_tasks.csv -> lista di dict (un task per riga, solo con tempi validi)."""
    rows = []
    with open(tasks_csv, newline="") as f:
        for r in _csv.DictReader(f):
            start, end = _safe_float(r.get("start_time_ms")), _safe_float(r.get("end_time_ms"))
            if start is None or end is None:
                continue
            device_id, device_type = r.get("device_id", ""), r.get("device_type", "")
            device_name = r.get("device_name") or device_id      # nome reale del satellite
            rows.append({
                "device_id":   device_id,
                "device_type": device_type,
                "device_name": device_name,
                "label":       _row_label(device_id, device_name),  # [scenario_id] nome reale
                "dag":         r.get("dag_id", ""),                  # pipeline di appartenenza
                "node_id":     r.get("node_id", ""),
                "operation":   r.get("operation", ""),
                "mission":     _mission_key(r.get("node_id", ""), r.get("operation", ""), r.get("mission_type", "")),
                "start":       start,
                "end":         end,
                "d_cpu":       _safe_float(r.get("d_cpu_ms")),
            })
    return rows


def _read_offline(offline_csv):
    """Legge offline_windows.csv se presente -> [(label_riga, start, end), ...]."""
    if not offline_csv or not os.path.isfile(offline_csv):
        return []
    windows = []
    with open(offline_csv, newline="") as f:
        for r in _csv.DictReader(f):
            start, end = _safe_float(r.get("start")), _safe_float(r.get("end"))
            if start is None or end is None:
                continue
            did   = r.get("device_id", "")
            name  = r.get("device_name") or did
            label = _row_label(did, name)
            windows.append((label, start, end))
    return windows


def _read_devices(devices_csv):
    """Legge devices.csv (TUTTI i device dello scenario, anche senza task) -> lista di (device_type,
    device_id, device_name)."""
    if not devices_csv or not os.path.isfile(devices_csv):
        return None
    devs = []
    with open(devices_csv, newline="") as f:
        for r in _csv.DictReader(f):
            did = r.get("device_id", "")
            devs.append((r.get("device_type", ""), did, r.get("device_name") or did))
    return devs


def _device_order(rows, all_devices=None):
    """Etichette-riga ordinate per tipo (SAT, SAS, GND) e poi per id."""
    devs = {(r["device_type"], r["device_id"]) for r in rows}
    names = {r["device_id"]: r["device_name"] for r in rows}
    if all_devices:
        devs |= {(t, i) for t, i, _n in all_devices}
        for t, i, n in all_devices:
            names.setdefault(i, n)
    ordered = sorted(devs, key=lambda td: (_TYPE_ORDER.get(td[0], 9), td[1]))
    return [_row_label(dev_id, names.get(dev_id, dev_id)) for _t, dev_id in ordered]


#######################################################################################################
# DISEGNO - barre dei task
#######################################################################################################

def _add_task_bars(fig, rows):
    """Una barra orizzontale (start->end) per task, un trace per missione (legenda/toggle)."""
    max_dur   = max([r["end"] - r["start"] for r in rows] + [1e-9])
    small_thr = SMALL_TASK_FRAC * max_dur

    for mk in ("IS", "CD", "GC", "CF"):
        sub = [r for r in rows if r["mission"] == mk]
        if not sub:
            continue
        durs   = [max(r["end"] - r["start"], 0.0) for r in sub]
        line_w = [BORDER_WIDTH if d >= small_thr else SMALL_BORDER for d in durs]
        fig.add_bar(
            name=_MISSION_LABELS.get(mk, mk), orientation="h", legendgroup=mk,
            y=[r["label"] for r in sub],
            base=[r["start"] for r in sub],
            x=durs,
            marker=dict(color=_MISSION_COLORS.get(mk, _DEFAULT_COLOR),
                        line=dict(color=BORDER_COLOR, width=line_w)),
            customdata=[[r["node_id"], r["device_id"], r["start"], r["end"],
                         r["d_cpu"], r["dag"]] for r in sub],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "pipeline: %{customdata[5]}<br>"
                "device: %{customdata[1]}<br>"
                "start: %{customdata[2]:.2f} ms<br>"
                "end: %{customdata[3]:.2f} ms<br>"
                "d_cpu: %{customdata[4]:.2f} ms<extra></extra>"
            ),
        )


#######################################################################################################
# DISEGNO - fasce offline
#######################################################################################################

def _add_row_anchors(fig, go, cat_array):
    """Trace invisibile con un punto per ogni riga-device: forza Plotly a disegnare SEMPRE tutte le
    righe (anche i device senza barre, es."""
    fig.add_trace(go.Scatter(
        x=[0] * len(cat_array), y=list(cat_array),
        mode="markers", marker=dict(opacity=0, size=1),
        showlegend=False, hoverinfo="skip",
    ))


def _add_offline_bands(fig, go, offline, row_idx):
    """Fasce offline come rettangoli di sfondo (layer='below'): visibili DIETRO le barre dei task e
    senza rubare l'hover."""
    drawn = False
    for label, ws, we in offline:
        if label in row_idx:
            i = row_idx[label]
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=ws, x1=we, y0=i - OFFLINE_HALF_H, y1=i + OFFLINE_HALF_H,
                          fillcolor=OFFLINE_COLOR, opacity=OFFLINE_OPACITY,
                          line_width=0, layer="below")
            drawn = True
    if drawn:   # le shape non vanno in legenda: trace fittizio solo per l'etichetta
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Offline",
                                 marker=dict(size=10, color=OFFLINE_COLOR, opacity=0.4)))


#######################################################################################################
# MAIN
#######################################################################################################

_DEFAULT_TITLE = "Scheduling pipeline - Gantt interattivo"


def _build_figure(go, tasks_csv, offline_csv, devices_csv, title, xmax_override=None):
    """Costruisce la figura Plotly del Gantt dai CSV; None se non generabile."""
    if not os.path.isfile(tasks_csv):
        print(f"[gantt_html] CSV non trovato: {tasks_csv}")
        return None

    rows = _read_tasks(tasks_csv)
    if not rows:
        print("[gantt_html] nessun task con tempi validi nel CSV.")
        return None

    # --- Struttura righe (device, TUTTI quelli dello scenario) e finestre offline ---
    all_devices = _read_devices(devices_csv)
    order     = _device_order(rows, all_devices)
    offline   = _read_offline(offline_csv)
    cat_array = list(reversed(order))                  # categoryarray (indice 0 = riga in basso)
    row_idx   = {label: i for i, label in enumerate(cat_array)}

    # --- Tracce ---
    fig = go.Figure()
    _add_row_anchors(fig, go, cat_array)     # una riga per OGNI device, anche senza barre
    _add_task_bars(fig, rows)
    _add_offline_bands(fig, go, offline, row_idx)

    # --- Layout: altezza ∝ n. device, larghezza responsive (0 a sinistra) ---
    xmax = xmax_override or max([r["end"] for r in rows] + [1.0])
    fig.update_layout(
        title=title,
        barmode="overlay",
        height=max(MIN_HEIGHT_PX, ROW_HEIGHT_PX * len(order) + HEIGHT_PADDING),
        autosize=True,
        margin=MARGIN,
        xaxis=dict(title="Tempo (ms)", range=[0, xmax * X_RANGE_PAD],
                   showline=True, linecolor="black", linewidth=1,
                   ticks="outside", tickcolor="black",
                   tick0=0, zeroline=False, rangemode="nonnegative"),
        yaxis=dict(title="Dispositivi", categoryorder="array", categoryarray=cat_array,
                   showline=True, linecolor="black", linewidth=1),
        bargap=BAR_GAP,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def build_gantt_fragment(tasks_csv: str, offline_csv: str = None, devices_csv: str = None,
                         include_plotlyjs: bool = False, title: str = _DEFAULT_TITLE,
                         xmax: float = None) -> str:
    """Gantt come frammento HTML (solo il <div> del grafico, niente <html>/<body>), per comporre
    pagine con più Gantt (es. il confronto della valutazione). plotly.js va incluso UNA volta per
    pagina: include_plotlyjs=True sul primo frammento, False sugli altri."""
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print(f"[gantt_html] plotly non disponibile: {e}")
        return ""

    offline_csv = offline_csv or os.path.join(os.path.dirname(tasks_csv), "offline_windows.csv")
    devices_csv = devices_csv or os.path.join(os.path.dirname(tasks_csv), "devices.csv")

    fig = _build_figure(go, tasks_csv, offline_csv, devices_csv, title, xmax_override=xmax)
    if fig is None:
        return ""
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False)


def generate_gantt_html(tasks_csv: str, output_html: str = None, offline_csv: str = None,
                        devices_csv: str = None, extra_html: str = None):
    """Genera il Gantt interattivo HTML dai CSV dei risultati."""
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print(f"[gantt_html] plotly non disponibile: {e}")
        return None

    output_html = output_html or os.path.join(os.path.dirname(tasks_csv), "gantt.html")
    offline_csv = offline_csv or os.path.join(os.path.dirname(tasks_csv), "offline_windows.csv")
    devices_csv = devices_csv or os.path.join(os.path.dirname(tasks_csv), "devices.csv")

    fig = _build_figure(go, tasks_csv, offline_csv, devices_csv, _DEFAULT_TITLE)
    if fig is None:
        return None

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    if extra_html:
        # Grafico + tabelle di riepilogo sotto, in un'unica pagina.
        fig_html = fig.to_html(include_plotlyjs=True, full_html=False)
        page = ('<!doctype html><html><head><meta charset="utf-8">'
                '<title>Gantt pipeline</title></head><body>'
                f'{fig_html}{extra_html}</body></html>')
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(page)
    else:
        fig.write_html(output_html, include_plotlyjs=True, full_html=True)
    print(f"[gantt_html] salvato: {output_html}")
    return output_html
