"""Curve di training interattive (HTML) dal CSV."""
import csv as _csv
import os

_PLOTLY_WARNED = False   # avvisa una sola volta se plotly non è installato (no spam in training)


def _moving_average(values, window=50):
    """Media mobile causale (stessa lunghezza dell'input); finestra che cresce all'inizio."""
    from collections import deque
    out, buf = [], deque(maxlen=window)
    for v in values:
        buf.append(v)
        out.append(sum(buf) / len(buf))
    return out


def _read_log(csv_path):
    cols = {k: [] for k in ("episode", "reward", "C_T", "C_E", "actor_loss", "critic_loss", "SAT", "SAS", "GND")}
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            try:
                cols["episode"].append(int(r["episode"]))
            except (TypeError, ValueError, KeyError):
                continue
            for k in ("reward", "C_T", "C_E", "actor_loss", "critic_loss", "SAT", "SAS", "GND"):
                try:
                    cols[k].append(float(r.get(k, "nan")))
                except (TypeError, ValueError):
                    cols[k].append(float("nan"))
    return cols


def generate_training_html(csv_path: str, output_html: str):
    """Genera la dashboard HTML interattiva dal training_log.csv."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        global _PLOTLY_WARNED
        if not _PLOTLY_WARNED:
            print(f"[training_html] plotly non disponibile, HTML saltato: {e}")
            _PLOTLY_WARNED = True
        return None

    if not os.path.isfile(csv_path):
        print(f"[training_html] CSV non trovato: {csv_path}")
        return None

    c = _read_log(csv_path)
    eps = c["episode"]
    if not eps:
        print("[training_html] CSV vuoto.")
        return None

    # Asse 0-based, come la console del training (stampa 'ep N' a partire da 0).
    x_axis = eps

    fig = make_subplots(
        rows=2, cols=3, shared_xaxes=False, vertical_spacing=0.13,
        subplot_titles=("Reward", "C<sup>T</sup> (ms)", "C<sup>E</sup> (Wms)",
                        "Actor loss", "Critic loss", "Distribuzione azioni"),
    )

    # (riga, col, serie, colore, con_media_mobile)
    panels = [
        (1, 1, "reward",      "#4472C4", True),
        (1, 2, "C_T",         "#ED7D31", True),
        (1, 3, "C_E",         "#70AD47", True),
        (2, 1, "actor_loss",  "#8E44AD", False),
        (2, 2, "critic_loss", "#C0392B", False),
    ]
    for row, col, key, color, smooth in panels:
        data = c[key]
        fig.add_trace(go.Scatter(x=x_axis, y=data, mode="lines", name=key,
                                 line=dict(color=color, width=1), opacity=0.30,
                                 showlegend=False), row=row, col=col)
        if smooth:
            fig.add_trace(go.Scatter(x=x_axis, y=_moving_average(data), mode="lines",
                                     name=f"{key} (media 50)", line=dict(color=color, width=2),
                                     showlegend=False), row=row, col=col)

    # Distribuzione azioni SAT/SAS/GND (conteggi per episodio), letta come gli altri pannelli: la
    # serie grezza resta sullo sfondo e la media mobile porta la tendenza, che a occhio nudo il
    # rumore per episodio nasconde.
    for key, color in (("SAT", "#ff7f0e"), ("SAS", "#1f77b4"), ("GND", "#2ca02c")):
        fig.add_trace(go.Scatter(x=x_axis, y=c[key], mode="lines", name=key,
                                 line=dict(color=color, width=1), opacity=0.30,
                                 showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=x_axis, y=_moving_average(c[key]), mode="lines",
                                 name=key, line=dict(color=color, width=2)),
                      row=2, col=3)

    fig.update_xaxes(title_text="Episode")        # asse Episode sotto ogni grafico (sono indipendenti)
    fig.update_yaxes(exponentformat="power")      # notazione potenza di 10 (×10ⁿ) invece di M/k
    fig.update_layout(
        title=f"SEC-MPH Training - episodio {eps[-1]}",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)
    print(f"[training_html] salvato: {output_html}")
    return output_html
