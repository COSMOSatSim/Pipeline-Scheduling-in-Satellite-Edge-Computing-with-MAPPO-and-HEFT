"""Confronto tra policy/modelli sullo stesso scenario."""
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import json5

from utility.file_utility import DATA_PATH, RESULTS_PATH

_ROOT      = Path(__file__).resolve().parents[1]     # SECMotionModel/
_EVAL_ROOT = Path(RESULTS_PATH) / "evaluation"


#######################################################################################################
# RUN
#######################################################################################################

def _derived_config(config: dict, run: dict, scenario: str, run_dir: Path) -> dict:
    """Config del singolo run: il config corrente con scenario, policy/modello
    e cartella risultati del candidato; tutti i branch precoci disattivati."""
    derived = copy.deepcopy(config)

    # Il run del candidato è una simulazione pura: senza questo, il sottoprocesso rilancerebbe
    # l'evaluation all'infinito.
    derived["CONTROL_PANEL"] = {"run": "simulation"}

    simulation = derived.setdefault("SIMULATION", {})
    simulation.setdefault("SCENARIO", {})["scenario"] = scenario

    scheduler = simulation["SCHEDULER"]["policy_scheduler"]
    scheduler["selected"] = run["policy"]
    if run["policy"] == "mappo":
        scheduler.setdefault("mappo", {})["model_folder"] = run["model"]

    derived["results_dir"] = str(run_dir)
    return derived


def _launch(run: dict, cfg_path: Path) -> None:
    """Esegue main.py in un sottoprocesso con il config derivato (SIM_CONFIG)."""
    env = dict(os.environ, SIM_CONFIG=str(cfg_path), PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-u", str(_ROOT / "main.py")],
                            cwd=str(_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Run '{run['name']}' fallito (exit code {result.returncode}).")


#######################################################################################################
# METRICHE
#######################################################################################################

def _collect_metrics(run_dir: Path, run: dict) -> dict:
    """Metriche aggregate del run dai suoi CSV (results_summary + results_tasks)."""
    summary_path = run_dir / "results_summary.csv"
    tasks_path   = run_dir / "results_tasks.csv"
    if not summary_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(f"Risultati mancanti per il run '{run['name']}' in {run_dir}")

    with open(summary_path, newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    makespans = [float(r["makespan_ms"]) for r in summary_rows if r["makespan_ms"] != "N/A"]

    # Energia per tipo di dispositivo, tenendo distinta l'esecuzione dalla trasmissione.
    tipi = ("SAT", "SAS", "GND")
    esecuzione = {t: 0.0 for t in tipi}
    trasmissione = {t: 0.0 for t in tipi}

    starts, ends = [], []
    with open(tasks_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["start_time_ms"] != "N/A":
                starts.append(float(r["start_time_ms"]))
            if r["end_time_ms"] != "N/A":
                ends.append(float(r["end_time_ms"]))
            tipo = r["device_type"]
            if tipo in esecuzione:
                esecuzione[tipo] += float(r["energy_cpu_J"])
                # I risultati prodotti prima dell'introduzione della colonna non la contengono: si
                # ripiega sul valore lato destinatario, che ne differisce per meno di un millesimo
                # del totale.
                trasmissione[tipo] += float(r.get("energy_tx_J")
                                            or r["energy_net_J"])

    # C_E come lo definisce il modello: l'elaborazione sui soli dispositivi in orbita, piu' le
    # trasmissioni fra dispositivi distinti, i collegamenti con la stazione di terra compresi.
    energy_CE = (esecuzione["SAT"] + esecuzione["SAS"]
                 + sum(trasmissione.values()))
    energy_all = energy_CE + esecuzione["GND"]

    model = run.get("model", "") if run["policy"] == "mappo" else ""
    metriche = {
        "name":                  run["name"],
        "policy":                run["policy"],
        "model_folder":          model,
        "n_pipelines":           len(summary_rows),
        "n_completed":           len(makespans),
        "makespan_avg_ms":       sum(makespans) / len(makespans) if makespans else None,
        "makespan_max_ms":       max(makespans) if makespans else None,
        "constellation_time_ms": (max(ends) - min(starts)) if starts and ends else None,
        "energy_satsas_J":       energy_CE,
        "energy_total_J":        energy_all,
        "t_end_ms":              max(ends) if ends else None,   # fondo scala del run (asse X comune)
    }
    for t in tipi:
        metriche["energy_exec_%s_J" % t] = esecuzione[t]
        metriche["energy_tran_%s_J" % t] = trasmissione[t]
    return metriche


def _fmt(value, suffix="") -> str:
    return f"{value:.4f}{suffix}" if value is not None else "N/A"


def _print_comparison(scenario: str, rows: list) -> None:
    """Tabella di confronto a console (stesso stile dei riepiloghi post-run)."""
    policies = [f"mappo: {m['model_folder']}" if m["model_folder"] else m["policy"]
                for m in rows]
    name_w = max(len("CANDIDATO"), *(len(m["name"]) for m in rows))
    pol_w  = max(len("POLICY / MODELLO"), *(len(p) for p in policies))
    ene_w  = len("COSTO ENERGETICO C_E")
    width  = name_w + pol_w + 9 + 13 + ene_w + 15 + 5 * 3   # colonne + separatori

    print(f"\n{'='*60}")
    print("CONFRONTO CANDIDATI")
    print(f"{'='*60}")
    print(f"  Scenario di test : {scenario}\n")

    sep = (f"  {'-'*name_w}-+-{'-'*pol_w}-+-{'-'*9}-+-{'-'*13}-+-"
           f"{'-'*ene_w}-+-{'-'*15}")
    print(f"  {'-'*width}")
    print(f"  {'CANDIDATO':^{name_w}} | {'POLICY / MODELLO':^{pol_w}} | {'PIPELINE':^9} | "
          f"{'TEMPO TOTALE':^13} | {'COSTO ENERGETICO C_E':^{ene_w}} | "
          f"{'ENERGIA TOTALE':^15}")
    print(sep)
    for m, policy in zip(rows, policies):
        pipes = f"{m['n_completed']}/{m['n_pipelines']}"
        print(f"  {m['name']:<{name_w}} | {policy:<{pol_w}} | {pipes:^9} | "
              f"{_fmt(m['constellation_time_ms']):>13} | "
              f"{_fmt(m['energy_satsas_J'], 'J'):>{ene_w}} | "
              f"{_fmt(m['energy_total_J'], 'J'):>15}")
    print(f"  {'-'*width}")
    print("  (tempi in ms; PIPELINE = completate/totali)")


def _save_comparison_csv(path: Path, rows: list) -> None:
    fields = ["name", "policy", "model_folder", "n_pipelines", "n_completed",
              "makespan_avg_ms", "makespan_max_ms", "constellation_time_ms",
              "energy_satsas_J", "energy_total_J",
              "energy_exec_SAT_J", "energy_tran_SAT_J",
              "energy_exec_SAS_J", "energy_tran_SAS_J",
              "energy_exec_GND_J", "energy_tran_GND_J"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[EVAL] tabella di confronto salvata: {path}")


#######################################################################################################
# REPORT HTML
#######################################################################################################

# Asset della pagina in evaluation/view/ (stile e logica client: selettore A/B, tabella-duello con
# vincitore/perdente per campo, resize dei Gantt), iniettati inline così la pagina resta
# autocontenuta.
_VIEW     = Path(__file__).with_name("view")
_CSS_PATH = _VIEW / "report.css"
_JS_PATH  = _VIEW / "report.js"


def _comparison_table_html(rows: list) -> str:
    head = ("<tr><th>Candidato</th><th>Policy / modello</th><th>Pipeline</th>"
            "<th>Tempo totale di completamento C_T</th><th>Costo energetico C_E</th>"
            "<th>Energia totale</th></tr>")
    trs = ""
    for m in rows:
        policy = f"mappo: {m['model_folder']}" if m["model_folder"] else m["policy"]
        trs += (f'<tr><td class="name">{m["name"]}</td><td>{policy}</td>'
                f'<td class="num">{m["n_completed"]}/{m["n_pipelines"]}</td>'
                f'<td class="num">{_fmt(m["constellation_time_ms"], "ms")}</td>'
                f'<td class="num">{_fmt(m["energy_satsas_J"], "J")}</td>'
                f'<td class="num">{_fmt(m["energy_total_J"], "J")}</td></tr>')
    return f'<table class="cmp"><thead>{head}</thead><tbody>{trs}</tbody></table>'

def _save_comparison_html(out_root: Path, scenario: str, runs: list, rows: list) -> None:
    """Report unico: tabella di confronto in cima + confronto A vs B sotto (due select scelgono i
    candidati: Simulation Info affiancate, Gantt uno sotto l'altro sulla STESSA scala temporale)."""
    from simulation_utility.utility.gantt_html import build_gantt_fragment
    from simulation_utility.utility.report_html import build_sim_info_html

    # Fondo scala comune a tutti i Gantt: senza, le barre non sono confrontabili.
    ends = [m["t_end_ms"] for m in rows if m["t_end_ms"] is not None]
    xmax = max(ends) if ends else None

    labels  = [f"mappo: {r['model']}" if r["policy"] == "mappo" else r["policy"]
               for r in runs]
    options = "".join(f'<option value="{i}">{r["name"]}</option>' for i, r in enumerate(runs))
    sel_b   = min(1, len(runs) - 1)     # default: primo vs secondo
    picker  = (f'<div class="eval-pick">Confronta: '
               f'<select id="eval-sel-a" onchange="evalUpdateCompare()">{options}</select> vs '
               f'<select id="eval-sel-b" onchange="evalUpdateCompare()">{options}</select></div>')
    picker  = picker.replace(f'<option value="{sel_b}">', f'<option value="{sel_b}" selected>', 1)

    infos, gantts = [], []
    for i, r in enumerate(runs):
        run_dir   = out_root / r["name"]
        tasks_csv = str(run_dir / "results_tasks.csv")
        with open(run_dir / "run_config.json5", encoding="utf-8") as f:
            run_config = json5.load(f)
        infos.append(
            f'<div class="eval-cand-info" style="display:none">'
            f'<p class="cand-title">{r["name"]} - {labels[i]}'
            f'<a href="{r["name"]}/gantt.html">riepiloghi completi</a></p>'
            + build_sim_info_html(run_config, tasks_csv) + '</div>')
        # plotly.js una sola volta, sul primo Gantt della pagina
        gantts.append(
            '<div class="eval-cand-gantt" style="display:none">'
            + build_gantt_fragment(tasks_csv, include_plotlyjs=(i == 0), xmax=xmax,
                                   title=f"Gantt - {r['name']} ({labels[i]})") + '</div>')

    # Dati per la tabella-duello (le due righe A/B con vincitore/perdente colorati) + logica client,
    # iniettati inline (pagina autocontenuta).
    duel_rows = [dict(m, label=labels[i]) for i, m in enumerate(rows)]
    data_js   = f"<script>var EVAL_ROWS = {json.dumps(duel_rows)};</script>"
    page_js   = f"<script>\n{_JS_PATH.read_text(encoding='utf-8')}</script>"
    page_css  = f"<style>\n{_CSS_PATH.read_text(encoding='utf-8')}</style>"

    page = ('<!doctype html><html><head><meta charset="utf-8">'
            '<title>Confronto candidati</title></head><body>'
            + page_css + '<div class="eval-wrap">'
            + "<h1>Confronto candidati</h1>"
            + f'<p class="sub">Scenario di test: <b>{scenario}</b></p>'
            + _comparison_table_html(rows)
            + "</div>" + picker
            + f'<div class="eval-infos">{"".join(infos)}</div>'
            + '<div class="eval-wrap eval-duel-wrap" id="eval-duel"></div>'
            + f'<div class="eval-gantts">{"".join(gantts)}</div>'
            + data_js + page_js + "</body></html>")

    path = out_root / "comparison.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[EVAL] report di confronto salvato: {path}")

#######################################################################################################
# ENTRY POINT
#######################################################################################################

def _carica_suite(nome: str) -> dict:
    """Legge config/evaluation/<nome>.json5: candidati, scenari, cartella."""
    percorso = _ROOT / "config" / "evaluation" / f"{nome}.json5"
    if not percorso.is_file():
        disponibili = sorted(x.stem for x in (_ROOT / "config" / "evaluation").glob("*.json5"))
        raise FileNotFoundError(
            f"Suite '{nome}' non trovata in config/evaluation/. "
            f"Disponibili: {', '.join(disponibili) or 'nessuna'}.")
    with open(percorso, encoding="utf-8") as f:
        return json5.load(f)


def _valida(candidati: list) -> None:
    nomi = [c["name"] for c in candidati]
    if len(set(nomi)) != len(nomi):
        raise ValueError(f"candidates: nomi duplicati {nomi}")
    for c in candidati:
        if c["policy"] == "mappo" and not c.get("model"):
            raise ValueError(f"candidates: '{c['name']}' e' mappo ma non indica `model`.")


def _per_scenario(candidati: list, scenario: str) -> list:
    """Sostituisce {scenario} nei modelli con il nome dello scenario in esame."""
    corto = scenario.rsplit("/", 1)[-1]
    risolti = []
    for c in candidati:
        c = dict(c)
        if c.get("model"):
            c["model"] = c["model"].replace("{scenario}", corto)
        risolti.append(c)
    return risolti


def _modello_disponibile(candidato: dict) -> bool:
    """True se il modello del candidato esiste su disco."""
    if candidato["policy"] != "mappo":
        return True
    modello = candidato.get("model") or ""
    return (Path(DATA_PATH) / "models" / "mappo" / modello).is_dir()


def _confronta(config: dict, candidati: list, scenario: str, out_root: Path) -> None:
    """Esegue i candidati su uno scenario e produce tabella, csv e report."""
    out_root.mkdir(parents=True, exist_ok=True)

    assenti = [r["name"] for r in candidati if not _modello_disponibile(r)]
    if assenti:
        print("[EVALUATION] modello non disponibile, candidati saltati: %s"
              % ", ".join(assenti))
        candidati = [r for r in candidati if _modello_disponibile(r)]
    if not candidati:
        print("[EVALUATION] nessun candidato valutabile su questo scenario.")
        return

    for i, r in enumerate(candidati):
        run_dir = out_root / r["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "run_config.json5"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json5.dump(_derived_config(config, r, scenario, run_dir), f, indent=2)

        label = f"mappo: {r['model']}" if r["policy"] == "mappo" else r["policy"]
        print()
        print(f"{'='*60}")
        print(f"[EVALUATION] {r['name']} ({label})  -  candidato {i + 1}/{len(candidati)}")
        print(f"{'='*60}")
        _launch(r, cfg_path)

    rows = [_collect_metrics(out_root / r["name"], r) for r in candidati]
    _print_comparison(scenario, rows)
    _save_comparison_csv(out_root / "comparison.csv", rows)
    _save_comparison_html(out_root, scenario, candidati, rows)


def run(config: dict) -> None:
    """Esegue le suite indicate da EVALUATION.suite, una dopo l'altra."""
    richieste = config["EVALUATION"]["suite"]
    if isinstance(richieste, str):
        richieste = [richieste]

    if len(richieste) > 1:
        print(f"[EVALUATION] {len(richieste)} suite in coda: {', '.join(richieste)}")
    for i, nome in enumerate(richieste, 1):
        if len(richieste) > 1:
            print()
            print("#" * 70)
            print(f"# SUITE {i}/{len(richieste)}: {nome}")
            print("#" * 70)
        _esegui_suite(config, nome)


def _esegui_suite(config: dict, nome_suite: str) -> None:
    """Confronta i candidati della suite sugli scenari che indica."""
    suite      = _carica_suite(nome_suite)
    candidati  = suite["candidates"]
    scenari    = suite["scenarios"]
    radice     = _EVAL_ROOT / suite["output"]

    _valida(candidati)

    print(f"[EVALUATION] suite '{nome_suite}': {len(scenari)} scenari, "
          f"{len(candidati)} candidati")
    print(f"[EVALUATION] output: {radice}")

    cartelle = []
    for scenario in scenari:
        corto    = scenario.rsplit("/", 1)[-1]
        out_root = radice / corto
        cartelle.append(f"{suite['output']}/{corto}")

        if (out_root / "comparison.csv").is_file():
            print(f"[EVALUATION] {corto:<14} gia' valutato, salto")
            continue

        print(f"[EVALUATION] {corto:<14} da valutare")
        _confronta(config, _per_scenario(candidati, scenario), scenario, out_root)

    if len(scenari) > 1:
        from evaluation.multi_evaluate import aggrega
        aggrega(cartelle, radice)
