"""Manager del salvataggio risultati della simulazione a pipeline."""
import os

from simulation_utility.utility.csv import save_devices_csv, save_offline_windows_csv, save_results_csv
from simulation_utility.utility.gantt_html import generate_gantt_html
from simulation_utility.utility.report_html import build_report_html


def save_results_sections(config: dict, devices: dict, satellite_pipelines: list):
    """Salva gli output abilitati dalla configurazione (CSV + Gantt)."""
    csv_name = config.get("csv_name", {}).get("name", "")
    out_dir  = os.path.dirname(csv_name) if csv_name else os.path.join("data", "results")

    save_config = config.get("SIMULATION", {}).get("SAVE", {})
    saved_results = False
    if save_config.get("save_results", True):
        save_results_csv(satellite_pipelines, out_dir)
        saved_results = True

    if save_config.get("save_gantt", False):
        # Gantt interattivo HTML (da CSV): assicura gli input, poi renderizza.
        if not saved_results:
            save_results_csv(satellite_pipelines, out_dir)
        save_devices_csv(devices, out_dir)
        save_offline_windows_csv(devices, out_dir)
        # Tabelle di riepilogo (le stesse della console) embeddate sotto il Gantt.
        report = build_report_html(config, os.path.join(out_dir, "results_tasks.csv"))
        generate_gantt_html(tasks_csv=os.path.join(out_dir, "results_tasks.csv"),
                            output_html=os.path.join(out_dir, "gantt.html"),
                            offline_csv=os.path.join(out_dir, "offline_windows.csv"),
                            devices_csv=os.path.join(out_dir, "devices.csv"),
                            extra_html=report)
