"""Writer CSV dei risultati della simulazione a pipeline."""
import csv
import os

from utility.converter_utility import s_to_ms

#######################################################################################################
# RESULTS
#######################################################################################################

def _ms(t_s):
    """Tempo s -> stringa in ms ('N/A' se assente)."""
    return f"{s_to_ms(t_s):.4f}" if t_s is not None else "N/A"


def save_results_csv(dags: list, output_path: str):
    """
    Salva i risultati della simulazione in due CSV:
    - results_tasks.csv   : una riga per TaskChunk
    - results_summary.csv : una riga per pipeline (makespan, energia)
    """
    os.makedirs(output_path, exist_ok=True)

    # --- CSV tasks ---
    tasks_path = os.path.join(output_path, "results_tasks.csv")
    with open(tasks_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dag_id", "mission_type", "node_id", "operation",
            "device_id", "device_name", "device_type", "d_cpu_ms",
            "start_time_ms", "end_time_ms", "energy_cpu_J", "energy_net_J",
            "energy_tx_J",
        ])
        for dag in dags:
            for task in dag.task_chunks.values():
                device = task.assigned_device
                writer.writerow([
                    dag.dag_id,
                    dag.mission_type,
                    task.node_id,
                    task.operation,
                    device.id if device else "N/A",
                    device.raw.name if device else "N/A",    # nome reale (topologia)
                    device.device_type if device else "N/A",
                    f"{task.d_cpu:.4f}" if task.d_cpu is not None else "N/A",
                    _ms(task.start_time),
                    _ms(task.end_time),
                    f"{task.energy_cpu_J:.6f}",
                    f"{task.energy_net_J:.6f}",
                    f"{task.energy_tx_J:.6f}",
                ])

    # --- CSV summary ---
    summary_path = os.path.join(output_path, "results_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dag_id", "mission_type", "n_tasks",
            "makespan_ms", "total_energy_J",
        ])
        for dag in dags:
            total_energy = sum(t.energy_cpu_J + t.energy_net_J
                               for t in dag.task_chunks.values())
            makespan = dag.get_makespan()
            writer.writerow([
                dag.dag_id,
                dag.mission_type,
                len(dag.task_chunks),
                _ms(makespan) if makespan is not None else "N/A",
                f"{total_energy:.6f}",
            ])

    print(f"[SAVE] risultati salvati in: {output_path}")
    print(f"  - {tasks_path}")
    print(f"  - {summary_path}")

#######################################################################################################
# DEVICES - elenco completo (per il Gantt: mostrare ogni device, anche senza task)
#######################################################################################################

def save_devices_csv(devices: dict, output_path: str):
    """Salva TUTTI i device dello scenario, una riga ciascuno, senza filtri (a differenza di
    `results_tasks.csv`/`offline_windows.csv`, che riportano solo chi ha avuto task/finestre)."""
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, "devices.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["device_id", "device_name", "device_type"])
        for group in devices.values():
            for device in group:
                writer.writerow([device.id, device.raw.name, device.device_type])

    print(f"[SAVE] elenco device salvato in: {path}")


#######################################################################################################
# OFFLINE WINDOWS
#######################################################################################################

def save_offline_windows_csv(devices: dict, output_path: str):
    """Salva le finestre offline per device, in ms (input del Gantt HTML)."""
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, "offline_windows.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["device_id", "device_name", "device_type", "start", "end"])
        for group in devices.values():
            for device in group:
                for start, end in device.offline_windows:
                    writer.writerow([device.id, device.raw.name, device.device_type,
                                     f"{start:.4f}", f"{end:.4f}"])

    print(f"[SAVE] finestre offline salvate in: {path}")
