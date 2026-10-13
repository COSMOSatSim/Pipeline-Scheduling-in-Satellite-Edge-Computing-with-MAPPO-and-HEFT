"""Calcolo di CE (vincolo 17 del paper SECORS)."""

#######################################################################################################
# IMPORT
#######################################################################################################
from devices.core.device import DEVICE_SAT, DEVICE_SAS

#######################################################################################################
# CE - ENERGIA TOTALE
#######################################################################################################

def compute_CE(devices: dict) -> float:
    """Energia totale CE (Wms) dello scheduling - vincolo 17."""
    return _execution_energy(devices) + _transmission_energy(devices)

#######################################################################################################
# COMPONENTI
#######################################################################################################

def _execution_energy(devices: dict) -> float:
    """Energia di esecuzione: solo SAT e SAS contribuiscono (paper eq. 17, k ∈ S)."""
    total = 0.0
    for group in devices.values():
        for device in group:
            if device.device_type not in (DEVICE_SAT, DEVICE_SAS):
                continue
            for task in device.assigned_task_queue:
                total += task.d_cpu * device.power_exec_W
    return total


def _transmission_energy(devices: dict) -> float:
    """Energia di trasmissione: somma su tutti gli archi cross-device del DAG."""
    total = 0.0
    # Cache locale alla valutazione: non conserva valori tra scenari o modifiche alle tabelle di
    # costo.
    durations = {}
    for group in devices.values():
        for device in group:
            for task in device.assigned_task_queue:
                for pred_id in task.predecessors:
                    pred = task.dag.task_chunks.get(pred_id)
                    if pred is None or pred.assigned_device is None:
                        continue
                    if pred.assigned_device == task.assigned_device:
                        continue
                    key = (pred.assigned_device, task.assigned_device,
                           pred.operation, task.image_size_px)
                    d_tran = durations.get(key)
                    if d_tran is None:
                        d_tran = pred.assigned_device.get_transmission_time(
                            pred.operation, task.image_size_px, task.assigned_device,
                        )
                        durations[key] = d_tran
                    total += d_tran * pred.assigned_device.power_tran_W
    return total
