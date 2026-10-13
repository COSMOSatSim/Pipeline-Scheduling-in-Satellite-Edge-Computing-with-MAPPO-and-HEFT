"""Esecuzione SimPy di un Device (modello a pipeline)."""
import itertools

import architecture  # noqa: F401 - attiva lo shim per gli import flat

import globals as sim_globals
import simulation as sim

from mission.core.task import DONE, PENDING, RUNNING
from utility.converter_utility import ms_to_s, s_to_ms

CHUNK_TASK_TYPE = "CPU_and_Data_Intensive"   # ramo CPU+NET del simulatore
TASK_ID_BASE    = 100000                     # id oltre quelli dei task del simulatore
BYTES_PER_PIXEL = 3                          # immagine RGB non compressa (approssimazione)

_task_ids = itertools.count(TASK_ID_BASE)


#######################################################################################################
# COSTI DEL CHUNK -> parametri del simulatore
#######################################################################################################

def _chunk_image_MB(task) -> float:
    """Dati del chunk (MB): immagine intera divisa sul numero di tile del task type."""
    from mission.core.task import TaskChunkUtils
    tiles   = TaskChunkUtils.count(task.operation, task.image_size_px)
    full_MB = (task.image_size_px ** 2) * BYTES_PER_PIXEL / (1024 ** 2)
    return full_MB / max(tiles, 1)


def _chunk_deadline(d_cpu_s: float, image_MB: float) -> float:
    """Deadline del chunk col criterio del simulatore: (1+δ)(d_cpu + d_net)."""
    try:
        delta = float(sim_globals.config.get("deadline", 0.2))
        if delta >= 1.0:
            delta = delta / 100.0
    except Exception:
        delta = 0.2
    delta = min(max(delta, 0.0), 1.0)
    bw_Bps, data_bytes = sim.network_metrics(image_MB)
    d_net = data_bytes / bw_Bps if bw_Bps > 0 else 0.0
    return (1.0 + delta) * (d_cpu_s + d_net)


#######################################################################################################
# DEVICE RUNTIME
#######################################################################################################

class DeviceRuntime:
    """Avvolge un Device e ne esegue i task come processo SimPy."""

    def __init__(self, device, registry: dict, monitor=None):
        self.device          = device
        self._registry       = registry   # device -> DeviceRuntime (per il notify cross-device)
        self._monitor        = monitor
        self._work_event     = None
        self._pending_notify = False      # notify arrivata mentre il device era occupato

    #-------------------------------------
    # PROCESSO SIMPY
    #-------------------------------------
    def run(self, env):
        """Esegue i task in ordine FIFO; se il prossimo non è ready, aspetta."""
        device = self.device
        while any(t.status == PENDING for t in device.assigned_task_queue):
            next_task = next(
                (t for t in device.assigned_task_queue if t.status == PENDING),
                None,
            )
            if next_task is None:
                break
            if (next_task.dag.is_ready()
                    and next_task.is_ready(next_task.dag)
                    and self._cross_device_inputs_ready(next_task)):
                yield env.process(self.execute_task(next_task, env))
            else:
                yield self.wait_for_work(env)

    def _cross_device_inputs_ready(self, task):
        """True se tutti i predecessori cross-device con d_tran > 0 hanno trasmesso."""
        for pred_id in task.predecessors:
            pred = task.dag.task_chunks[pred_id]
            if pred.assigned_device is None or pred.assigned_device is self.device:
                continue
            d_tran = pred.assigned_device.get_transmission_time(
                pred.operation, task.image_size_px, self.device
            )
            if d_tran > 0 and pred_id not in task.input_data:
                return False
        return True

    #-------------------------------------
    # EVENTI SIMPY
    #-------------------------------------
    def wait_for_work(self, env):
        """Si addormenta finché notify_work() non lo sveglia."""
        if self._pending_notify:
            self._pending_notify = False
            return env.timeout(0)
        if self._work_event is None or self._work_event.triggered:
            self._work_event = env.event()
        return self._work_event

    def notify_work(self):
        """Sveglia il device: un dato è arrivato e potrebbe esserci lavoro da fare."""
        if self._work_event is not None and not self._work_event.triggered:
            self._work_event.succeed()
        else:
            self._pending_notify = True   # il device non era in ascolto: non perdere la sveglia

    #-------------------------------------
    # ESECUZIONE TASK
    #-------------------------------------
    def execute_task(self, task, env):
        """Esegue un TaskChunk via il simulatore, poi lancia le trasmissioni ai successori."""
        device = self.device
        dag = task.dag
        if dag.start_time is None:
            dag.start_time = env.now
            self._notify("on_mission_start", dag, env)

        task.status     = RUNNING
        task.start_time = env.now
        dag.update_status()
        self._notify("on_task_start", task, env)

        # Esecuzione nel simulatore: code CPU/NET, energia (C_sen), deadline
        image_MB      = _chunk_image_MB(task)
        d_cpu_s       = ms_to_s(task.d_cpu)
        energy_before = device.energy
        yield from sim.TaskAssignment(env, device.raw, next(_task_ids), image_MB,
                                      env.now, 0, 0.0, CHUNK_TASK_TYPE,
                                      d_cpu_s, _chunk_deadline(d_cpu_s, image_MB))
        task.energy_cpu_J = max(energy_before - device.energy, 0.0)

        task.status   = DONE
        task.end_time = env.now
        dag.update_status()
        self._notify("on_task_end", task, env)

        # Trasmissioni verso successori cross-device (fire-and-forget)
        for succ in self._get_cross_device_successors(task):
            d_tran = device.get_transmission_time(
                task.operation, succ.image_size_px, succ.assigned_device
            )
            if d_tran > 0:
                env.process(self.transmit_to(succ, task.node_id, d_tran, env))

        # Ultimo task del DAG?
        if dag.is_complete() and dag.end_time is None:
            dag.end_time = max(t.end_time for t in dag.task_chunks.values())
            dag.update_status()
            self._notify("on_mission_end", dag, env)

    #-------------------------------------
    # TRASMISSIONE
    #-------------------------------------
    def transmit_to(self, target_task, source_label, d_tran_ms, env):
        """Trasmette verso il device del target_task, notifica il monitor, aggiorna input_data."""
        device        = self.device
        target_device = target_task.assigned_device
        d_tran_s      = ms_to_s(d_tran_ms)
        # il modello (finestre offline) lavora in ms: conversione ai bordi
        start_time_s  = ms_to_s(device.get_available_transmission_start(
            s_to_ms(env.now), d_tran_ms, target_device))

        if start_time_s > env.now:
            yield env.timeout(start_time_s - env.now)

        self._notify("on_tran_start", target_task.dag, device, source_label, target_task, d_tran_ms, env)

        yield env.timeout(d_tran_s)

        self._notify("on_tran_end", target_task.dag, device, source_label, target_task, env)

        # Energia di trasmissione: modello del simulatore (Ptrasm × tempo), a carico della sorgente
        energy_net = sim_globals.config.get("Ptrasm", 1.0) * d_tran_s
        target_task.energy_net_J += energy_net      # quanto e' costato ricevere
        device.raw.energy        -= energy_net
        # La stessa spesa registrata dove viene sostenuta: sul task sorgente, e quindi sul device
        # che ha spedito.
        source_task = target_task.dag.task_chunks.get(source_label)
        if source_task is not None:
            source_task.energy_tx_J += energy_net

        target_task.input_data.add(source_label)
        self._registry[target_device].notify_work()

    def _get_cross_device_successors(self, task):
        """Ritorna i TaskChunk successori del task assegnati a device diversi da questo."""
        device = self.device
        successors = []
        for candidate in task.dag.task_chunks.values():
            if task.node_id in candidate.predecessors:
                if candidate.assigned_device is not None and candidate.assigned_device is not device:
                    successors.append(candidate)
        return successors

    #-------------------------------------
    # MONITOR UPDATE
    #-------------------------------------

    def _notify(self, hook, *args):
        """Invoca un hook del monitor se presente."""
        if self._monitor:
            getattr(self._monitor, hook)(*args)
