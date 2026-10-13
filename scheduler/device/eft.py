"""HEFT insertion-based con vista GLOBALE su tutti i DAG (Topcuoglu et al., 2002)."""

#######################################################################################################
# IMPORT
#######################################################################################################
import bisect
from scheduler.core.scheduler import DeviceScheduler
from scheduler.device.utility.eft_diagnostic import EFTDiagnostic
from scheduler.device.utility.eft_utility import (
    collect_tasks_by_pipeline,
    get_start_time,
    get_end_time,
    persist_scheduled_times,
    transmission_window,
    device_signature,
    mean_exec_cost_cached,
    mean_comm_cost_cached,
    find_earliest_slot,
    insert_busy_interval,
)


#######################################################################################################
# EFT DEVICE SCHEDULER
#######################################################################################################

class EFTScheduler(DeviceScheduler):
    """Sequenzia le code assegnate dei device con EFT insertion-based (HEFT, Topcuoglu 2002)."""

    def __init__(self, diagnostic: bool = False):
        self._diagnostic = diagnostic   # confronto finish_time PRE/POST sequencing

    #######################################################################################################
    # EFT INSERTION-BASED (paper HEFT, Section 4.2)
    #######################################################################################################

    def _compute_upward_ranks(self, by_pipeline: dict, devices: dict) -> dict:
        """
        Upward rank di ogni task (eq. 8-9 del paper) con costi medi.

        rank_u(n_exit) = w_exit_bar
        rank_u(n_i)    = w_i_bar + max_{n_j in succ(n_i)} ( c_ij_bar + rank_u(n_j) )

        Returns:
            dict { (dag_id, node_id) : rank_u }
        """
        all_devices = [d for group in devices.values() for d in group]
        sig = device_signature(all_devices)   # calcolata una volta: chiave della cache costi medi
        ranks: dict = {}

        for dag_id, entries in by_pipeline.items():
            # Mappa successori (entries e' in ordine topologico)
            successors = {nid: [] for nid in entries}
            for nid, entry in entries.items():
                if entry["predecessors"]:
                    for pred_id in entry["predecessors"]:
                        if pred_id in successors:
                            successors[pred_id].append(nid)

            # Risale dall'exit (ordine topologico inverso)
            for nid in reversed(list(entries.keys())):
                task  = entries[nid]["task"]
                w_avg = mean_exec_cost_cached(sig, task, all_devices)
                if not successors[nid]:                    # exit task
                    ranks[(dag_id, nid)] = w_avg
                else:
                    max_term = 0.0
                    for succ_id in successors[nid]:
                        succ_task = entries[succ_id]["task"]
                        c_avg     = mean_comm_cost_cached(sig, task, succ_task, all_devices)
                        max_term  = max(max_term, c_avg + ranks[(dag_id, succ_id)])
                    ranks[(dag_id, nid)] = w_avg + max_term

        return ranks

    def _eft_insertion_based(self, devices: dict) -> dict:
        """HEFT insertion-based con vista GLOBALE su tutti i DAG. 1."""
        by_pipeline = collect_tasks_by_pipeline(devices, sort_tasks=True)

        # 1-2. Upward rank + ordine globale per rank decrescente
        ranks = self._compute_upward_ranks(by_pipeline, devices)
        global_order = sorted(
            ((dag_id, nid, entry)
             for dag_id, entries in by_pipeline.items()
             for nid, entry in entries.items()),
            key=lambda x: -ranks[(x[0], x[1])],
        )

        # 3. Insertion-based EFT in ordine globale di rank
        device_slots: dict = {}   # device_id -> [(start, end), ...] ordinata per start
        # Solo le durate sono riutilizzabili: le finestre si ricalcolano sempre usando il nuovo
        # end_time del predecessore e il satellite effettivo.
        duration_cache = {}
        compact_slots = all(entry["task"].d_cpu > 0 for _, _, entry in global_order)

        for dag_id, _nid, entry in global_order:
            task   = entry["task"]
            device = task.assigned_device
            dev_id = device.device_id

            device_slots.setdefault(dev_id, [])

            # ready_time = 0 per gli entry (eq. 4); altrimenti max(t_pred^tx_end)
            if entry["predecessors"] is None:
                ready_time = 0.0
            else:
                ready_time = 0.0
                pipeline_local = by_pipeline[dag_id]
                for pred_id in entry["predecessors"]:
                    pred_entry = pipeline_local[pred_id]
                    pred_task  = pred_entry["task"]
                    # rank decrescente garantisce che i padri siano gia' schedulati.
                    # transmission_window incapsula i vincoli (8), (9), (10), (11).
                    tx_start, tx_end = transmission_window(
                        pred_task, pred_task.assigned_device, task.assigned_device,
                        pred_entry["end_time"],
                        duration_cache=duration_cache,
                    )
                    # Persisti sui campi del pred (mapping 1:1 col paper):
                    pred_task.scheduled_transmission_start = tx_start
                    pred_task.scheduled_transmission_end   = tx_end
                    ready_time = max(ready_time, tx_end)   # ready_succ = tx_end_pred

            # d_cpu è già popolato da device.assign_task -> usalo direttamente.
            start_time = find_earliest_slot(device_slots[dev_id], ready_time, task.d_cpu)
            end_time   = start_time + task.d_cpu

            entry["start_time"] = start_time
            entry["end_time"]   = end_time

            if compact_slots:
                insert_busy_interval(device_slots[dev_id], start_time, end_time)
            else:
                bisect.insort(device_slots[dev_id], (start_time, end_time))

        return by_pipeline

    #######################################################################################################
    # EFT ordina i task ( start-time, end-time, node_id, dag_id )
    #######################################################################################################

    def sort_tasks(self, devices: dict, by_pipeline: dict):
        for group in devices.values():
            for d in group:
                if len(d.assigned_task_queue) > 1:
                    d.assigned_task_queue.sort(
                        key=lambda task: (
                            get_start_time(by_pipeline, task),
                            get_end_time(by_pipeline, task),
                            getattr(task, "node_id", ""),
                            task.dag.dag_id if task.dag is not None else "",
                        )
                    )

    #######################################################################################################
    # MAIN METHOD
    #######################################################################################################

    def apply(self, devices: dict) -> None:
        """Applica EFT insertion-based a tutte le code assegnate dei device."""
        # --- Diagnostica opzionale: snapshot PRE ---
        diagnostic = EFTDiagnostic() if self._diagnostic else None
        if diagnostic is not None:
            diagnostic.snapshot_pre(devices)

        # --- EFT insertion-based + sort delle code ---
        by_pipeline = self._eft_insertion_based(devices)
        self.sort_tasks(devices, by_pipeline)

        # --- Diagnostica opzionale: report PRE vs POST ---
        if diagnostic is not None:
            diagnostic.report_post(devices, by_pipeline)

        # --- Output: tempi schedulati sui task ---
        persist_scheduled_times(devices, by_pipeline)
