"""Funzioni di utilita' condivise dagli scheduler."""
import bisect
import weakref
from collections import deque

# Ordine topologico memoizzato sul DAG: il DAG (task + predecessori) è fisso per tutta la durata di
# un episodio, quindi il topo-sort è invariante tra gli step.
_TOPO_ORDER_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def collect_tasks_by_pipeline(devices: dict, sort_tasks: bool = True) -> dict:
    """Raccoglie tutti i task assegnati ai device e li raggruppa per pipeline_id."""
    # 1) Raccoglie i task in liste raggruppate per pipeline
    tasks_by_pipeline: dict = {}
    for group in devices.values():
        for device in group:
            for task in device.assigned_task_queue:
                pipeline_id = task.dag.dag_id
                tasks_by_pipeline.setdefault(pipeline_id, []).append(task)

    # 2) Sort topologico (opzionale) - ordine memoizzato sul DAG, vedi topological_order
    if sort_tasks:
        for pipeline_id, tasks in tasks_by_pipeline.items():
            order   = topological_order(tasks[0].dag, tasks)
            by_node = {t.node_id: t for t in tasks}
            tasks_by_pipeline[pipeline_id] = (
                [by_node[nid] for nid in order] if len(by_node) == len(order)
                else topological_sort_tasks(tasks)   # fallback se la struttura cambia
            )

    # 3) Trasforma ogni task in dizionario { task, device_id, predecessors, start_time, end_time }
    by_pipeline: dict = {}
    for pipeline_id, tasks in tasks_by_pipeline.items():
        by_pipeline[pipeline_id] = {
            task.node_id: {
                "task":         task,
                "device_id":    task.assigned_device.device_id,
                "predecessors": list(task.predecessors) if task.predecessors else None,
                "start_time":   None,
                "end_time":     None,
            }
            for task in tasks
        }
    return by_pipeline

#######################################################################################################
# QUERY HELPERS - predicati su un TaskChunk usando la struttura by_pipeline
#######################################################################################################

def _entry_of(by_pipeline: dict, task) -> dict:
    """Estrae l'entry del task da by_pipeline usando task.dag.dag_id e task.node_id."""
    return by_pipeline[task.dag.dag_id][task.node_id]

def set_start_time(by_pipeline: dict, task, start_time: float) -> None:
    """Imposta lo start_time di task nel dizionario by_pipeline."""
    _entry_of(by_pipeline, task)["start_time"] = start_time

def get_start_time(by_pipeline: dict, task):
    """Ritorna lo start_time di task (None se non ancora calcolato)."""
    return _entry_of(by_pipeline, task)["start_time"]

def set_end_time(by_pipeline: dict, task, end_time: float) -> None:
    """Imposta l'end_time di task nel dizionario by_pipeline."""
    _entry_of(by_pipeline, task)["end_time"] = end_time

def get_end_time(by_pipeline: dict, task):
    """Ritorna l'end_time di task (None se non ancora calcolato)."""
    return _entry_of(by_pipeline, task)["end_time"]

def get_transmission_cost(parent_task, child_task) -> float:
    """Tempo di trasmissione (ms) dal device del padre al device del figlio."""
    return parent_task.assigned_device.get_transmission_time(
        parent_task.operation, child_task.image_size_px, child_task.assigned_device,
    )

def has_parent(by_pipeline: dict, task) -> bool:
    """True se il task ha almeno un predecessore (non e' un task entry)."""
    return _entry_of(by_pipeline, task)["predecessors"] is not None

def has_end_time(by_pipeline: dict, task) -> bool:
    """True se l'end_time del task e' gia' stato calcolato."""
    return _entry_of(by_pipeline, task)["end_time"] is not None

def all_parents_have_end_time(by_pipeline: dict, task) -> bool:
    """True se tutti i predecessori del task hanno gia' end_time != None."""
    entry = _entry_of(by_pipeline, task)
    if entry["predecessors"] is None:
        return True
    pipeline_entries = by_pipeline[task.dag.dag_id]
    return all(
        pipeline_entries[pred_id]["end_time"] is not None
        for pred_id in entry["predecessors"]
    )

def get_pending_parents(by_pipeline: dict, task) -> set:
    """Ritorna l'insieme dei TaskChunk predecessori che NON hanno ancora end_time."""
    entry = _entry_of(by_pipeline, task)
    if entry["predecessors"] is None:
        return set()
    pipeline_entries = by_pipeline[task.dag.dag_id]
    return {
        pipeline_entries[pred_id]["task"]
        for pred_id in entry["predecessors"]
        if pipeline_entries[pred_id]["end_time"] is None
    }

def get_max_end_parent(by_pipeline: dict, task):
    """Ritorna il padre di task che massimizza (end_time + tempo di trasmissione)."""
    entry = _entry_of(by_pipeline, task)
    if entry["predecessors"] is None:
        return None

    pipeline_entries = by_pipeline[task.dag.dag_id]

    candidates = []
    for pred_id in entry["predecessors"]:
        pred_entry = pipeline_entries[pred_id]
        if pred_entry["end_time"] is None:
            raise ValueError(
                f"Il padre {pred_id} del task {task.node_id} non ha end_time."
            )
        pred_task = pred_entry["task"]
        trans     = pred_task.assigned_device.get_transmission_time(
            pred_task.operation, task.image_size_px, task.assigned_device,
        )
        candidates.append((pred_entry, pred_entry["end_time"] + trans))

    # max su (end_time + trans): il vero bottleneck di DataReady
    chosen, _ = max(candidates, key=lambda c: c[1])
    return chosen["task"]

#######################################################################################################
# OUTPUT - scrittura dei tempi schedulati sui task
#######################################################################################################

def persist_scheduled_times(devices: dict, by_pipeline: dict) -> None:
    """Scrive su ogni task i tempi calcolati in by_pipeline (output EFT)."""
    for group in devices.values():
        for d in group:
            for task in d.assigned_task_queue:
                task.scheduled_execution_start = get_start_time(by_pipeline, task)
                task.scheduled_execution_end   = get_end_time(by_pipeline, task)

#######################################################################################################
# HEFT - COSTI MEDI E INSERTION POLICY (funzioni pure, Topcuoglu 2002)
#######################################################################################################

def mean_exec_cost(task, all_devices: list) -> float:
    """
    Costo medio di computazione del task n_i sui processor (eq. 1):
        w_i_bar = (1/q) * sum_j w_{i,j}
    """
    costs = [d.get_d_cpu(task.operation, task.image_size_px) for d in all_devices]
    return sum(costs) / len(costs) if costs else 0.0

def mean_comm_cost(task_i, task_j, all_devices: list) -> float:
    """Costo medio di comunicazione sull'edge (i, j) (eq. 3): c_ij_bar = media su tutte le coppie
    (src_dev, dst_dev)."""
    total = 0.0
    count = 0
    for src in all_devices:
        for dst in all_devices:
            total += src.get_transmission_time(
                task_i.operation, task_j.image_size_px, dst,
            )
            count += 1
    return total / count if count else 0.0

#######################################################################################################
# CACHE DEI COSTI MEDI HEFT
#######################################################################################################
# w_i_bar e c_ij_bar (rank_u) dipendono solo da DAG + insieme device: sono INVARIANTI
# per (tipi device, operazione, image_size). Si calcolano una volta e si riusano, invece
# di rifarli ad ogni step (mean_comm_cost è O(n_device²)).
_MEAN_EXEC_CACHE: dict = {}
_MEAN_COMM_CACHE: dict = {}


def device_signature(all_devices: list) -> tuple:
    """Segnatura dell'insieme device che determina i costi medi: i tipi (SAT/SAS/GND)."""
    return tuple(sorted(d.device_type for d in all_devices))


def mean_exec_cost_cached(sig: tuple, task, all_devices: list) -> float:
    """mean_exec_cost memoizzato su (sig, operation, image_size)."""
    key = (sig, task.operation, task.image_size_px)
    if key not in _MEAN_EXEC_CACHE:
        _MEAN_EXEC_CACHE[key] = mean_exec_cost(task, all_devices)
    return _MEAN_EXEC_CACHE[key]


def mean_comm_cost_cached(sig: tuple, task_i, task_j, all_devices: list) -> float:
    """mean_comm_cost memoizzato su (sig, operation_i, image_size_j)."""
    key = (sig, task_i.operation, task_j.image_size_px)
    if key not in _MEAN_COMM_CACHE:
        _MEAN_COMM_CACHE[key] = mean_comm_cost(task_i, task_j, all_devices)
    return _MEAN_COMM_CACHE[key]

def find_earliest_slot(intervals: list, ready_time: float, exec_time: float) -> float:
    """Primo idle time slot sul processor per un task con exec_time, partendo da ready_time."""
    n = len(intervals)
    if n == 0:
        return ready_time

    # Slot prima del primo task già schedulato
    if ready_time + exec_time <= intervals[0][0]:
        return ready_time

    # I gap che chiudono entro ready_time non possono ospitare il task (slot_start sarebbe
    # ready_time ma il gap finisce prima): si saltano con una ricerca binaria del primo intervallo
    # che inizia dopo ready_time.
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) >> 1
        if intervals[mid][0] > ready_time:
            hi = mid
        else:
            lo = mid + 1
    i = lo - 1 if lo > 0 else 0

    # Slot tra task consecutivi già schedulati (max inlinato: hot path)
    while i < n - 1:
        prev_end   = intervals[i][1]
        slot_start = prev_end if prev_end > ready_time else ready_time
        if slot_start + exec_time <= intervals[i + 1][0]:
            return slot_start
        i += 1

    # Accodamento dopo l'ultimo task
    last_end = intervals[-1][1]
    return last_end if last_end > ready_time else ready_time

#######################################################################################################
# TRASMISSIONE - finestra (tx_start, tx_end)
#######################################################################################################

def transmission_window(task, src_device, dst_device, pred_end: float,
                        duration_cache=None) -> tuple[float, float]:
    """Vincoli (8-11): calcola (tx_start, tx_end) per trasmettere l'output del task da src_device a
    dst_device, considerando d_tran e finestre offline."""
    if src_device == dst_device:
        return pred_end, pred_end
    if duration_cache is None:
        d_tran = src_device.get_transmission_time(task.operation, task.image_size_px, dst_device)
    else:
        key = (src_device, dst_device, task.operation, task.image_size_px)
        d_tran = duration_cache.get(key)
        if d_tran is None:
            d_tran = src_device.get_transmission_time(task.operation, task.image_size_px, dst_device)
            duration_cache[key] = d_tran
    tx_start = src_device.get_available_transmission_start(pred_end, d_tran, dst_device)
    return tx_start, tx_start + d_tran


def insert_busy_interval(intervals: list, start: float, end: float) -> None:
    """Inserisce un intervallo occupato fondendo solo quelli esattamente adiacenti."""
    index = bisect.bisect_left(intervals, (start, end))
    stop = index
    if index and intervals[index - 1][1] == start:
        index -= 1
        start = intervals[index][0]
    if stop < len(intervals) and intervals[stop][0] == end:
        end = intervals[stop][1]
        stop += 1
    intervals[index:stop] = [(start, end)]

#######################################################################################################
# TOPOLOGICAL SORT
#######################################################################################################

def topological_order(dag, tasks: list) -> list:
    """Ordine topologico dei node_id del DAG, memoizzato sul DAG."""
    order = _TOPO_ORDER_CACHE.get(dag)
    if order is None:
        order = [t.node_id for t in topological_sort_tasks(tasks)]
        _TOPO_ORDER_CACHE[dag] = order
    return order

def topological_sort_tasks(tasks: list) -> list:
    """Riordina una lista di task in ordine topologico secondo i predecessori del DAG."""
    if not tasks:
        return []

    id_to_task = {id(t): t for t in tasks}
    valid      = set(id_to_task.keys())
    in_degree  = {id(t): 0  for t in tasks}
    successors = {id(t): [] for t in tasks}

    for task in tasks:
        for pred_node_id in task.predecessors:
            pred = task.dag.task_chunks.get(pred_node_id)
            if pred is None or id(pred) not in valid:
                continue
            successors[id(pred)].append(id(task))
            in_degree[id(task)] += 1

    ready   = deque(tid for tid, deg in in_degree.items() if deg == 0)
    ordered = []
    while ready:
        tid = ready.popleft()
        ordered.append(id_to_task[tid])
        for succ_id in successors[tid]:
            in_degree[succ_id] -= 1
            if in_degree[succ_id] == 0:
                ready.append(succ_id)

    if len(ordered) != len(tasks):
        raise ValueError("Ciclo rilevato nei predecessori del DAG.")
    return ordered





