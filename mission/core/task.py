"""Task layer (livello base): `TaskChunk` = nodo di un DAG."""
import os

from utility.converter_utility import ms_to_s
from utility.file_utility import MISSION_PATH, load_json

#######################################################################################################
# STATUS
#######################################################################################################
PENDING = "pending"
RUNNING = "running"
DONE    = "done"
FAILED  = "failed"

#######################################################################################################
# TASK CHUNK - nodo del DAG
#######################################################################################################

class TaskChunk:
    """Nodo di un DAG di missione."""

    def __init__(self, node_id, operation, image_size_px=None,
                 predecessors=None, is_assignable: bool = True):
        self.node_id       = node_id
        self.operation     = operation
        self.image_size_px = image_size_px
        self.predecessors  = predecessors or []
        self.is_assignable = is_assignable
        self.dag           = None
        self.status        = PENDING
        self.task          = None
        # Scheduling / esecuzione (riempiti da assign_task e dal runtime)
        self.assigned_device = None
        self.d_cpu           = None       # ms, dal device assegnato
        self.start_time      = None       # s (clock della simulazione)
        self.end_time        = None
        self.energy_cpu_J    = 0.0
        self.energy_net_J    = 0.0   # ricevere: costo delle trasmissioni in ingresso
        self.energy_tx_J     = 0.0   # spedire: costo sostenuto dal device di questo task
        self.input_data      = set()      # output dei predecessori già ricevuti

    def is_ready(self, dag) -> bool:
        """True se tutti i predecessori sono DONE."""
        return all(dag.task_chunks[p].status == DONE for p in self.predecessors)

    def __repr__(self):
        return f"TaskChunk(id={self.node_id}, op={self.operation}, status={self.status})"


#######################################################################################################
# TASK CHUNK UTILS - registry dei parametri di missione (config/mission/*.json5)
#######################################################################################################

class TaskChunkUtils:
    """Carica e fonde i config/mission/*.json5 ed espone `count` / `is_assignable`."""

    _tasks: dict = {}

    @staticmethod
    def _load_mission_files() -> dict:
        merged: dict = {}
        for fname in sorted(os.listdir(MISSION_PATH)):
            if not fname.endswith(".json5"):
                continue
            data = load_json(fname, base_path=MISSION_PATH)
            for _, tasks in data.items():                  # una chiave top-level per file
                for task_name, props in tasks.items():
                    if task_name in merged:
                        raise ValueError(f"Task '{task_name}' definito in più mission file.")
                    merged[task_name] = props
        return merged

    @classmethod
    def count(cls, task_name: str, image_size_px: int, default: int = 1) -> int:
        """Numero di nodi paralleli per task type × scala immagine."""
        if task_name not in cls._tasks:
            return default
        counts = cls._tasks[task_name].get("count", {})
        return counts.get(str(image_size_px), default)

    @classmethod
    def is_assignable(cls, task_name: str) -> bool:
        if task_name not in cls._tasks:
            return True
        return bool(cls._tasks[task_name].get("is_assignable", True))

    @classmethod
    def execution_time(cls, task_name: str, device_type: str, image_size_px: int) -> float:
        """Tempo (ms) di esecuzione del task type sul device_type per la scala data."""
        if task_name not in cls._tasks:
            raise ValueError(f"Task '{task_name}' non trovato nei mission file.")
        exec_times = cls._tasks[task_name].get("execution_time", {})
        if device_type not in exec_times:
            raise ValueError(
                f"Device type '{device_type}' non supportato per '{task_name}'. "
                f"Device validi: {list(exec_times.keys())}"
            )
        key = str(image_size_px)
        if key not in exec_times[device_type]:
            raise ValueError(
                f"Dimensione {image_size_px}px non supportata per '{task_name}' su {device_type}."
            )
        return exec_times[device_type][key]

    @classmethod
    def output_trasmissione_time(cls, task_name: str, link_type: str, image_size_px: int) -> float:
        """Tempo (ms) per trasmettere l'output del task via link_type, dato image_size."""
        if task_name not in cls._tasks:
            return 0.0
        trans = cls._tasks[task_name].get("output_trasmissione_time", {})
        if link_type not in trans:
            return 0.0
        return trans[link_type].get(str(image_size_px), 0.0)

    #-------------------------------------
    # COSTI IN SECONDI (clock della simulazione)
    #-------------------------------------

    @classmethod
    def d_cpu(cls, chunk: "TaskChunk", device) -> float:
        """Tempo di esecuzione (s) di `chunk` su `device` (colonna del suo device_type)."""
        return ms_to_s(cls.execution_time(chunk.operation, device.device_type, chunk.image_size_px))

    @classmethod
    def d_tran(cls, chunk: "TaskChunk", src_device, dst_device) -> float:
        """Tempo (s) per trasmettere l'output di `chunk` da `src_device` a `dst_device`.

        0.0 se sorgente e destinazione coincidono; il link è `satellite_to_ground`
        se uno dei due è GND, altrimenti `inter_satellite`.
        """
        if src_device.id == dst_device.id:
            return 0.0
        from devices.core.device import GND
        link = ("satellite_to_ground"
                if GND in (src_device.device_type, dst_device.device_type)
                else "inter_satellite")
        return ms_to_s(cls.output_trasmissione_time(chunk.operation, link, chunk.image_size_px))


#######################################################################################################
# TASK CHUNK FACTORY
#######################################################################################################

class TaskChunkFactory:
    """Costruisce TaskChunk; legge is_assignable dal registry."""

    @classmethod
    def create(cls, task_name: str, image_size_px: int, predecessors=None, index=None) -> TaskChunk:
        node_id = f"{task_name}-{index}" if index is not None else task_name
        return TaskChunk(
            node_id       = node_id,
            operation     = task_name,
            image_size_px = image_size_px,
            predecessors  = predecessors or [],
            is_assignable = TaskChunkUtils.is_assignable(task_name),
        )

#######################################################################################################
# TASK METHODS SIMPLE SIGNATURE
#######################################################################################################

# Carica i mission file a import time + alias usato dalle missioni.
TaskChunkUtils._tasks = TaskChunkUtils._load_mission_files()
get_task_count = TaskChunkUtils.count
