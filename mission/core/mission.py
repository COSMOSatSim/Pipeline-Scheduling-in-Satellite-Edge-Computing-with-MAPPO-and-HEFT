"""Task layer, livello intermedio: `MissionDAG`."""
from abc import ABC, abstractmethod
from mission.core.task import TaskChunk, PENDING, DONE, FAILED, RUNNING


class MissionDAG(ABC):
    """Classe base astratta per una missione (DAG di `TaskChunk`)."""

    def __init__(self, dag_id, mission_type):
        self.dag_id              = dag_id
        self.mission_type        = mission_type
        self.predecessor_mission = None
        self.task_chunks         = {}        # node_id -> TaskChunk
        self.status              = PENDING
        self.start_time          = None
        self.end_time            = None
        self.source_device       = None

    @abstractmethod
    def build(self, dag_id, image_size_px) -> "MissionDAG":
        """Costruisce il DAG della missione (ogni missione concreta lo implementa)."""
        ...

    #-------------------------------------
    # NODES
    #-------------------------------------
    def add_task_chunk(self, node: TaskChunk):
        """Aggiunge un `TaskChunk` al DAG (e imposta il riferimento al DAG nel nodo)."""
        node.dag = self
        self.task_chunks[node.node_id] = node

    #-------------------------------------
    # STATUS
    #-------------------------------------
    def update_status(self):
        if self.has_failed():
            self.status = FAILED
        elif self.is_complete():
            self.status = DONE
        elif any(n.status == RUNNING for n in self.task_chunks.values()):
            self.status = RUNNING
        else:
            self.status = PENDING

    def is_ready(self) -> bool:
        """True se la missione predecessore è completata."""
        return self.predecessor_mission is None or self.predecessor_mission.is_complete()

    def is_complete(self) -> bool:
        return all(n.status == DONE for n in self.task_chunks.values())

    def has_failed(self) -> bool:
        return any(n.status == FAILED for n in self.task_chunks.values())

    def get_makespan(self):
        """Durata della missione (s), None se non completata."""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def reset(self):
        self.status = PENDING
        self.start_time = self.end_time = None
        for chunk in self.task_chunks.values():
            chunk.status = PENDING
            chunk.task   = None          # scollega la loro Task (ricreata al dispatch)
            chunk.assigned_device = None
            chunk.d_cpu           = None
            chunk.start_time      = None
            chunk.end_time        = None
            chunk.energy_cpu_J    = 0.0
            chunk.energy_net_J    = 0.0
            chunk.energy_tx_J     = 0.0
            chunk.input_data.clear()

    def __repr__(self):
        return f"MissionDAG(id={self.dag_id}, mission={self.mission_type}, nodes={len(self.task_chunks)})"
