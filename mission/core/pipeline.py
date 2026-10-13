"""Task layer, livello alto: `PipelineMission`."""
import importlib
from collections import deque

from mission.core.mission import MissionDAG
from utility.file_utility import PIPELINES_PATH, load_json


#######################################################################################################
# PIPELINE MISSION - fusione dei DAG di missione
#######################################################################################################

class PipelineMission(MissionDAG):
    """Fonde una lista ordinata di `MissionDAG` in un singolo DAG flat."""

    def __init__(self):
        super().__init__(dag_id="", mission_type="full_pipeline")
        self.mission_task_ids = {}            # mission_type -> [node_id, ...]

    def build(self, dag_id, sub_dags: list, bridge_edges: dict = None) -> "PipelineMission":
        self.dag_id = dag_id
        bridge_edges = bridge_edges or {}

        for sub_dag in sub_dags:
            ids = []
            for task in sub_dag.task_chunks.values():
                # arco bridge cross-missione: operation -> node_id predecessore
                bridge = bridge_edges.get(task.operation)
                if bridge is not None and bridge not in task.predecessors:
                    task.predecessors.append(bridge)
                self.add_task_chunk(task)
                ids.append(task.node_id)
            self.mission_task_ids[sub_dag.mission_type] = ids

        self._validate_acyclic()
        return self

    def _validate_acyclic(self) -> None:
        """Kahn: se restano nodi con in-degree>0 c'è un ciclo (bridge sbagliato)."""
        in_degree  = {nid: 0  for nid in self.task_chunks}
        successors = {nid: [] for nid in self.task_chunks}
        for task in self.task_chunks.values():
            for pred_id in task.predecessors:
                if pred_id in self.task_chunks:
                    successors[pred_id].append(task.node_id)
                    in_degree[task.node_id] += 1

        ready, processed = deque(n for n, d in in_degree.items() if d == 0), 0
        while ready:
            nid = ready.popleft(); processed += 1
            for succ in successors[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)

        if processed < len(self.task_chunks):
            cycle = [n for n, d in in_degree.items() if d > 0]
            raise ValueError(f"Ciclo nel DAG della pipeline '{self.dag_id}'. Nodi: {cycle}.")

    def tasks(self) -> list:
        return list(self.task_chunks.values())

    def __repr__(self):
        return f"PipelineMission(id={self.dag_id}, n_mission={len(self.mission_task_ids)}, n_task={len(self.task_chunks)})"


#######################################################################################################
# UTILS / FACTORY
#######################################################################################################

class PipelineUtils:
    """Lettura delle configurazioni pipeline da config/pipelines/."""

    @staticmethod
    def load(ids: list) -> dict:
        return {pid: load_json(f"{pid}.json5", base_path=PIPELINES_PATH) for pid in ids}


class PipelineFactory:
    """Costruisce una `PipelineMission` da una configurazione."""

    @staticmethod
    def build(pipeline_id: str, pipeline_config: dict) -> "PipelineMission":
        image_size_px = pipeline_config["image_size_px"]

        sub_dags = []
        for mission_config in pipeline_config["missions"]:
            module    = importlib.import_module(mission_config["module"])   # es. "mission.imagery_slicing"
            dag_class = getattr(module, mission_config["class"])
            dag_id    = f"{pipeline_id}_{mission_config['suffix']}"
            sub_dags.append(dag_class().build(dag_id, image_size_px))

        return PipelineMission().build(
            pipeline_id, sub_dags, pipeline_config.get("bridge_edges", {}),
        )
