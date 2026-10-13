"""Missione Geometric Correction (GC)."""
from mission.core.mission import MissionDAG
from mission.core.task import TaskChunkFactory, get_task_count


class GeometricCorrectionDAG(MissionDAG):

    def __init__(self):
        super().__init__(dag_id="", mission_type="geometric_correction")

    def build(self, dag_id, image_size_px=256) -> "GeometricCorrectionDAG":
        self.dag_id = dag_id

        rfm_ids = []
        for i in range(get_task_count("GC-RFM", image_size_px)):
            chunk = TaskChunkFactory.create("GC-RFM", image_size_px, predecessors=[], index=i)
            self.add_task_chunk(chunk)
            rfm_ids.append(chunk.node_id)

        self.add_task_chunk(TaskChunkFactory.create("GC-MEG", image_size_px, predecessors=rfm_ids))
        return self
