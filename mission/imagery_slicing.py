"""Missione Imagery Slicing (IS)."""
from mission.core.mission import MissionDAG
from mission.core.task import TaskChunkFactory


class ImagerySlicingDAG(MissionDAG):

    def __init__(self):
        super().__init__(dag_id="", mission_type="imagery_slicing")

    def build(self, dag_id, image_size_px=256) -> "ImagerySlicingDAG":
        self.dag_id = dag_id
        self.add_task_chunk(TaskChunkFactory.create("IS-LAD", image_size_px))
        self.add_task_chunk(TaskChunkFactory.create("IS-SEG", image_size_px, predecessors=["IS-LAD"]))
        return self
