"""Missione Cloud Detection (CD)."""
from mission.core.mission import MissionDAG
from mission.core.task import TaskChunkFactory, get_task_count


class CloudDetectionDAG(MissionDAG):

    def __init__(self):
        super().__init__(dag_id="", mission_type="cloud_detection")

    def build(self, dag_id, image_size_px=256) -> "CloudDetectionDAG":
        self.dag_id = dag_id

        fmk_ids = []
        for i in range(get_task_count("CD-FMK", image_size_px)):
            chunk = TaskChunkFactory.create("CD-FMK", image_size_px, predecessors=[], index=i)
            self.add_task_chunk(chunk)
            fmk_ids.append(chunk.node_id)

        self.add_task_chunk(TaskChunkFactory.create("CD-GEN", image_size_px, predecessors=fmk_ids))
        return self
