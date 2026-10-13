"""Missione Classification (CF)."""
from mission.core.mission import MissionDAG
from mission.core.task import TaskChunkFactory, get_task_count


class ClassificationDAG(MissionDAG):

    def __init__(self):
        super().__init__(dag_id="", mission_type="classification")

    def build(self, dag_id, image_size_px=256) -> "ClassificationDAG":
        self.dag_id = dag_id

        self.add_task_chunk(TaskChunkFactory.create("CF-PRE", image_size_px, predecessors=[]))

        cal_ids = []
        for i in range(get_task_count("CF-CAL", image_size_px)):
            chunk = TaskChunkFactory.create("CF-CAL", image_size_px, predecessors=["CF-PRE"], index=i)
            self.add_task_chunk(chunk)
            cal_ids.append(chunk.node_id)

        self.add_task_chunk(TaskChunkFactory.create("CF-SRT", image_size_px, predecessors=cal_ids))
        self.add_task_chunk(TaskChunkFactory.create("CF-RES", image_size_px, predecessors=["CF-SRT"]))
        return self
