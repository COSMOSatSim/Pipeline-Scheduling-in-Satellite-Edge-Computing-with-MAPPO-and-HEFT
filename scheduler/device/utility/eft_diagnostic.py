"""Diagnostica PRE/POST sequencing (HEFT/EFT)."""
from utility.log_colors import COLOR_GREEN, COLOR_RED, COLOR_CYAN, color_for_device, colorize
from scheduler.device.utility.eft_utility import (
    collect_tasks_by_pipeline,
    has_end_time,
    has_parent,
    all_parents_have_end_time,
    get_max_end_parent,
    get_pending_parents,
    get_end_time,
    get_transmission_cost,
    set_start_time,
    set_end_time,
)


class EFTDiagnostic:
    """Misura e stampa la riduzione di finish_time per device dopo il sequencing."""

    def __init__(self):
        self._finish_pre: dict | None = None

    #-------------------------------------
    # SNAPSHOT PRE / REPORT POST
    #-------------------------------------
    def snapshot_pre(self, devices: dict) -> None:
        """finish_time PRIMA del sequencing (ordine originale, modello FIFO)."""
        self._finish_pre = self._compute_finish_times(devices, self._sequential_completion_times(devices))

    def report_post(self, devices: dict, by_pipeline_post: dict) -> None:
        """Tabella PRE/POST."""
        if self._finish_pre is None:
            return
        finish_post = self._compute_finish_times(devices, by_pipeline_post)
        self._print_finish_times_diff(self._finish_pre, finish_post)

    #-------------------------------------
    # MODELLO SEQUENZIALE - baseline PRE-EFT
    #-------------------------------------
    @staticmethod
    def _sequential_completion_times(devices: dict) -> dict:
        """Completion time con modello sequenziale FIFO sull'ordine ORIGINALE della coda:
        start = max(prev_end, data_ready)."""
        by_pipeline = collect_tasks_by_pipeline(devices, True)
        pending = set()
        while True:
            for type_d in devices.values():
                for d in type_d:
                    for i in range(len(d.assigned_task_queue)):
                        task_i = d.assigned_task_queue[i]
                        if has_end_time(by_pipeline, task_i):
                            continue

                        p = None
                        if has_parent(by_pipeline, task_i):
                            if all_parents_have_end_time(by_pipeline, task_i):
                                p = get_max_end_parent(by_pipeline, task_i)
                            else:
                                pending.update(get_pending_parents(by_pipeline, task_i))
                                pending.add(task_i)
                                break

                        prev_end   = get_end_time(by_pipeline, d.assigned_task_queue[i - 1]) if i > 0 else 0.0
                        data_ready = (get_end_time(by_pipeline, p) + get_transmission_cost(p, task_i)
                                      if p is not None else 0.0)

                        exec_time  = d.get_d_cpu(task_i.operation, task_i.image_size_px)
                        start_time = max(prev_end, data_ready)
                        set_start_time(by_pipeline, task_i, start_time)
                        set_end_time(by_pipeline, task_i, start_time + exec_time)
                        pending.discard(task_i)

            if not pending:
                break
        return by_pipeline

    #-------------------------------------
    # FINISH TIME - calcolo e stampa
    #-------------------------------------
    @staticmethod
    def _compute_finish_times(devices: dict, by_pipeline: dict) -> dict:
        """Per ogni device con task assegnati: end_time dell'ultimo task in coda."""
        return {
            d.device_id: get_end_time(by_pipeline, d.assigned_task_queue[-1])
            for group in devices.values() for d in group if d.assigned_task_queue
        }

    @staticmethod
    def _print_finish_times_diff(finish_pre: dict, finish_post: dict) -> None:
        """Tabella finish_time PRE vs POST sort, celle colorate per device."""
        print("\n========== finish_time PRIMA vs DOPO il sort ==========")
        print(f"{'device':<8} | {'PRE':>10} | {'POST':>10} | {'Δ':>10}")
        delta_sum = 0
        for dev_id in finish_pre:
            pre   = finish_pre[dev_id]
            post  = finish_post[dev_id]
            color = color_for_device(dev_id)
            name_c  = colorize(f"{dev_id:<8}",          color)
            pre_c   = colorize(f"{pre:>10.4f}",         color)
            post_c  = colorize(f"{post:>10.4f}",        color)
            delta_c = colorize(f"{post - pre:>+10.4f}", color)
            print(f"{name_c} | {pre_c} | {post_c} | {delta_c}")
            delta_sum += (post - pre)

        if delta_sum < 0:    color_delta_sum = COLOR_GREEN
        elif delta_sum == 0: color_delta_sum = COLOR_CYAN
        else:                color_delta_sum = COLOR_RED
        print(colorize(f"\nΔ sum: {delta_sum}", color_delta_sum))
