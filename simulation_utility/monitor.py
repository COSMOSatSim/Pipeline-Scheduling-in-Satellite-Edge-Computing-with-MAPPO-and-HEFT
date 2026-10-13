"""ExecutionMonitor - responsabile unico della stampa durante il run delle pipeline."""
from devices.core.device import GND
from utility.converter_utility import s_to_ms
from utility.log_colors import COLOR_GREEN, COLOR_RED, color_for_device, colorize


class ExecutionMonitor:

    def __init__(
        self,
        show_event_table      = True,
        show_mission_column   = True,
        show_start_message    = False,
        show_end_message      = False,
        show_generated_queues = False,
        show_assigned_queues  = False,
        table_at_end          = True,
    ):
        self.show_event_table      = show_event_table
        self.show_mission_column   = show_mission_column
        self.show_start_message    = show_start_message
        self.show_end_message      = show_end_message
        self.show_generated_queues = show_generated_queues
        self.show_assigned_queues  = show_assigned_queues
        self.table_at_end          = table_at_end
        self._event_header_printed = False
        self._events = []                    # (tempo, riga formattata) da stampare al flush

    @staticmethod
    def from_config(config: dict) -> "ExecutionMonitor":
        """Crea il monitor leggendo la sezione SIMULATION.MONITOR della configurazione."""
        m = config.get("SIMULATION", {}).get("MONITOR", {})
        return ExecutionMonitor(
            show_event_table      = m.get("show_event_table",      True),
            show_mission_column   = m.get("show_mission_column",   True),
            show_start_message    = m.get("show_start_message",    False),
            show_end_message      = m.get("show_end_message",      False),
            show_generated_queues = m.get("show_generated_queues", False),
            show_assigned_queues  = m.get("show_assigned_queues",  False),
            table_at_end          = m.get("table_at_end",          True),
        )

    #-------------------------------------
    # TASK QUEUE
    #-------------------------------------

    def on_tasks_generated(self, satellite):
        """Chiamato dopo che un generatore ha creato le proprie missioni."""
        if self.show_generated_queues:
            self._stampa_coda(satellite, satellite.generated_task_queue, "GENERATA")

    def on_tasks_assigned(self, devices):
        """Chiamato dopo lo scheduling: stampa le code assegnate di tutti i device."""
        if self.show_assigned_queues:
            for group in devices.values():
                for device in group:
                    if device.assigned_task_queue:
                        self._stampa_coda(device, device.assigned_task_queue, "ASSEGNATA")

    def _stampa_coda(self, device, queue, label):
        print()
        print(f"  {'DEVICE':^21} | {'DAG ID':^28} | {'ORDINE':^6} | {'TASK':^10} | {'CODA':^10}")
        print(f"  {'-'*21:<21} | {'-'*28:<28} | {'-'*6:>6} | {'-'*10:<10} | {'-'*10:<10}")
        if not queue:
            print(f"  {device.id:<21} | {'':<28} | {'':>6} | {'vuota':<10} | {label:<10}")
            return
        for index, task in enumerate(queue, start=1):
            dag_id = task.dag.dag_id if task.dag is not None else ""
            print(f"  {device.id:<21} | {dag_id:<28} | {index:>6} | {task.node_id:<10} | {label:<10}")

    #-------------------------------------
    # SIMULATION EVENTS
    #-------------------------------------

    @staticmethod
    def _link_state(device, t_s: float) -> str:
        """'OFF' se il device è in una finestra offline al tempo t (s), 'ON' altrimenti."""
        return "OFF" if device.is_offline(s_to_ms(t_s)) else "ON"

    @classmethod
    def _link_cell(cls, device, t: float) -> str:
        """Cella LINK colorata: verde ON, rosso OFF."""
        if device.device_type == GND:
            return f"{'':<4}"
        state = cls._link_state(device, t)
        color = COLOR_GREEN if state == "ON" else COLOR_RED
        return colorize(f"{state:<4}", color)

    def _print_event_header(self):
        """Stampa l'header della tabella eventi la prima volta che serve."""
        if self._event_header_printed or not self.show_event_table:
            return
        self._event_header_printed = True
        print()
        mission_head = f"{'MISSION':^28} | " if self.show_mission_column else ""
        mission_rule = f"{'-' * 28:<28} | "  if self.show_mission_column else ""
        print(
            f"  {mission_head}{'TYPE':^4} | {'DEVICE':^21} | {'LINK':^4} | {'TIME':^14} | "
            f"{'STATUS':^11} | {'TASK':^10} | {'TO_DEVICE':^27} | {'TO_TASK':^10} | "
            f"{'TIME COST':^12} | {'ENERGY COST':^13}"
        )
        print(
            f"  {mission_rule}{'-' * 4:<4} | {'-' * 21:<21} | {'-' * 4:<4} | {'-' * 14:>14} | "
            f"{'-' * 11:<11} | {'-' * 10:<10} | {'-' * 27:<27} | {'-' * 10:<10} | "
            f"{'-' * 12:>12} | {'-' * 13:>13}"
        )

    def _emit(self, t_s: float, line: str):
        if self.table_at_end:
            self._events.append((t_s, line))
        else:
            self._print_event_header()
            print(line)

    def flush(self):
        """Stampa la tabella eventi intera (fine run), ordinata per tempo."""
        if not self._events:
            return
        print(f"\n{'=' * 60}")
        print(">>> TABELLA EVENTI PIPELINE <<<")
        print(f"{'=' * 60}")
        self._event_header_printed = False
        self._print_event_header()
        for _t, line in sorted(self._events, key=lambda e: e[0]):
            print(line)
        self._events.clear()

    @staticmethod
    def _to_device_cell(device, color) -> str:
        """Cella TO_DEVICE: `[TIPO] nome`."""
        return colorize(f"{f'[{device.device_type}] {device.id}':<27}", color)

    def _row(self, mission, device, t_s, status, task_label,
             to_device_cell, to_task_cell, cost_cell, energy_cell):
        """Compone una riga della tabella eventi (stampa o accumula)."""
        src           = mission.source_device
        mission_color = color_for_device(src.id if src else device.id)
        color         = color_for_device(device.id)
        prefix = (f"{colorize(f'{mission.dag_id:<28}', mission_color)} | "
                  if self.show_mission_column else "")
        line = (
            f"  {prefix}"
            f"{colorize(f'{device.device_type:<4}', color)} | "
            f"{colorize(f'{device.id:<21}', color)} | "
            f"{self._link_cell(device, t_s)} | "
            f"{colorize(f'{s_to_ms(t_s):.4f}ms'.rjust(14), color)} | "
            f"{colorize(f'{status:<11}', color)} | "
            f"{colorize(f'{task_label:<10}', color)} | "
            f"{to_device_cell} | {to_task_cell} | {cost_cell} | {energy_cell}"
        )
        self._emit(t_s, line)

    #-------------------------------------
    # MISSION EVENTS
    #-------------------------------------

    def on_mission_start(self, mission, env):
        if not self.show_start_message:
            return
        run_label = "pipeline" if mission.mission_type == "full_pipeline" else "missione"
        print(f"\n  Inizio {run_label}: id={mission.dag_id} | "
              f"tipo={mission.mission_type} | t={s_to_ms(env.now):.4f}ms")

    def on_mission_end(self, mission, env):
        if not self.show_end_message:
            return
        run_label = "pipeline" if mission.mission_type == "full_pipeline" else "missione"
        print(f"\n  Fine {run_label}: id={mission.dag_id} | tipo={mission.mission_type} | "
              f"t={s_to_ms(env.now):.4f}ms | makespan={s_to_ms(mission.get_makespan()):.4f}ms")

    #-------------------------------------
    # TASK EVENTS
    #-------------------------------------

    def on_task_start(self, task, env):
        if not self.show_event_table:
            return
        color = color_for_device(task.assigned_device.id)
        cost  = colorize(f"{task.d_cpu:.4f}ms".rjust(12), color)
        self._row(task.dag, task.assigned_device, env.now, "TASK START", task.node_id,
                  f"{'':27}", f"{'':10}", cost, f"{'':>13}")

    def on_task_end(self, task, env):
        if not self.show_event_table:
            return
        color  = color_for_device(task.assigned_device.id)
        energy = colorize(f"{task.energy_cpu_J:.4f}J".rjust(13), color)
        self._row(task.dag, task.assigned_device, env.now, "TASK END", task.node_id,
                  f"{'':27}", f"{'':10}", f"{'':>12}", energy)

    #-------------------------------------
    # TRANSMISSION EVENTS
    #-------------------------------------

    def on_tran_start(self, mission, source_device, source_label, target_task, d_tran_ms, env):
        if not self.show_event_table:
            return
        src_color = color_for_device(source_device.id)
        tgt_color = color_for_device(target_task.assigned_device.id)
        to_dev    = self._to_device_cell(target_task.assigned_device, tgt_color)
        to_task   = colorize(f"{target_task.node_id:<10}", tgt_color)
        cost      = colorize(f"{d_tran_ms:.4f}ms".rjust(12), src_color)
        self._row(mission, source_device, env.now, "TRAN START", source_label,
                  to_dev, to_task, cost, f"{'':>13}")

    def on_tran_end(self, mission, source_device, source_label, target_task, env):
        if not self.show_event_table:
            return
        tgt_color = color_for_device(target_task.assigned_device.id)
        to_dev    = self._to_device_cell(target_task.assigned_device, tgt_color)
        to_task   = colorize(f"{target_task.node_id:<10}", tgt_color)
        energy    = colorize(f"{target_task.energy_net_J:.4f}J".rjust(13),
                             color_for_device(source_device.id))
        self._row(mission, source_device, env.now, "TRAN END", source_label,
                  to_dev, to_task, f"{'':>12}", energy)
