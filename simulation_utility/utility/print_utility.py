"""Stampe pre/post simulazione (info di configurazione e tabelle di riepilogo per missione /
pipeline / costellazione)."""
from utility.converter_utility import s_to_ms
from utility.log_colors import color_for_device, colorize


def _pipeline_color(pipeline):
    """Colore della pipeline = colore del suo source_device (stesso usato dal monitor)."""
    src_id = pipeline.source_device.device_id if pipeline.source_device else pipeline.dag_id
    return color_for_device(src_id)


def _ms(t_s) -> str:
    return f"{s_to_ms(t_s):.4f}ms" if t_s is not None else "N/A"


#######################################################################################################
# MAIN SECTIONS
#######################################################################################################
def print_pre_sim(config: dict, pipelines_config: dict, n_generators: int = None):
    """Stampa le info di configurazione prima della simulazione."""
    if config.get("SIMULATION", {}).get("PRINT", {}).get("print_simulation_info", True):
        print_simulation_info(config, pipelines_config, n_generators)


def print_post_sim(config: dict, satellite_pipelines: list):
    """Stampa i risultati dopo la simulazione."""
    print_config = config.get("SIMULATION", {}).get("PRINT", {})

    _print_results(satellite_pipelines)

    if print_config.get("print_pipeline_summaries", False):
        print(f"\n{'='*60}")
        print("DETTAGLIO TASK")
        print(f"{'='*60}")
        for pipeline in satellite_pipelines:
            if hasattr(pipeline, "summary"):
                pipeline.summary()
    print()

#######################################################################################################
# RESULTS
#######################################################################################################
def _print_results(satellite_pipelines: list):
    print(f"\n{'='*60}")
    print("RISULTATI FINALI")
    print(f"{'='*60}")

    SEP = f"  {'-'*28}-+-{'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}"

    print(f"\n  {'-'*118}")
    print(f"  {'DETTAGLIO PER MISSIONE':^118}")
    print(f"  {'-'*118}")
    print(f"  {'PIPELINE':^28} | {'MISSION':^25} | {'INIZIO':^12} | {'FINE':^12} | {'DURATA':^12} | {'ENERGIA':^14}")
    print(SEP)
    for i, pipeline in enumerate(satellite_pipelines):
        if i > 0:
            print(SEP)
        color = _pipeline_color(pipeline)
        for j, (mission_type, ids) in enumerate(pipeline.mission_task_ids.items()):
            tasks  = [pipeline.task_chunks[nid] for nid in ids]
            energy = sum(t.energy_cpu_J + t.energy_net_J for t in tasks)
            if tasks and all(t.start_time is not None and t.end_time is not None for t in tasks):
                m_start = min(t.start_time for t in tasks)
                m_end   = max(t.end_time   for t in tasks)
                start_str, end_str = _ms(m_start), _ms(m_end)
                dur_str = _ms(m_end - m_start)
            else:
                start_str = end_str = dur_str = "N/A"
            pid       = pipeline.dag_id if j == 0 else ""
            pid_c     = colorize(f"{pid:<28}",             color)
            mt_c      = colorize(f"{mission_type:<25}",    color)
            start_c   = colorize(f"{start_str:>12}",       color)
            end_c     = colorize(f"{end_str:>12}",         color)
            dur_c     = colorize(f"{dur_str:>12}",         color)
            energy_c  = colorize(f"{energy:>13.4f}J",      color)
            print(f"  {pid_c} | {mt_c} | {start_c} | {end_c} | {dur_c} | {energy_c}")

    SEP2 = f"  {'-'*28}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}"

    print(f"\n  {'-'*90}")
    print(f"  {'RIEPILOGO PER PIPELINE':^90}")
    print(f"  {'-'*90}")
    print(f"  {'PIPELINE':^28} | {'INIZIO':^12} | {'FINE':^12} | {'DURATA':^12} | {'ENERGIA':^14}")
    print(SEP2)
    for pipeline in satellite_pipelines:
        total_energy = sum(t.energy_cpu_J + t.energy_net_J for t in pipeline.task_chunks.values())
        makespan     = pipeline.get_makespan()
        color    = _pipeline_color(pipeline)
        pid_c    = colorize(f"{pipeline.dag_id:<28}",         color)
        start_c  = colorize(f"{_ms(pipeline.start_time):>12}", color)
        end_c    = colorize(f"{_ms(pipeline.end_time):>12}",   color)
        dur_c    = colorize(f"{(_ms(makespan) if makespan is not None else 'N/A'):>12}", color)
        energy_c = colorize(f"{total_energy:>13.4f}J",        color)
        print(f"  {pid_c} | {start_c} | {end_c} | {dur_c} | {energy_c}")

    starts = [p.start_time for p in satellite_pipelines if p.start_time is not None]
    ends   = [p.end_time   for p in satellite_pipelines if p.end_time   is not None]
    if starts and ends:
        constellation_start = min(starts)
        constellation_end   = max(ends)
        constellation_time  = constellation_end - constellation_start
    else:
        constellation_start = constellation_end = constellation_time = None
    constellation_energy     = sum(t.energy_cpu_J + t.energy_net_J
                                   for p in satellite_pipelines for t in p.task_chunks.values())
    constellation_energy_sat = sum(
        t.energy_cpu_J + t.energy_net_J
        for p in satellite_pipelines for t in p.task_chunks.values()
        if t.assigned_device and t.assigned_device.device_type != "GND"
    )

    SEP3 = f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*16}-+-{'-'*14}"
    print(f"\n  {'-'*78}")
    print(f"  {'TOTALE COSTELLAZIONE':^78}")
    print(f"  {'-'*78}")
    print(f"  {'INIZIO':^12} | {'FINE':^12} | {'DURATA':^12} | {'ENERGIA(SAT+SAS)':^16} | {'ENERGIA TOT':^14}")
    print(SEP3)
    ct_str = _ms(constellation_time) if constellation_time is not None else "N/A"
    print(f"  {_ms(constellation_start):>12} | {_ms(constellation_end):>12} | {ct_str:>12} | "
          f"{f'{constellation_energy_sat:.4f}J':^16} | {f'{constellation_energy:.4f}J':^14}")

#######################################################################################################
# SIMULATION INFO
#######################################################################################################
def print_simulation_info(config: dict, pipelines_config: dict, n_generators: int = None):
    pipeline_types = ", ".join(pipelines_config.keys())
    pipeline_sizes = ", ".join(
        f"{pipeline_config['image_size_px']}px"
        for pipeline_config in pipelines_config.values()
    )
    sim_config = config.get("SIMULATION", {})
    scheduler_policy = sim_config.get("SCHEDULER", {}).get("policy_scheduler", {}).get("selected", "?")
    # Istanze = una pipeline per generatore (per ogni tipo dichiarato).
    n_instances = (n_generators * len(pipelines_config)) if n_generators is not None else len(pipelines_config)
    print(f"\n{'='*60}")
    print("Simulation Info")
    print(f"{'='*60}")
    print(f"Seed                : {config.get('seed', '?')}")
    print(f"Scheduling policy   : {scheduler_policy} (+ EFT)")
    print(f"Pipelines (istanze) : {n_instances}")
    print(f"Tipi pipeline       : {pipeline_types}")
    print(f"Pipeline image px   : {pipeline_sizes}")
    print(f"Scenario            : {sim_config.get('SCENARIO', {}).get('scenario', '?')}")
