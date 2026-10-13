"""Orchestrazione delle pipeline nel simulatore."""
import architecture  # noqa: F401 - attiva lo shim per gli import flat

import globals as sim_globals

from devices.core.device import DeviceUtils
from scenario.scenario import ScenarioFactory
from scheduler.core.scheduler import SchedulerFactory
from simulation_utility.runtime.device_runtime import DeviceRuntime

_run_state = None      # (monitor, devices) dell'ultimo generate_pipelines


def generate_pipelines(env, scenario_id=None):
    """Prepara e avvia il run a pipeline (fasi 1-3)."""
    global _run_state
    scenario_id = scenario_id or (sim_globals.config.get("SIMULATION", {})
                                  .get("SCENARIO", {}).get("scenario", "base"))
    devices, pipeline_cfgs = ScenarioFactory.build(scenario_id, env)

    from simulation_utility.monitor import ExecutionMonitor
    from simulation_utility.utility.print_utility import print_pre_sim
    monitor = ExecutionMonitor.from_config(sim_globals.config)
    print_pre_sim(sim_globals.config, pipeline_cfgs, n_generators=len(DeviceUtils.generators(devices)))

    # FASE 1 - generazione
    for generator in DeviceUtils.generators(devices):
        generator.generate_task_queue(pipeline_cfgs)
        monitor.on_tasks_generated(generator)

    # FASE 2 - scheduling policy + EFT (sempre: sequenzia le code rispettando l'ordine topologico
    # globale; senza, la FIFO del runtime può stallare)
    from scheduler.device.eft import EFTScheduler
    policy_scheduler = SchedulerFactory.build_policy(sim_globals.config)
    policy_scheduler.apply(devices, device_scheduler=EFTScheduler())
    monitor.on_tasks_assigned(devices)

    # FASE 3 - un DeviceRuntime per device dello scenario
    registry = {}
    for device in DeviceUtils.device_list(devices):
        registry[device] = DeviceRuntime(device, registry, monitor)
    for runtime in registry.values():
        env.process(runtime.run(env))

    _run_state = (monitor, devices)
    n_pipelines  = len(DeviceUtils.collect_satellite_pipelines(devices))
    n_generators = len(DeviceUtils.generators(devices))
    print(f"[PIPE] avviate {n_pipelines} pipeline (scenario '{scenario_id}', "
          f"{n_generators} generatori, policy {type(policy_scheduler).__name__}).")


def finalize():
    """Da chiamare a fine simulazione: stampa la tabella eventi (table_at_end)
    e salva i risultati (sezione SAVE)."""
    if _run_state is None:
        return
    monitor, devices = _run_state
    monitor.flush()
    satellite_pipelines = DeviceUtils.collect_satellite_pipelines(devices)

    from simulation_utility.utility.print_utility import print_post_sim
    print_post_sim(sim_globals.config, satellite_pipelines)

    from simulation_utility.save import save_results_sections
    save_results_sections(sim_globals.config, devices, satellite_pipelines)
