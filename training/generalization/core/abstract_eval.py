"""Valutazione astratta di un actor su una lista di scenari."""
from devices.core.device import DeviceUtils
from scenario.scenario import ScenarioFactory
from scheduler.device.eft import EFTScheduler
from utility.file_utility import ENV_HP_PATH, load_json
from onpolicy.envs.secors import cost


def reward_weights() -> tuple:
    """Pesi (w_tempo, w_energia) del costo combinato, da secors.json5."""
    hp = load_json("secors.json5", base_path=ENV_HP_PATH)["PARAMETER SETTING OF SEC-MPH"]
    return hp["λ1"] * hp["β"], hp["λ2"] * (1 - hp["β"])


def evaluate_state_dict(state_dict: dict, scenario_ids: list) -> tuple:
    """Costo combinato medio dell'actor sugli scenari (build astratto)."""
    from scheduler.policy.mappo_scheduler import MAPPOScheduler

    w_t, w_e  = reward_weights()
    scheduler = MAPPOScheduler(state_dict=state_dict)   # actor riusato tra gli scenari
    detail    = []
    for sid in scenario_ids:
        devices, pipeline = ScenarioFactory.build(sid)
        for gen in DeviceUtils.generators(devices):
            gen.generate_task_queue(pipeline)
        scheduler.apply(devices, device_scheduler=EFTScheduler())
        c_t, c_e = cost.evaluate(devices)
        detail.append((sid, c_t, c_e, w_t * c_t + w_e * c_e))

    mean_cost = sum(d[3] for d in detail) / len(detail)
    return mean_cost, detail
