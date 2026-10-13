"""Interpolazione tra il migliore specialista e la media."""
import torch

from training.generalization.core.abstract_eval import evaluate_state_dict
from training.generalization.fedavg import fedavg, load_specialists, save_aggregated

_METHOD = "lambda_scaling"          # nome del metodo = sottocartella dell'output
_GRID   = (0.0, 0.25, 0.5, 0.75, 1.0)   # griglia per lambda: "auto"


def interpolate(base: dict, target: dict, lam: float) -> dict:
    """θ(λ) = base + λ·(target − base), sui soli tensori float."""
    out = {}
    for k in base:
        if torch.is_floating_point(base[k]):
            out[k] = base[k].float() + lam * (target[k].float() - base[k].float())
        else:
            out[k] = base[k].clone()
    return out


def run(config: dict) -> None:
    """Legge Settings.lambda_scaling, interpola migliore->media e salva."""
    gen      = config["GENERALIZATION"]
    settings = gen["Settings"][_METHOD]
    models_folder = gen["models_folder"]
    out_folder    = gen.get("output_folder", "generalization")
    n             = int(settings["n_models"])
    lam           = settings.get("lambda", "auto")
    eval_ids      = [f"{settings['scenarios_folder']}/scenario_{i}"
                     for i in range(int(settings["n_scenarios"]))]

    print(f"[LAMBDA-SCALING] {n} modelli ({models_folder}), lambda={lam}, "
          f"validazione su {len(eval_ids)} scenari ({settings['scenarios_folder']})")
    state_dicts = load_specialists(models_folder, n)
    average     = fedavg(state_dicts)

    # Il migliore singolo fa da base dell'interpolazione.
    scores = []
    for i, sd in enumerate(state_dicts):
        mean_cost, _ = evaluate_state_dict(sd, eval_ids)
        scores.append(mean_cost)
        print(f"[LAMBDA-SCALING]   scenario_{i}: costo validazione {mean_cost:.1f}")
    best = min(range(n), key=lambda i: scores[i])
    print(f"[LAMBDA-SCALING] base: scenario_{best} (costo {scores[best]:.1f})")

    if lam == "auto":
        chosen, chosen_cost = None, float("inf")
        for candidate_lam in _GRID:
            sd = interpolate(state_dicts[best], average, candidate_lam)
            mean_cost, _ = evaluate_state_dict(sd, eval_ids)
            marker = ""
            if mean_cost < chosen_cost:
                chosen, chosen_cost = candidate_lam, mean_cost
                marker = "  <- migliore finora"
            print(f"[LAMBDA-SCALING]   lambda={candidate_lam:.2f} -> costo {mean_cost:.1f}{marker}")
        lam = chosen
        print(f"[LAMBDA-SCALING] scelto lambda={lam:.2f} (costo {chosen_cost:.1f})")

    aggregated = interpolate(state_dicts[best], average, float(lam))
    save_aggregated(aggregated, models_folder, out_folder, _METHOD, "LAMBDA-SCALING")
