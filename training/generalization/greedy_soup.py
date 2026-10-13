"""Aggregazione selettiva dei modelli per-scenario (greedy soup)."""
from training.generalization.core.abstract_eval import evaluate_state_dict
from training.generalization.fedavg import fedavg, load_specialists, save_aggregated

_METHOD = "greedy_soup"   # nome del metodo = sottocartella dell'output


def build_soup(state_dicts: list, names: list, eval_ids: list) -> tuple:
    """Classifica i modelli e costruisce la zuppa greedy."""
    scores = []
    for name, sd in zip(names, state_dicts):
        mean_cost, _ = evaluate_state_dict(sd, eval_ids)
        scores.append(mean_cost)
        print(f"[GREEDY-SOUP]   {name}: costo validazione {mean_cost:.1f}")

    order = sorted(range(len(state_dicts)), key=lambda i: scores[i])
    soup, soup_cost = [order[0]], scores[order[0]]
    print(f"[GREEDY-SOUP] parto dal migliore: {names[order[0]]} (costo {soup_cost:.1f})")

    for idx in order[1:]:
        candidate    = fedavg([state_dicts[i] for i in soup + [idx]])
        cand_cost, _ = evaluate_state_dict(candidate, eval_ids)
        if cand_cost < soup_cost:
            soup.append(idx)
            soup_cost = cand_cost
            print(f"[GREEDY-SOUP]   + {names[idx]} -> costo {cand_cost:.1f} (tengo)")
        else:
            print(f"[GREEDY-SOUP]   + {names[idx]} -> costo {cand_cost:.1f} (scarto)")

    return soup, soup_cost, scores


def run(config: dict) -> None:
    """Legge Settings.greedy_soup, costruisce la zuppa e salva l'aggregato."""
    gen      = config["GENERALIZATION"]
    settings = gen["Settings"][_METHOD]
    models_folder = gen["models_folder"]
    out_folder    = gen.get("output_folder", "generalization")
    n             = int(settings["n_models"])
    eval_ids      = [f"{settings['scenarios_folder']}/scenario_{i}"
                     for i in range(int(settings["n_scenarios"]))]

    print(f"[GREEDY-SOUP] {n} modelli ({models_folder}), "
          f"validazione su {len(eval_ids)} scenari ({settings['scenarios_folder']})")
    state_dicts = load_specialists(models_folder, n)
    names       = [f"scenario_{i}" for i in range(n)]

    soup, soup_cost, _scores = build_soup(state_dicts, names, eval_ids)
    aggregated = fedavg([state_dicts[i] for i in soup])

    print(f"[GREEDY-SOUP] zuppa finale: {', '.join(names[i] for i in soup)} "
          f"({len(soup)}/{n} modelli, costo {soup_cost:.1f})")
    save_aggregated(aggregated, models_folder, out_folder, _METHOD, "GREEDY-SOUP")
