"""Generalizzazione per randomizzazione del dominio."""
_METHOD = "random_domain"   # nome del metodo = sottocartella dell'output


def run(config: dict) -> None:
    """Legge TRAINING e GENERALIZATION e lancia il training randomizzato."""
    training = dict(config["TRAINING"])
    gen      = config["GENERALIZATION"]

    settings         = gen["Settings"]["random_domain"]
    scenarios_folder = settings["scenarios_folder"]               # sotto config/scenarios/
    n                = int(settings["n_scenarios"])               # scenario_0 ... scenario_{n-1}
    models_folder    = gen["models_folder"]                       # sotto data/models/mappo/
    out_folder       = gen.get("output_folder", "generalization") # sotto models_folder

    pool = [f"{scenarios_folder}/scenario_{i}" for i in range(n)]

    training["scenario"]      = pool[0]     # riferimento (forma di obs/azioni, T, run_info)
    training["scenario_pool"] = pool
    training["folder"]        = f"{models_folder}/{out_folder}/{_METHOD}"

    print(f"[RANDOM-DOMAIN] training su {n} scenari: {scenarios_folder}/scenario_0 ... scenario_{n - 1}")
    print(f"[RANDOM-DOMAIN] modello -> {training['folder']}")

    args = []
    if training.get("ep_override") is not None:
        args += ["--ep_override", str(training["ep_override"])]

    from onpolicy.train.secors.train_secors import main
    main(args, training_cfg=training)

    print(f"[RANDOM-DOMAIN] per usarlo -> SCHEDULER.policy_scheduler.mappo.model_folder: "
          f"\"{training['folder']}\"")
