"""avvio del training MAPPO dal config."""
import architecture  # noqa: F401 - shim per gli import flat (import globals)

import globals as sim_globals


def run(config: dict = None):
    """Addestra uno specialista sullo scenario indicato in TRAINING."""
    config = config if config is not None else sim_globals.config
    cfg = dict(config.get("TRAINING", {}))

    if not cfg.get("scenario"):
        raise ValueError("TRAINING.scenario non indicato: serve lo scenario su cui addestrare.")

    # ep_override passa via CLI (train_secors lo legge da all_args, non dal cfg).
    args = []
    if cfg.get("ep_override") is not None:
        args += ["--ep_override", str(cfg["ep_override"])]

    from onpolicy.train.secors.train_secors import main
    main(args, training_cfg=cfg)


def run_generalization(config: dict = None):
    """Un modello solo per piu' scenari, secondo GENERALIZATION.method.

    random_domain e' un addestramento: un'unica run che campiona il pool.
    fedavg, greedy_soup e lambda_scaling partono invece SUBITO sui modelli
    per-scenario gia' presenti (errore se ne mancano).
    """
    config = config if config is not None else sim_globals.config
    gen = config.get("GENERALIZATION", {})
    method = gen.get("method", "fedavg")

    if method == "random_domain":
        from training.generalization.random_domain import run as run_random_domain
        run_random_domain(config)
        return

    from training.generalization import run_method
    run_method(config, method)
