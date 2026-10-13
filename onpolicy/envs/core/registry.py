"""Registro degli environment selezionabili per il training."""


def get_env_class(name: str):
    """Ritorna la classe dell'environment dato il suo nome."""
    if name == "secors":
        from onpolicy.envs.secors.SECORS_env import SECORSEnv
        return SECORSEnv
    raise ValueError(f"Environment '{name}' non riconosciuto. Registrati: ['secors'].")
