"""cost/ - Calcolatori dei costi SEC-RSM."""

#######################################################################################################
# IMPORT
#######################################################################################################
from onpolicy.envs.secors.cost.execution_time import compute_CT
from onpolicy.envs.secors.cost.energy         import compute_CE

#######################################################################################################
# EVALUATE - wrapper (CT, CE)
#######################################################################################################

def evaluate(devices: dict) -> tuple[float, float]:
    """Ritorna (CT, CE) per uno scheduling già prodotto."""
    return compute_CT(devices), compute_CE(devices)


__all__ = ["compute_CT", "compute_CE", "evaluate"]
