"""Conversioni di unità."""

MS_PER_S = 1000.0


def ms_to_s(value_ms: float) -> float:
    """Millisecondi -> secondi."""
    return value_ms / MS_PER_S


def s_to_ms(value_s: float) -> float:
    """Secondi -> millisecondi."""
    return value_s * MS_PER_S
