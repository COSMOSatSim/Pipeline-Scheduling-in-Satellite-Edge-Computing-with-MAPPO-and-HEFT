"""Stato globale s_t per il modello SEC-RSM / SEC-MPH."""

from dataclasses import dataclass, field


@dataclass
class State:
    """Stato globale s_t = { NQ, CQ, CS, F } - visibile a tutti gli agenti."""
    NQ: list = field(default_factory=list)
    CQ: list = field(default_factory=list)
    CS: list = field(default_factory=list)
    F:  list = field(default_factory=list)

    def to_vector(self) -> list:
        return self.NQ + self.CQ + self.CS + [float(f) for f in self.F]
