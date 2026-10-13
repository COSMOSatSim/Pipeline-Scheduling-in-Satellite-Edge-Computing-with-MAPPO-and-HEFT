"""Osservazione locale o_t^e per il modello SEC-RSM / SEC-MPH."""

from dataclasses import dataclass, field


@dataclass
class Observation:
    """Osservazione locale del satellite e al passo t = { ND, CQ, sat_cs, sat_offline }."""
    ND:          list  = field(default_factory=list)
    CQ:          list  = field(default_factory=list)
    sat_cs:      float = 0.0
    sat_offline: bool  = False

    def to_vector(self) -> list:
        return self.ND + self.CQ + [self.sat_cs, float(self.sat_offline)]
