"""Tipo device SAT: satellite comune (dalla topologia)."""
from devices.core.device import Device, SAT


class Satellite(Device):
    """SAT = un `EdgeServer` (satellite) dalla topologia orbitale."""
    device_type  = SAT
    power_exec_W = 15.0
