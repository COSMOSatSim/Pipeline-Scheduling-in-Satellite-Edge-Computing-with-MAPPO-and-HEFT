"""Tipo device GND: server di calcolo a terra (creato da noi)."""
from devices.core.device import Device, GND


class GroundServer(Device):
    """GND = nodo di calcolo a terra (EdgeServer "a terra" creato da noi, non in topologia)."""
    device_type  = GND
    power_exec_W = 220.0
