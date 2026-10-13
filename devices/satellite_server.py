"""Tipo device SAS: satellite-server (dalla topologia, promosso)."""
from devices.core.device import Device, SAS


class SatelliteServer(Device):
    """SAS = un `EdgeServer` (satellite) promosso a "server" (sottoinsieme designato)."""
    device_type  = SAS
    power_exec_W = 25.0
