"""costruzione dei Device per il TRAINING (MDP astratto)."""
from devices.core.device import GND, SAS, SAT
from devices.ground_server import GroundServer
from devices.satellite import Satellite
from devices.satellite_server import SatelliteServer


class _StubRaw:
    """Segnaposto di EdgeServer per il training: espone solo `name`."""

    def __init__(self, name: str):
        self.name = name


def build_abstract(device_specs: dict) -> dict:
    """Crea i device dallo scenario senza topologia (`.raw` stub)."""
    builders = {SAT: Satellite, SAS: SatelliteServer, GND: GroundServer}
    devices = {SAT: [], SAS: [], GND: []}

    for dev_id, spec in device_specs.items():
        # tipo dal prefisso dell'id ("SAT-1" -> SAT); "type" esplicito = override
        dev_type = spec.get("type") or dev_id.split("-")[0].upper()
        if dev_type not in (SAT, SAS, GND):
            raise ValueError(f"Device '{dev_id}': tipo '{dev_type}' non valido (SAT|SAS|GND).")
        device = builders[dev_type](_StubRaw(dev_id))
        device.scenario_id  = dev_id
        device.is_generator = bool(spec.get("generator", False))
        device.set_offline_windows(spec.get("offline", []))
        devices[dev_type].append(device)

    return devices
