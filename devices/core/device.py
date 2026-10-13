"""Strato device dell'integrazione (3 tipi: SAT / SAS / GND)."""

#######################################################################################################
# TIPI / REGISTRY
#######################################################################################################
SAT = "SAT"
SAS = "SAS"
GND = "GND"

# Alias con prefisso DEVICE_ attesi dal codice onpolicy vendorizzato (cost/energy.py).
DEVICE_SAT = SAT
DEVICE_SAS = SAS
DEVICE_GND = GND

# "Sempre offline" (end = Infinity nello scenario) clampato a un orizzonte finito (ms).
OFFLINE_RUN_HORIZON_MS = 1_000_000.0

# Nomi degli EdgeServer designati SAS (gli altri satelliti sono SAT). Lo imposta la Factory.
_SAS_NAMES: set = set()


#######################################################################################################
# DEVICE - wrapper base
#######################################################################################################

class Device:
    """Vista pulita di un nodo di calcolo (`EdgeServer`), con un `device_type`."""

    device_type  = SAT          # sovrascritto dalle sottoclassi
    # Potenze hardware (W, Tab.
    power_exec_W = 15.0         # potenza CPU
    power_tran_W = 5.0          # potenza di trasmissione

    def __init__(self, edge_server):
        self.raw = edge_server                  # il nodo avvolto (satellite o ground)
        self.scenario_id = None                 # nome nello scenario (es. "SAT-1")
        # Ruolo e scheduling (API del modello a pipeline)
        self.is_generator         = False
        self.generated_task_queue = []          # task generate da questo device
        self.assigned_task_queue  = []          # task assegnati
        self.offline_windows      = []          # finestre offline [(start_s, end_s), ...]

    # --- Identità / ruolo ---
    @property
    def id(self):
        return self.scenario_id or self.raw.name

    @property
    def device_id(self):
        """Alias di `id` (nome usato dal modello a pipeline, es. da EFT)."""
        return self.id

    @property
    def is_access_point(self) -> bool:
        return bool(getattr(self.raw, "is_acc_point", False))

    # --- Risorse di calcolo ed energia ---
    @property
    def energy(self) -> float:
        return self.raw.energy

    @property
    def cpu_capacity(self) -> float:
        """Capacità CPU (`C_sen` nel loro modello)."""
        return self.raw.C_sen

    def wait_cpu(self) -> float:
        return self.raw.W_cpu()

    def wait_net(self) -> float:
        return self.raw.W_net()

    # --- Geometria (per i satelliti; per il GND sono valori "neutri") ---
    @property
    def elev_angle(self) -> float:
        return getattr(self.raw, "elev_angle", 90.0)

    @property
    def orbital_sunset(self):
        return getattr(self.raw, "orbitalSunset", None)

    # --- Connettività ---
    def neighbors(self) -> list:
        """Vicini a 1 hop, avvolti col tipo corretto (SAT/SAS dal registry)."""
        return [wrap(n) for n in self.raw.get_neighbors()]

    def bandwidth_to(self, other: "Device"):
        return self.raw.get_bandwidth(other.raw)

    # --- Task (modello a pipeline) ---
    def generate_task_queue(self, pipelines_config: dict) -> list:
        """Crea la coda dei task generati da questo device (una pipeline per id)."""
        from mission.core.pipeline import PipelineFactory
        self.generated_task_queue = []
        for pipeline_id, pipeline_config in pipelines_config.items():
            pipeline = PipelineFactory.build(f"{self.id}_{pipeline_id}", pipeline_config)
            pipeline.source_device = self
            self.generated_task_queue.extend(pipeline.task_chunks.values())
        return self.generated_task_queue

    def get_generated_task_queue_dags(self) -> list:
        """Ritorna i DAG presenti nella coda delle task generate."""
        dags, seen = [], set()
        for task in self.generated_task_queue:
            if task.dag is None or id(task.dag) in seen:
                continue
            dags.append(task.dag)
            seen.add(id(task.dag))
        return dags

    def assign_task(self, task):
        """Assegna questo device al TaskChunk, calcola d_cpu e aggiunge alla coda."""
        if task.assigned_device is not None:
            raise ValueError(
                f"Il task {task.node_id} è già assegnato a "
                f"{task.assigned_device.id}, non può andare anche a {self.id}."
            )
        if not task.is_assignable:
            src = task.dag.source_device if task.dag is not None else None
            if src is not None and self is not src:
                raise ValueError(
                    f"Task '{task.node_id}' non assegnabile a {self.id}: "
                    f"is_assignable=False richiede source_device={src.id}."
                )
        task.assigned_device = self
        task.d_cpu           = self.get_d_cpu(task.operation, task.image_size_px)
        self.assigned_task_queue.append(task)

    # --- Tempi (tabelle di costo, in ms) ---
    def get_d_cpu(self, task_name, image_size_px) -> float:
        """Tempo di esecuzione CPU (ms) del task type su questo device."""
        from mission.core.task import TaskChunkUtils
        return TaskChunkUtils.execution_time(task_name, self.device_type, image_size_px)

    def get_transmission_time(self, predecessor_op, image_size_px, successor_device) -> float:
        """Tempo (ms) per trasmettere l'output di predecessor_op verso
        successor_device. 0 se stesso device o GND->GND."""
        from mission.core.task import TaskChunkUtils
        if self.id == successor_device.id:
            return 0.0
        if self.device_type == GND and successor_device.device_type == GND:
            return 0.0
        elif self.device_type == GND or successor_device.device_type == GND:
            link_type = "satellite_to_ground"
        else:
            link_type = "inter_satellite"
        return TaskChunkUtils.output_trasmissione_time(predecessor_op, link_type, image_size_px)

    # --- Finestre offline (vincolo sul link sat↔ground) ---
    def set_offline_windows(self, windows):
        """Imposta le finestre offline del device, in ms (l'unità del modello: tabelle di costo,
        EFT)."""
        if windows and isinstance(windows[0], (int, float)):
            windows = [windows]                       # coppia secca [s, e]
        parsed = []
        for window in windows:
            if isinstance(window, dict):
                start, end = float(window["start"]), float(window["end"])
            else:
                start, end = float(window[0]), float(window[1])
            parsed.append((start, min(end, OFFLINE_RUN_HORIZON_MS)))
        self.offline_windows = sorted(parsed)

    def is_offline(self, t_ms: float) -> bool:
        """True se il device è in una finestra offline al tempo `t_ms` (end escluso)."""
        return any(start <= t_ms < end for start, end in self.offline_windows)

    def _get_link_offline_windows(self, other_device):
        """Finestre offline rilevanti per il link verso other_device (solo sat↔ground)."""
        if self.device_type == GND and other_device.device_type != GND:
            return other_device.offline_windows
        if other_device.device_type == GND and self.device_type != GND:
            return self.offline_windows
        return []

    def get_available_transmission_start(self, requested_start, duration, other_device) -> float:
        """Primo istante in cui il link è disponibile per tutta la durata
        (rinvia oltre le finestre offline; blocca solo il link sat↔ground)."""
        start = float(requested_start)
        if duration <= 0:
            return start
        changed = True
        while changed:
            changed = False
            for offline_start, offline_end in self._get_link_offline_windows(other_device):
                if start < offline_end and start + duration > offline_start:
                    start   = offline_end
                    changed = True
                    break
        return start

    def reset(self):
        self.assigned_task_queue = []

    def __repr__(self):
        # In training i Device hanno `.raw` stub (niente energia): repr degradato.
        try:
            return f"{self.device_type}({self.id}, E={self.energy:.0f}J)"
        except AttributeError:
            return f"{self.device_type}({self.id})"


#######################################################################################################
# FACTORY - device espliciti dallo scenario
#######################################################################################################

class DeviceFactory:
    """Costruzione della costellazione dello scenario."""

    @staticmethod
    def create_from_devices(device_specs: dict, env=None) -> dict:
        """Crea i device dallo scenario (dict id -> {generator, offline}). - Il tipo si deduce dal
        prefisso dell'id ("SAT-1" -> SAT); un campo "type" esplicito fa da override. - SAT/SAS
        avvolgono gli `EdgeServer` della topologia, in ordine di dichiarazione (il nome dello
        scenario resta come `scenario_id`). - GND = `EdgeServer` creati a parte come nodi terrestri
        (serve l'`env` SimPy; niente orbita, sempre in vista). - `generator: true` = il device
        genera task (ammesso su qualunque tipo)."""
        global _SAS_NAMES
        import architecture  # noqa: F401 - attiva lo shim per gli import flat
        import globals as sim_globals
        from EdgeServer import EdgeServer

        from devices.ground_server import GroundServer
        from devices.satellite import Satellite
        from devices.satellite_server import SatelliteServer

        topology = [s for s in sim_globals.edge_servers if s is not None]
        builders = {SAT: Satellite, SAS: SatelliteServer}

        devices = {SAT: [], SAS: [], GND: []}
        topo_index = 0
        for dev_id, spec in device_specs.items():
            # tipo dal prefisso dell'id ("SAT-1" -> SAT); "type" esplicito = override
            dev_type = spec.get("type") or dev_id.split("-")[0].upper()
            if dev_type not in (SAT, SAS, GND):
                raise ValueError(f"Device '{dev_id}': tipo '{dev_type}' non valido (SAT|SAS|GND).")
            if dev_type == GND:
                edge   = EdgeServer(env, dev_id, None, float("inf"), False, 90.0)
                device = GroundServer(edge)
            else:
                if topo_index >= len(topology):
                    raise ValueError(
                        f"Lo scenario dichiara più satelliti ('{dev_id}', ...) di quanti "
                        f"ne offra la topologia ({len(topology)})."
                    )
                device = builders[dev_type](topology[topo_index])
                topo_index += 1
            device.scenario_id  = dev_id
            device.is_generator = bool(spec.get("generator", False))
            device.set_offline_windows(spec.get("offline", []))
            devices[dev_type].append(device)

        _SAS_NAMES = {d.raw.name for d in devices[SAS]}
        return devices


#######################################################################################################
# UTILS - query sull'insieme dei device
#######################################################################################################

class DeviceUtils:

    @staticmethod
    def type_of(name: str) -> str:
        """Tipo del device dal solo nome: SAS se promosso nel registry, GND se
        creato da noi (prefisso), SAT altrimenti (satellite di topologia)."""
        if name in _SAS_NAMES:
            return SAS
        if name.startswith("GND-"):
            return GND
        return SAT

    @staticmethod
    def device_list(devices: dict) -> list:
        """Lista piatta dei device in ordine fisso SAT-1...n | SAS | GND."""
        return (
            devices.get(SAT, []) +
            devices.get(SAS, []) +
            devices.get(GND, [])
        )

    @staticmethod
    def generators(devices: dict) -> list:
        """Device che generano task, in ordine."""
        return [d for d in DeviceUtils.device_list(devices) if d.is_generator]

    @staticmethod
    def collect_satellite_pipelines(devices: dict) -> list:
        """Raccoglie tutti i DAG generati dalla costellazione, senza duplicati."""
        dags, seen = [], set()
        for group in devices.values():
            for device in group:
                for dag in device.get_generated_task_queue_dags():
                    if id(dag) not in seen:
                        dags.append(dag)
                        seen.add(id(dag))
        return dags

    @staticmethod
    def candidates(device: "Device") -> list:
        """Candidati per un task su `device`: vicini a 1 hop + sé stesso."""
        return device.neighbors() + [device]
