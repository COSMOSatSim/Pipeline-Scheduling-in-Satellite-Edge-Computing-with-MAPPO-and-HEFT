"""costruzione e generazione degli scenari."""
import os
import random
from datetime import timedelta

from devices.core.device import DeviceFactory
from mission.core.pipeline import PipelineUtils
from utility.converter_utility import s_to_ms
from utility.file_utility import DATA_PATH, load_json

ONLINE_OFFLINE_CSV = os.path.join(DATA_PATH, "online_offline.csv")

#######################################################################################################
# SCENARIO FACTORY - Costruisce device + pipeline da uno scenario self-contained
#######################################################################################################

class ScenarioFactory:
    """Costruisce device + pipeline da uno scenario self-contained."""

    @staticmethod
    def load(scenario_id: str) -> dict:
        """Legge lo scenario grezzo da config/scenarios/<id>.json5."""
        return load_json(os.path.join("scenarios", f"{scenario_id}.json5"))

    @staticmethod
    def build(scenario_id: str, env=None):
        """Ritorna (devices, pipeline) costruiti dallo scenario `scenario_id`."""
        spec = ScenarioFactory.load(scenario_id)
        if env is None:
            from training.abstract_devices import build_abstract
            devices = build_abstract(spec["devices"])
        else:
            devices = DeviceFactory.create_from_devices(spec["devices"], env)
        pipeline = PipelineUtils.load(spec["pipelines"])
        return devices, pipeline


#######################################################################################################
# SCENARIO BUILDER - osservazione orbitale + generazione scenari
#######################################################################################################

def _merge_intervals(intervals, eps=1e-6):
    """Fonde intervalli sovrapposti/adiacenti (list di (start, end))."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


class ScenarioBuilder:
    """Genera scenari a partire dalle dinamiche orbitali reali."""

    #-------------------------------------
    # FASE 1 - osservazione ON/OFF
    #-------------------------------------
    def collect_online_offline(self, t0, duration_s: float) -> dict:
        """Ritorna { sat_name: {"ON": [(s,e),...], "OFF": [(s,e),...]} } (secondi da t0)
        per i satelliti osservati dal GU nella finestra [t0, t0+duration_s]."""
        import architecture  # noqa: F401 - shim per gli import flat
        import globals as sim_globals
        from skyfield.api import EarthSatellite
        from user_based_topology import OBSERVER, TLE_DATA, ts

        cfg = sim_globals.config
        phi_max    = cfg["Phi_max"]                      # dome: elev > Phi_max
        buffer_phi = cfg["Phi_max"] - cfg["Phi_buffer"]  # buffer: buffer_phi < elev < Phi_max

        if not TLE_DATA:
            raise FileNotFoundError(
                "TLE non disponibili (data/tle_data.txt). Eseguire una volta con "
                "Build_Configurations:true, oppure fornire il file TLE."
            )

        t1 = ts.utc(t0.utc_datetime() + timedelta(seconds=duration_s))
        table = {}
        n_sat_tot = len(TLE_DATA) // 3
        print(f"[BUILDER] osservazione ON/OFF su {n_sat_tot} satelliti, finestra {duration_s:.0f}s "
              f"(dome>{phi_max}°, buffer {buffer_phi}-{phi_max}°)...")

        for idx in range(0, len(TLE_DATA) - 2, 3):
            name  = TLE_DATA[idx].strip()
            line1 = TLE_DATA[idx + 1].strip()
            line2 = TLE_DATA[idx + 2].strip()
            try:
                sat = EarthSatellite(line1, line2, name, ts)

                # Istanti di attraversamento (rise=0 / set=2) delle due soglie.
                boundaries = {t0.tt, t1.tt}
                for alt in (phi_max, buffer_phi):
                    times, events = sat.find_events(OBSERVER, t0, t1, altitude_degrees=alt)
                    for ti, ev in zip(times, events):
                        if ev in (0, 2):
                            boundaries.add(ti.tt)
                bts = sorted(b for b in boundaries if t0.tt <= b <= t1.tt)

                # Classifica ogni sotto-intervallo dall'elevazione al suo punto medio.
                on, off = [], []
                for a, b in zip(bts[:-1], bts[1:]):
                    mid = ts.tt_jd((a + b) / 2.0)
                    alt_deg = (sat - OBSERVER).at(mid).altaz()[0].degrees
                    start_s = (a - t0.tt) * 86400.0
                    end_s   = (b - t0.tt) * 86400.0
                    if alt_deg > phi_max:
                        on.append((start_s, end_s))
                    elif alt_deg > buffer_phi:
                        off.append((start_s, end_s))
                    # elev < buffer_phi -> non visibile: ignorato

                if on or off:
                    table[name] = {"ON": _merge_intervals(on), "OFF": _merge_intervals(off)}
            except Exception:
                continue  # satellite con TLE/propagazione problematici: saltato

        print(f"[BUILDER] satelliti osservati (in dome o buffer): {len(table)}.")
        return table

    @staticmethod
    def _windows_to_str(windows) -> str:
        """Finestre [(a,b),...] -> stringa 'a-b;c-d' (secondi, in una cella)."""
        return ";".join(f"{a:.3f}-{b:.3f}" for a, b in windows)

    @staticmethod
    def _str_to_windows(s: str) -> list:
        """Inverso di `_windows_to_str`: 'a-b;c-d' -> [(a,b),(c,d)]."""
        s = (s or "").strip()
        if not s:
            return []
        out = []
        for win in s.split(";"):
            a, b = win.split("-")
            out.append((float(a), float(b)))
        return out

    def save_online_offline_csv(self, table: dict, path: str = ONLINE_OFFLINE_CSV) -> str:
        """Scrive la tabella in CSV, UNA RIGA PER SATELLITE: satellite, ON, OFF
        (le finestre di ogni stato in un'unica cella, 'start-end' separate da ';')."""
        import csv as _csv
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["satellite", "ON", "OFF"])
            for name, states in table.items():
                w.writerow([name,
                            self._windows_to_str(states["ON"]),
                            self._windows_to_str(states["OFF"])])
        print(f"[BUILDER] tabella online/offline salvata: {path}")
        return path

    def load_online_offline_csv(self, path: str = ONLINE_OFFLINE_CSV) -> dict:
        """Rilegge la tabella dal CSV (per riusarla senza rifare l'osservazione)."""
        import csv as _csv
        table = {}
        with open(path, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                table[r["satellite"]] = {"ON":  self._str_to_windows(r["ON"]),
                                         "OFF": self._str_to_windows(r["OFF"])}
        return table

    #-------------------------------------
    # FASE 2 - generazione scenari
    #-------------------------------------
    def build_scenarios(self, table: dict, builder_cfg: dict) -> list:
        """Genera `n_scenarios` file scenario json5 dalla tabella."""
        n_sat = int(builder_cfg["n_sat"])
        n_sas = int(builder_cfg["n_sas"])
        n_gnd = int(builder_cfg["n_gnd"])
        n_gen = int(builder_cfg["n_generator"])
        pipeline = builder_cfg["pipeline"]
        seed = int(builder_cfg["seed"])
        n_scenarios = int(builder_cfg["n_scenarios"])
        out_folder = builder_cfg["output_folder"]

        if n_gen > n_sat:
            raise ValueError(f"n_generator ({n_gen}) deve essere <= n_sat ({n_sat}).")
        need = n_sat + n_sas
        observed = list(table.keys())
        if need > len(observed):
            raise ValueError(
                f"Servono {need} satelliti (n_sat+n_sas) ma ne sono stati osservati {len(observed)}."
            )

        out_dir = os.path.join("config", "scenarios", out_folder)
        os.makedirs(out_dir, exist_ok=True)
        paths = []

        for i in range(n_scenarios):
            rng = random.Random(seed + i)
            chosen = rng.sample(observed, need)          # satelliti distinti per scenario
            sat_names = chosen[:n_sat]
            sas_names = chosen[n_sat:need]

            devices = {}
            source  = {}
            for k, name in enumerate(sat_names, start=1):
                dev_id = f"SAT-{k}"
                devices[dev_id] = {"generator": k <= n_gen,
                                   "offline": self._offline_ms(table[name]["OFF"])}
                source[dev_id] = name
            for k, name in enumerate(sas_names, start=1):
                dev_id = f"SAS-{k}"
                devices[dev_id] = {"generator": False,
                                   "offline": self._offline_ms(table[name]["OFF"])}
                source[dev_id] = name
            for k in range(1, n_gnd + 1):
                devices[f"GND-{k}"] = {}

            scenario = {"devices": devices, "pipelines": [pipeline]}
            path = os.path.join(out_dir, f"scenario_{i}.json5")
            self._write_scenario(path, scenario, i, seed + i, source)
            paths.append(path)

        print(f"[BUILDER] generati {len(paths)} scenari in {out_dir} "
              f"(scenario_0 ... scenario_{n_scenarios - 1}).")
        return paths

    @staticmethod
    def _offline_ms(off_windows_s) -> list:
        """Converte le finestre OFF (secondi) in {start,end} in ms (interi)."""
        return [{"start": int(round(s_to_ms(a))), "end": int(round(s_to_ms(b)))}
                for a, b in off_windows_s]

    @staticmethod
    def _compact_device(dev: dict) -> str:
        """Un device su una riga sola, stile scenario scritto a mano."""
        if not dev:
            return "{}"
        parts = []
        if "generator" in dev:
            parts.append(f'"generator": {"true" if dev["generator"] else "false"}')
        if "offline" in dev:
            off_str = ", ".join(f'{{ "start": {w["start"]}, "end": {w["end"]} }}' for w in dev["offline"])
            parts.append(f'"offline": [{off_str}]')
        return "{ " + ", ".join(parts) + " }"

    # Separatore + etichette di sezione, identici agli scenari scritti a mano.
    _SEP = "        " + "/" * 38
    _GROUPS = [
        ("SAT", "SAT - satelliti comuni"),
        ("SAS", "SAS - satellite server"),
        ("GND", "GND - ground server"),
    ]

    @classmethod
    def _write_scenario(cls, path, scenario, index, seed, source):
        """Scrive lo scenario come json5 COMPATTO (una riga per device), raggruppato
        per tipo con i separatori di sezione, e con header (mappa device -> satellite)."""
        lines = [
            "// =========================================================================",
            f"// scenario_{index} - generato da ScenarioBuilder (seed {seed}).",
            "// Mappa device -> satellite reale osservato (offline = finestre buffer, in ms):",
        ]
        lines += [f"//   {dev_id}: {name}" for dev_id, name in source.items()]
        lines.append("// =========================================================================")
        lines.append("{")
        lines.append('    "devices": {')

        items = list(scenario["devices"].items())
        last_id = items[-1][0] if items else None
        for prefix, label in cls._GROUPS:
            group = [(d, v) for d, v in items if d.split("-")[0] == prefix]
            if not group:
                continue
            lines.append("")                       # riga vuota prima di ogni sezione
            lines.append(cls._SEP)
            lines.append(f"        // {label}")
            lines.append(cls._SEP)
            for dev_id, dev in group:
                comma = "" if dev_id == last_id else ","
                lines.append(f'        "{dev_id}": {cls._compact_device(dev)}{comma}')

        lines.append("    },")
        pipes = ", ".join(f'"{p}"' for p in scenario["pipelines"])
        lines.append(f'    "pipelines": [{pipes}]')
        lines.append("}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    #-------------------------------------
    # ENTRY - chiamato dal main quando CONTROL_PANEL.run è "scenarios"
    #-------------------------------------
    @classmethod
    def run(cls, config: dict) -> None:
        """Fase 1 + Fase 2: raccoglie la tabella (o la riusa se già presente) e genera gli scenari."""
        from user_based_topology import get_current_time

        builder = config["SCENARIO_BUILDER"]
        self = cls()

        if os.path.isfile(ONLINE_OFFLINE_CSV):
            print(f"[BUILDER] tabella online/offline già presente, riuso: {ONLINE_OFFLINE_CSV}")
            table = self.load_online_offline_csv(ONLINE_OFFLINE_CSV)
        else:
            duration = float(builder.get("observation_seconds", 180))  # finestra di osservazione (default 3 min)
            table = self.collect_online_offline(get_current_time(), duration)
            self.save_online_offline_csv(table, ONLINE_OFFLINE_CSV)

        self.build_scenarios(table, builder)
