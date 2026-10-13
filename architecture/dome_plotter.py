from enum import StrEnum, auto
import json
import sys
import os
from turtle import pos
import json5
from matplotlib import cm, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

PATH = "Generated_datasets/simulation_dataset.json"
PATH_TASK = "Generated_datasets/generated_tasks_counter.json5"
PLOT_PATH = "Dome_Plots/"

class Mode(StrEnum):
    NORMAL = auto()
    BATTERY = auto()
    COMPLETED_TASK = auto()


gradient = LinearSegmentedColormap.from_list(
    "green_red_smooth",
    [   
        "#228b22",   # Satellite con pochi Task completati - verde scuro
        "#FFD700",   
        "#ffa500",
        "#ff0000",
        "#000000"   # Satellite con più Task completati
    ]
)

# Ensure output directory exists
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH, exist_ok=True)
elif not os.path.isdir(PLOT_PATH):
    raise RuntimeError(f"Path '{PLOT_PATH}' exists and is not a directory")

def clean_name(name: str) -> str:
    return name.replace("STARLINK", "").replace("[DTC]", "").strip()

def open_generated_dataset(config_path: str) -> dict:
    with open(config_path, "r") as file:
        simulation_dataset = json.load(file)
    return simulation_dataset


def count_completed_tasks(tasks: list) -> dict:
    completed_count = {}
    for task in tasks:
        sat_name = task.get("execution_server")
        if sat_name not in completed_count:
            completed_count[sat_name] = 0
        else:
            completed_count[sat_name] += 1
    
    #print(completed_count)
    if completed_count:
        top_key, top_value = max(completed_count.items(), key=lambda kv: kv[1])
        print("Satellite con più task completati:", top_key, "con", top_value, "task")
        completed_count["_top"] = {"satellite": top_key, "count": top_value}
    else:
        completed_count["_top"] = {"satellite": None, "count": 0}
    return top_value


def plot_dome(snapshot: dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = [satellite["position"][0] for satellite in snapshot]
    ys = [satellite["position"][1] for satellite in snapshot]
    zs = [satellite["position"][2] for satellite in snapshot]

    ax.scatter(xs, ys, zs)

    for satellite in snapshot:
        x, y, z = satellite["position"]
        ax.text(x, y, z, satellite["satellite"])

    plt.savefig(f"{PLOT_PATH}dome_plot.png")
    plt.close()

def expand_axes(ax, factor=1.5):
    """
    Aumenta le dimensioni del cubo 3D espandendo i limiti degli assi.
    factor > 1 ingrandisce il cubo.
    """
    x_min, x_max = ax.get_xlim3d()
    y_min, y_max = ax.get_ylim3d()
    z_min, z_max = ax.get_zlim3d()

    # Centri
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    # Range
    x_range = (x_max - x_min) * factor / 2
    y_range = (y_max - y_min) * factor / 2
    z_range = (z_max - z_min) * factor / 2

    ax.set_xlim3d(x_mid - x_range, x_mid + x_range)
    ax.set_ylim3d(y_mid - y_range, y_mid + y_range)
    ax.set_zlim3d(z_mid - z_range, z_mid + z_range)

    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='z', labelsize=6)

def zoom_3d(ax: plt.Axes, factor: float):
    """
    Zoom su un plot 3D riducendo il range degli assi.

    - factor > 1 → zoom in
    - factor < 1 → zoom out
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_mid = sum(x_limits) / 2
    y_mid = sum(y_limits) / 2
    z_mid = sum(z_limits) / 2

    x_range = (x_limits[1] - x_limits[0]) / factor
    y_range = (y_limits[1] - y_limits[0]) / factor
    z_range = (z_limits[1] - z_limits[0]) / factor

    ax.set_xlim3d(x_mid - x_range / 2, x_mid + x_range / 2)
    ax.set_ylim3d(y_mid - y_range / 2, y_mid + y_range / 2)
    ax.set_zlim3d(z_mid - z_range / 2, z_mid + z_range / 2)


def set_3d_view(ax, elev: int = 30, azim: int = 45):
    """
    Imposta la prospettiva di un grafico 3D.

    Parametri:
    - ax: l'oggetto Axes3D
    - elev: angolo di elevazione (gradi), 0 = piano xy
    - azim: angolo azimutale (gradi), rotazione attorno all'asse z
    - distance: opzionale, distanza della “camera” dal centro (float)
    """
    ax.view_init(elev=elev, azim=azim)


def _set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    This makes spheres/positions look correct instead of distorted.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_mid = sum(x_limits) / 2.0
    y_mid = sum(y_limits) / 2.0
    z_mid = sum(z_limits) / 2.0

    ax.set_xlim3d(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim3d(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim3d(z_mid - max_range / 2, z_mid + max_range / 2)


def plot_dome_network(snapshot: dict, save_path: str, mode: Mode):
    """Plot satellites as points and draw lines to their neighbors.

    - `snapshot` is the list of satellite dicts (each must have `satellite` and `position`).
    - neighbor names are read from `satellite['neighbors_metrics'][i]['name']`.

    The function avoids drawing duplicate edges by tracking drawn pairs.
    """

    sat_info = {sat.get("satellite"): tuple(sat.get("position")) for sat in snapshot}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")


    # Draw edges (avoid duplicates)
    drawn = set()  # Coppia già disegnata
    for sat in snapshot:
        name = sat.get("satellite")
        pos = sat.get("position")
        if mode == Mode.NORMAL:
            ax.scatter(pos[0], pos[1], pos[2], s=5, c="tab:blue", alpha=0.8)         # Node point 

        elif mode == Mode.BATTERY:
            battery_level = sat.get("energy_budget_J", 100)
            max_battery = dataset.get("metadata").get("battery_life")
            norm = float(battery_level) / float(max_battery) 
            print(f"{name} : battery {norm}")
            cmap = plt.cm.RdYlGn(norm)
            ax.scatter(pos[0], pos[1], pos[2], s=5, c=[cmap], alpha=0.8)  # Node point

        elif mode == Mode.COMPLETED_TASK:
            executed_tasks = sat.get("completed_tasks_count")
            norm = executed_tasks / MAX_TASKS_VALUE
    
            color = gradient(norm)
            ax.scatter(pos[0], pos[1], pos[2], s=5, c=[color], alpha=0.8)  # Node point
            



        ax.text(pos[0], pos[1], pos[2], clean_name(name), fontsize=4, alpha=0.8)  # Name label

        p1 = (pos[0], pos[1], pos[2])
        neighbors = sat.get("neighbors_metrics")

        for neigh in neighbors:
            nname = neigh.get("name")
            p2 = sat_info.get(nname)
            if p2 :
                #print(f"{nname} p2: {p2}")
                link = tuple(sorted((name, nname)))
                if link in drawn:
                    continue
                drawn.add(link)
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color="gray",
                    linewidth=0.4,
                    linestyle='--',
                    alpha=0.5,
                )
    # Legenda
    norm_obj = Normalize(vmin=0, vmax=1)     # la tua norma va da 0 a 1
    scalar_map = cm.ScalarMappable(norm=norm_obj, cmap=gradient)
    scalar_map.set_array([])  # necessario per pyplot
    
    cbar = plt.colorbar(scalar_map, ax=ax, shrink=0.3)  # shrink riduce la dimensione
    cbar.set_label("Task completati", fontsize=8)

    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Pochi task", "Tanti task"])

    _set_axes_equal(ax)
    set_3d_view(ax, elev=30, azim=30)   # Gestore della view 
    expand_axes(ax, factor=1.3)         # Gestore dello zoom degli assi 
    zoom_3d(ax, factor=1.7)             # Gestore dello zoom dei punti
    #plt.show()
    plt.savefig(save_path, dpi=600)
    plt.close()

try:
    with open(PATH_TASK, "r") as f:
        tasks_list = json5.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(f"Errore: Impossibile aprire o decodificare il file '{PATH_TASK}'.")

MAX_TASKS_VALUE = count_completed_tasks(tasks_list) # un satellite ha eseguito questo numero massimo di task in tutta la simulazione

dataset = open_generated_dataset(PATH)
mode = Mode.COMPLETED_TASK

plot_path = f"{PLOT_PATH}_TEST_legend_{mode}_.png"
satellites_snapshot = dataset["snapshots"][0]["satellites"]
plot_dome_network(satellites_snapshot, plot_path, mode)

# for i, snapshot in enumerate(dataset["snapshots"]):
#     print(f"Processing {i}")

#     plot_path = f"{PLOT_PATH}{i}_Dome_network_{mode}_.png"
#     satellites_snapshot = snapshot["satellites"]
#     plot_dome_network(satellites_snapshot, plot_path, mode)

