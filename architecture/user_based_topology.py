"""
Module for user-based topology management in a LEO satellite constellation.

This module provides functions to create and manage the satellite topology
based on the user's position on Earth. It includes utilities for loading TLE
data, computing satellite proximities, and filtering satellites based on
visibility and access point criteria.
"""

import json5
from math import sqrt
import math
import numpy as np
from datetime import timedelta

from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time
from skyfield.toposlib import GeographicPosition

from Satellite import Satellite
import sys
from enums import AccPointMode, Topology, tle_data
import globals
from utils import colorize

# Leggi il file di configurazione JSON
config = globals.config

ts = load.timescale()  # ts : time management with astronomical time
time_now = ts.now()


# reading TLE DATA from File
def loadTLEFromFile(filename):
    try:
        with open(filename, "r") as file:
            tle_data = file.read().splitlines()
        return tle_data
    except FileNotFoundError:
        print(colorize(f"[TLE] Il file {filename} non è stato trovato.", "red"))
        return None
    except IOError as e:
        print(colorize(f"[TLE] Errore nella lettura del file: {e}", "red"))
        return None


TLE_DATA = loadTLEFromFile("./data/tle_data.txt")  # Load TLE Data
# ---------------------------------------------------------------------------- #
#                                    Printer                                   #
# ---------------------------------------------------------------------------- #


def printSatList(*sats):
    for l in sats:
        print("-" * 20)
        [print(f"SATELLITE: {s[0].name} Distance: {s[1]}") for s in l]


# ---------------------------------------------------------------------------- #
#                                    Getter                                    #
# ---------------------------------------------------------------------------- #


def get_current_time() -> Time:
    """
    Get the current time using Skyfield's timescale.

    This function loads the timescale from the Skyfield library and returns the current time.

    Returns:
        skyfield.timelib.Time: The current time according to Skyfield's timescale.
    """
    ts = load.timescale()  # Carica la scala temporale di Skyfield
    return ts.now()


def advance_time(current_time, minutes_to_add):
    """
    Advance the given Skyfield Time object by a specified number of minutes.

    Args:
        current_time (skyfield.timelib.Time): The current time from ts.now().
        minutes_to_add (int): The number of minutes to add to the current time.

    Returns:
        skyfield.timelib.Time: The new time advanced by the specified number of minutes.
    """
    # Convert the Skyfield Time object to a datetime object
    current_datetime = current_time.utc_datetime()

    # Add the specified number of minutes
    new_datetime = current_datetime + timedelta(minutes=minutes_to_add)

    # Convert the new datetime object back to a Skyfield Time object
    ts = load.timescale()
    new_time = ts.from_datetime(new_datetime)

    return new_time


# Getter Topos 'Observer' object
def getObserverObj(location=config["simulation_location"]):
    if location in config["locations"]:
        lat = config["locations"][location]["lat"]
        lon = config["locations"][location]["lon"]
        return wgs84.latlon(lat, lon)
    else:
        sys.exit(
            f"Errore: The User Position '{location}' not found in the config file."
        )


OBSERVER = getObserverObj()


# Getter reference system from a Satellite
def getSystemFromSat(satellite, time, Geocentric=False):
    """
    Get the reference system from a satellite.

    Args:
        satellite (EarthSatellite): The satellite object.
        Geocentric (bool): If False, return the system relative to the observer's point of view.
                           If True, return the satellite's position relative to the center of the Earth (Geocentric).
        time (Time): The time at which to get the satellite's position.
    Returns:
        Geocentric or Topocentric: The reference system of the satellite at the given time.
    """
    if Geocentric:  # Satellite Position from the Center of Earth
        return satellite.at(time)  # return geocentric system
    else:  # Satellite Position from the Observer Position
        difference = (
            satellite - OBSERVER
        )  # Calculate the difference between the satellite's position and the observer's position to get the topocentric reference system
        return difference.at(time)  # return topocentric system


def get_pos_proximity(pos1, pos2):
    """
    Calcola la distanza fra due punti in uno spazio tridimensionale
    Args:
        pos1: Vettore posizionale dell'obj1
        pos2: Vettore posizionale dell'obj2
    :return: lunghezza del segmento obj1 -> obj2
    """
    # Extraction of coordinate components
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2

    # Calculate the Euclidean distance
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def get_orbit_proximity(sat1, sat2, t):
    """
    Calculates the distance between two satellites in kilometers.

    Args:
        satellite1: EarthSatellite object representing the first satellite.
        satellite2: EarthSatellite object representing the second satellite.
        timestamp: Skyfield Time object representing the moment of calculation.

    Returns: The distance between the two satellites in kilometers.
    """
    # Get geocentric positions in km
    pos_sat1 = getSystemFromSat(sat1, t, True).position.km
    pos_sat2 = getSystemFromSat(sat2, t, True).position.km

    # Extraction of coordinate components
    x1, y1, z1 = pos_sat1
    x2, y2, z2 = pos_sat2

    return sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    )  # Calculate the Euclidean distance


def are_satellites_equal(sat1, sat2):
    """
    Compare two satellites to check if they are the same based on their name and satellite number.

    Args:
        sat1: EarthSatellite object representing the first satellite.
        sat2: EarthSatellite object representing the second satellite.

    Returns:
        bool: True if the satellites are the same, False otherwise.
    """
    # Confronta per nome e numero satnum
    return sat1.name == sat2.name and sat1.model.satnum == sat2.model.satnum


def getLatency(distance: float):
    LIGHT_SPEED = 299792458  # m/s
    # convert distance to meters
    distance_m = distance * 1000

    return distance_m / LIGHT_SPEED  # Calculate latency in seconds


# ---------------------------------------------------------------------------- #
#                                    Filter                                    #
# ---------------------------------------------------------------------------- #


def filterSatellitesInView(satellite, t):
    """
    Input: Satellites
    Output: Boolean value | True : sat is coming in our direction
                          | False : sat is not coming in our direction
    """
    sys = getSystemFromSat(satellite, t)

    velocity = sys.velocity

    r = sys.position.km  # r è la posizione relativa del SAT rispetto all'observer
    r_unit = r / globals.np.linalg.norm(r)  # Vettore unitario
    v_rel = globals.np.dot(
        velocity.km_per_s, r_unit
    )  # Calcoliamo la velocità calcolando il prodotto scalare tra r e r_unit

    return True if v_rel < 0 else False


def getAllSatOnPoint(
    t: Time,
    tle_data: tle_data,
    point_pos: GeographicPosition,
    Phi_max: float,
    Phi_buffer: float,
    num_acc_point: int,
    ap_mode: AccPointMode = AccPointMode.BASE,
) -> tuple[list[Satellite], list[Satellite]]:
    """
    This function return all the satellite above a given point

    Args:
        t (Time): time of observation.
        tle_data (tle_data): TLE data of the satellites.
        point_pos (GeographicPosition): position of specific point.
        Phi_max (float): Elevation Angle of a given point to search satellite in that area.
        num_acc_point (int): Number of acc point we are searching in that area
        ap_mode (AccPointMode): Access Point Mode to get Access Point.
    Returns:
        topology (topology)
    """
    buffer_Phi = Phi_max - Phi_buffer  # Angle of a Buffer Zone
    sats_dome, sats_buffer = [], []

    for i in range(0, len(tle_data), 3):
        name = tle_data[i].strip()
        line1 = tle_data[i + 1].strip()
        line2 = tle_data[i + 2].strip()

        satellite = EarthSatellite(
            line1, line2, name, ts
        )  # Converting tle Data in SGP4 Satellite Object

        syst = (satellite - point_pos).at(
            t
        )  # Calculate the difference between the satellite's position and the point position to get the topocentric reference system in a given time.
        alt, _, distance = syst.altaz()

        if alt.degrees > buffer_Phi:
            s = Satellite(satellite, distance.km, (name, line1, line2), False)
            if alt.degrees > Phi_max:
                sats_dome.append(s)
            else:
                sats_buffer.append(s)

    # Ordino i satelliti in base alla posizione rispetto all'utente
    sat_sort_dome = sorted(sats_dome, key=lambda s: s.distance)
    sat_sort_buff = sorted(sats_buffer, key=lambda s: s.distance)

    counter, dome = 0, []
    if num_acc_point > len(sat_sort_dome):
        num_acc_point = (
            len(sat_sort_dome) // 2
        )  # Non ci sono abbastanza satelliti da soddisfare la richiesta di Access_point
        print(
            colorize(
                f"WARNING: Not enough satellites to satisfy the request. The number of access points has been set to {num_acc_point}.",
                "yellow",
            )
        )

    # Determino Access Points
    for sat in sat_sort_dome:

        # Decido se eseguire il filtro o meno
        if ap_mode == AccPointMode.BASE:
            can_take = counter < num_acc_point
        elif ap_mode == AccPointMode.OPTIMAL:
            can_take = counter < num_acc_point and filterSatellitesInView(
                sat.satellite, t
            )

        if can_take:
            sat.is_acc_point = True
            dome.append(sat)
            counter += 1
        else:
            dome.append(sat)

    return dome, sat_sort_buff


def getAllSatOnLine(
    t: Time,
    tle_data,
    pos_A: GeographicPosition,
    pos_B: GeographicPosition,
    corridor_deg: float = config["corridor_angle"],
    Phi_max: int = config["Phi_max"],
    Buffer_zone: int = config["Phi_buffer"],
    ap_mode_a: AccPointMode = AccPointMode.BASE,
    num_acc_point_A: int = config["access_point"]
) -> Topology:
    """
    This function return all satellites between a given point A,
    and a given point B, in a specific time t.
    It use the TLE data to find all satellites in the line of sight between A and B,
    including the dome/buffer zone exactly above A and B.
    """
    corridor_rad = np.radians(corridor_deg)  
    BZED = Phi_max - Buffer_zone 
    BZED_rad = np.radians(BZED) # Convertito una volta sola fuori dal ciclo
    
    sat_on_line = [] 
    
    # 1. Calcolo vettori osservatori (Geocentrici)
    vA_km = pos_A.at(t).position.km
    vB_km = pos_B.at(t).position.km

    # Normalizzati (Vettori Zenit)
    vA = vA_km / np.linalg.norm(vA_km)
    vB = vB_km / np.linalg.norm(vB_km)

    # Gestione caso A = B
    if np.linalg.norm(vA - vB) < 1e-7:
        print(colorize("[DEBUG] A e B sono uguali. Utilizzare la funzione su punto singolo.", "yellow"))
        # Inserisci qui la logica di fallback se serve
        pass

    else:
        # 2. Costruzione del piano del corridoio
        cross_prod = np.cross(vA, vB) 
        norm = np.linalg.norm(cross_prod) 

        # Caso limite: A e B opposti
        if norm < 1e-7:
            v_ref = np.array([0, 0, 1]) 
            if np.linalg.norm(np.cross(vA, v_ref)) < 1e-7:
                v_ref = np.array([1, 0, 0])
            cross_prod = np.cross(vA, v_ref)
            norm = np.linalg.norm(cross_prod)

            if norm < 1e-7:
                sys.exit(colorize("[ERROR]: Non è possibile definire un piano per A e B !!! Verifica i dati di input. ", "red"))

        # Vettore normale del piano e "Muri"
        n = cross_prod / norm
        nA = np.cross(n, vA)
        nB = np.cross(vB, n)

        # 3. FASE ACQUISIZIONE
        for i in range(0, len(tle_data), 3):
            name = tle_data[i].strip()
            line1 = tle_data[i + 1].strip()
            line2 = tle_data[i + 2].strip()

            # Creazione satellite e calcolo posizione orbitale (LA PARTE PIÙ PESANTE, FATTA UNA VOLTA SOLA)
            satellite = EarthSatellite(line1, line2, name, ts)
            vP_km = satellite.at(t).position.km  
            vP = vP_km / np.linalg.norm(vP_km)  

            # --- FASE 1: Verifica Corridoio Centrale ---
            # Angolo rispetto al piano centrale
            theta = np.arcsin(abs(np.clip(np.dot(n, vP), -1.0, 1.0)))
            in_corridor_inf = theta < corridor_rad
            # Verifica che sia racchiuso tra A e B
            between_walls = np.dot(vP, nA) >= 0 and np.dot(vP, nB) >= 0

            # --- FASE 2: Verifica Cupole (Metodo Topocentrico Vettoriale) ---
            # Vettore da A al satellite
            top_A = vP_km - vA_km
            top_A_norm = top_A / np.linalg.norm(top_A)
            alt_A_rad = np.arcsin(np.clip(np.dot(top_A_norm, vA), -1.0, 1.0))

            # Vettore da B al satellite
            top_B = vP_km - vB_km
            top_B_norm = top_B / np.linalg.norm(top_B)
            alt_B_rad = np.arcsin(np.clip(np.dot(top_B_norm, vB), -1.0, 1.0))

            # --- UNIONE LOGICA ---
            # È valido se si trova nel corridoio OPPURE sopra l'orizzonte BZED di A OPPURE di B
            if (in_corridor_inf and between_walls) or (alt_A_rad > BZED_rad) or (alt_B_rad > BZED_rad):
                # ! : controlla 
                # TODO : CONTROLLA CHE QUESTA SEZIONE SIA GIUSTA E FAI IL COMMIT
                sat_on_line.append(Satellite(satellite, np.linalg.norm(vP_km - vA_km), (name, line1, line2), False))
        
        # Sorting Satellite by distance from A
        sat_on_line.sort(key=lambda sat: np.linalg.norm(sat.satellite.at(t).position.km - vA_km))  
        
        
        if num_acc_point_A > len(sat_on_line):
            num_acc_point_A = len(sat_on_line)//2 # Non ci sono abbastanza satelliti da usare come AP
            print(colorize(f"WARNING: Not enough satellites to satisfy the request. The number of access points has been set to {num_acc_point_A}.", "orange"))
        
        
        counter = 0 # Contatore degli access Point
        # Determino Access Points
        for sat in sat_on_line:
            
            if counter >= num_acc_point_A:
                # Ho tutti gli AP di A, posso uscire
                break
            
            # Check della modalità
            if ap_mode_a == AccPointMode.BASE:
                can_take = counter < num_acc_point_A
            elif ap_mode_a == AccPointMode.OPTIMAL:
                can_take = counter < num_acc_point_A and filterSatellitesInView(sat, t)
            
            if can_take:
                sat.is_acc_point = True
                counter += 1
                
        return sat_on_line


# ---------------------------------------------------------------------------- #
def getAllSatOnMe(
    t,
    tle_data,
    ap_mode: AccPointMode,
    Phi_max=config["Phi_max"],
    Num_Access_point=config["access_point"],
):
    """
    This function return all the satellite above a given point

    Args:
        t (Time): time of observation.
        tle_data (tle_data): TLE data of the satellites.
        point_pos (GeographicPosition): position of specific point.
        Phi_max (float): Elevation Angle of a given point to search satellite in that area.
        num_acc_point (int): Number of acc point we are searching in that area

    """
    buffer_Phi = Phi_max - config["Phi_buffer"]  # Angle of a Buffer Zone
    satellites_dome, satellites_buffer = [], []

    # Get all satellites within the observer's dome and buffer zone
    for i in range(0, len(tle_data), 3):
        name = tle_data[i].strip()
        line1 = tle_data[i + 1].strip()
        line2 = tle_data[i + 2].strip()

        satellite = EarthSatellite(
            line1, line2, name, ts
        )  # Converting tle Data in SGP4 Satellite Object
        syst = getSystemFromSat(satellite, t)  # reference system

        alt, az, distance = (
            syst.altaz()
        )  # alt : Altitude in degrees relative to the observer
        # az : Sat Azimuth Angle relative to the observer
        # distance: distance Sat - Observer
        if alt.degrees > buffer_Phi:
            if alt.degrees > Phi_max:
                # Satellite in the Dome
                satellites_dome.append(
                    Satellite(satellite, distance.km, (name, line1, line2), False)
                )
            else:
                # Satellite in the Buffer Zone
                satellites_buffer.append(
                    Satellite(satellite, distance.km, (name, line1, line2), False)
                )

    # Ordino i satelliti in base alla posizione rispetto all'utente
    sat_sort_dome = sorted(satellites_dome, key=lambda s: s.distance)
    sat_sort_buff = sorted(satellites_buffer, key=lambda s: s.distance)

    counter = 0
    if Num_Access_point > len(sat_sort_dome):
        Num_Access_point = (
            len(sat_sort_dome) // 2
        )  # Non ci sono abbastanza satelliti da soddisfare la richiesta di Access_point
        print(
            f"WARNING: Not enough satellites to satisfy the request. The number of access points has been set to {Num_Access_point}."
        )

    # Determino Access Points
    for sat in sat_sort_dome:
        if counter >= Num_Access_point:
            break
        # Decido se eseguire il filtro o meno
        if ap_mode == AccPointMode.BASE:
            can_take = counter < Num_Access_point
        elif ap_mode == AccPointMode.OPTIMAL:
            can_take = counter < Num_Access_point and filterSatellitesInView(
                sat.satellite, t
            )

        if can_take:
            sat.is_acc_point = True
            counter += 1


    return sat_sort_dome, sat_sort_buff


def compute_distances_from_target_satellite(sat, closerSatellite_Sorted, t):
    """
    Compute the distances from the target satellite to other satellites.

    :param sat: The target satellite.
    :param closerSatellite_Sorted: List of satellites sorted by proximity.
    :param t: Current time.

    :return: List of tuples containing satellites and their distances from the target satellite.
    """

    vector_Sat_Topology = []
    for i in range(0, len(closerSatellite_Sorted)):
        if are_satellites_equal(sat.satellite, closerSatellite_Sorted[i].satellite):
            pass
        else:
            proximity = get_orbit_proximity(
                sat.satellite, closerSatellite_Sorted[i].satellite, t
            )
            if proximity < config["Laser_Communication_Range"]:  # Check laser distance
                latency = getLatency(proximity)  # Calculate latency
                vector_Sat_Topology.append(
                    (closerSatellite_Sorted[i].satellite.name, proximity, latency)
                )

    sat_vector_Topology_sorted = sorted(
        vector_Sat_Topology, key=lambda x: x[1]
    )  # Ordino i vicini in base alla distanza dal satellite
    return sat_vector_Topology_sorted


def find_angle(t, satellite):
    """
    Find the elevation angle of a satellite relative to the observer at a given time.

    Args:
        t (skyfield.timelib.Time): The time at which to find the elevation angle.
        satellite (EarthSatellite): The satellite object.

    Returns:
        float: The elevation angle in degrees.
    """
    sys = getSystemFromSat(satellite, t)
    alt, az, distance = sys.altaz()
    return alt.degrees


def create_satellite_Identity_card(
    satellite, life, neighbors_info, t, access_points=False, line_setup=False
):
    """
    Create a dictionary representing a satellite and its neighbors.

    Args:
        satellite (tuple): A tuple containing the satellite object and its distance from the user.
        life (dict): A dictionary with information about the satellite's life.
        neighbors_info (list of tuples): A list of tuples with (neighbor_name, distance, latency).
        t (skyfield.timelib.Time): The current time.
        access_points (bool): Whether the satellite is an access point.

    Returns:
        dict: A dictionary representing the satellite and its neighbors.
    """
    elevation_angle = None
    if not line_setup:
        elevation_angle = find_angle(
            t, satellite.satellite
        )  # Mi trovo l'angolo di elevazione del satellite rispetto l'obverver

    satellite_data = {
        "satellite": satellite.satellite.name,
        "distance_from_user": satellite.distance,
        "is_access_point": access_points,
        "elev_angle": elevation_angle,
        "TLE-DATA": [
            {
                "name": satellite.tle[0],
                "line1": satellite.tle[1],
                "line2": satellite.tle[2],
            }
        ],
        "life": life,
        "neighbors": [
            {"name": neighbor_name, "distance": distance, "latency": latency}
            for neighbor_name, distance, latency in neighbors_info
        ],
    }
    return satellite_data


def classifySat_BufferZone(buffer_satellites, time=time_now):
    time_end = ts.utc(time.utc_datetime() + timedelta(minutes=40))
    selected_satellites = []

    min_elev_cone = (
        90 - config["Phi_max"]
    )  # Quantità di gradi al di sotto della soglia dove osservo.
    buffer_Phi = config["Phi_max"] - config["Phi_buffer"]  # Angle of a Buffer Zone

    for s in buffer_satellites:
        # Calcolo eventi di passaggio
        t, events = s[0].find_events(
            OBSERVER, time, time_end, altitude_degrees=buffer_Phi
        )
        max_elevation = 0  # ? Masimo punto di elevazione del satellite

        for ti, event in zip(t, events):
            if event == 1:  # Max Alt

                syst = getSystemFromSat(s[0], time=time)  # reference system
                alt, az, distance = syst.altaz()

                max_elevation = alt.degrees  # Elevazione in gradi
                break
        print(f"SAT: {s[0].name} - Elevation: {max_elevation} >= {min_elev_cone}")
        # Verifica se il satellite raggiunge almeno la soglia del cono di osservazione
        if max_elevation >= min_elev_cone:
            selected_satellites.append((s[0], s[1], max_elevation))

    return selected_satellites


# ---------------------------------------------------------------------------- #


def check_satellite_visibility(satellite, location, start, end):
    # Intervallo di tempo con step di 1 minuto
    times = ts.utc_range(start.utc_datetime(), end.utc_datetime(), timedelta(minutes=1))

    # Trova elevazioni per ogni punto temporale
    elevations = []
    for t in times:
        difference = satellite - location
        topocentric = difference.at(t)
        alt, _, _ = topocentric.altaz()
        elevations.append((t, alt.degrees))

    # Filtra le elevazioni tra 20° e 40° e verifica se superano i 40°
    between_20_40 = any(20 <= alt <= 40 for _, alt in elevations)
    exceeds_40 = any(alt > 40 for _, alt in elevations)

    return between_20_40 and exceeds_40
