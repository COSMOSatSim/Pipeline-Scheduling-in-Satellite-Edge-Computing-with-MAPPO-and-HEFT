import sys

from skyfield.api import load, wgs84
from Observer import Observer_Temp
from SaveCurrentSATOnFile import openTLEFromFile, saveTLEOnFile
from user_based_topology import getAllSatOnLine
from Vessel import Vessel
import json5
from utils import colorize
from Plotters import plot_orbital_corridor
from enums import AccPointMode
from topology import genConfigs_line
import numpy as np


def load_vessels_settings(name_vassel:str, file_path: str = "config/vessel_setting.json5") -> Vessel:
    """
    This function loads vessel settings from a JSON5 configuration file and returns a list of Vessel instances.
    Args:
        name_vessel (str): The name of the vessel to load from the configuration file.
        file_path (str): The path to the configuration file.
    Returns:
        Vessel: An instance of the Vessel class.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json5.load(file)
        info = data.get("vessels", {}).get(name_vassel)
        print(colorize(f"[DEBUG] Loaded vessel position for '{name_vassel}': {info}", "cyan"))
        vessel = Vessel(info.get("name"), wgs84.latlon(info.get("lat"), info.get("lon")))

        return vessel


def load_obs_settings(file_path: str = "config/config.json5") -> Observer_Temp:
    """
    This function loads observer settings from a JSON5 configuration file and returns an Observer instance.
    Args:
        file_path (str): The path to the configuration file.

    Returns:
        Observer: An instance of the Observer class.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json5.load(file)
        location = data["locations"]["Roma"]
        observer = Observer_Temp(
            loc_name="Roma",
            position=wgs84.latlon(location.get("lat"), location.get("lon")),
        )
        return observer

def manager():

    Vessel = load_vessels_settings("A")
    observer = load_obs_settings()
    #tle_data = saveTLEOnFile()
    #sys.exit()
    tle_data = openTLEFromFile()
    ts = load.timescale()
    t0 = ts.now()

    
    # sats = getAllSatOnLine(
    #     t=t0, 
    #     tle_data=tle_data, 
    #     pos_A=observer.position, 
    #     pos_B=Vessel.position,
    #     corridor_deg=10.0
    # )
    
    # plot_orbital_corridor(sats, observer.position, Vessel.position, t0, 10.0)

    # genConfigs_line(
    #     t0=t0,
    #     interval=2, 
    #     num_configs=10, 
    #     tle_data=tle_data,
    #     json_path="here.json5",
    #     mode=AccPointMode.BASE, 
    #     pos_A=observer.position,
    #     pos_B=Vessel.position
    # )
    
    
    

if __name__ == "__main__":
    manager()
