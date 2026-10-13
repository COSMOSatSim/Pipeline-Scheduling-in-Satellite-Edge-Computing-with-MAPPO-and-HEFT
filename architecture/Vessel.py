
from skyfield.toposlib import GeographicPosition

class Vessel:
    def __init__(self, name:str, position: GeographicPosition):
        self.name = name   
        self.position = position


    def __str__(self):
        return f"Vessel(name={self.name}, position={self.position})"