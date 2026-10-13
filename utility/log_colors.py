"""Colori ANSI per l'output su terminale (monitor, diagnostica, print)."""
import random

#######################################################################################################
# CODICI ANSI
#######################################################################################################
COLOR_RESET = "\033[0m"
COLOR_CYAN  = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_RED   = "\033[91m"

#######################################################################################################
# PALETTE PER-DEVICE
#######################################################################################################
DEVICE_COLOR_PALETTE = [
    "\033[92m",  # green
    "\033[93m",  # yellow
    "\033[94m",  # blue
    "\033[95m",  # magenta
    "\033[91m",  # red
    "\033[97m",  # white
    "\033[90m",  # bright black / gray
    "\033[36m",  # cyan-dark
]

# Equivalenti hex della palette ANSI, nello stesso ordine: servono ai plot (Gantt) per colorare un
# device con lo STESSO colore che ha in console.
DEVICE_HEX_PALETTE = [
    "#2ca02c",  # green
    "#c8a200",  # yellow (scurito per contrasto su bianco)
    "#1f77b4",  # blue
    "#b10dc9",  # magenta
    "#e11d1d",  # red
    "#555555",  # white -> grigio scuro
    "#808080",  # gray
    "#17a2b8",  # cyan-dark
]
_ANSI_TO_HEX = dict(zip(DEVICE_COLOR_PALETTE, DEVICE_HEX_PALETTE))

_rng = random.Random()
_available_colors = DEVICE_COLOR_PALETTE.copy()
_device_colors = {}

#######################################################################################################
# API
#######################################################################################################

def color_for_device(device_id):
    """Colore stabile per `device_id`: ne pesca uno a caso dalla palette e lo memoizza
    (ricarica la palette quando i colori distinti finiscono)."""
    if device_id not in _device_colors:
        if not _available_colors:
            _available_colors.extend(DEVICE_COLOR_PALETTE)
        color = _rng.choice(_available_colors)
        _available_colors.remove(color)
        _device_colors[device_id] = color
    return _device_colors[device_id]


def colorize(text, color):
    """Avvolge `text` nel codice `color` + reset."""
    if not color:
        return text
    return f"{color}{text}{COLOR_RESET}"


def hex_for_device(device_id):
    """Colore hex del device, corrispondente al colore ANSI che ha in console."""
    return _ANSI_TO_HEX.get(color_for_device(device_id), "#000000")
