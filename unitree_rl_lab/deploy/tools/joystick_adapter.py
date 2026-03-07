import socket
import time

ADDR = ("127.0.0.1", 15001)

# Xbox-style → Unitree mapping
BUTTON_MAP = {
    "RB": "rt",
    "LB": "lb",
    "LT": "lt",
    "RT": "rt",

    "A": "a",
    "B": "b",
    "X": "b",   # FSM X corresponds to script "b"
    "Y": "y",

    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
}

def _send(msg: str):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(msg.encode(), ADDR)
    s.close()

def press(btn):
    _send(f"{btn.lower()}=1")

def release(btn):
    _send(f"{btn.lower()}=0")

def tap(btn, duration=0.12):
    press(btn)
    time.sleep(duration)
    release(btn)

def hold(btn, duration):
    press(btn)
    time.sleep(duration)
    release(btn)

def combo(hold_btn, tap_btn, delay=0.15):
    press(hold_btn)
    time.sleep(delay)
    tap(tap_btn)
    time.sleep(delay)
    release(hold_btn)