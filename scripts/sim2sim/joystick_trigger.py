"""
Joystick button injector for g1_ctrl FSM transitions.

Sends UDP packets to the JoystickInjector (127.0.0.1:15001) to trigger
FSM state transitions without a physical gamepad.

Usage:
    python joystick_trigger.py --policy military_march
    python joystick_trigger.py --policy dance_102
    python joystick_trigger.py --policy gangnam_style
    python joystick_trigger.py --policy 08clip01
    python joystick_trigger.py --policy cr7_youtube_run

Button Bitmask Reference (unitree_joystick_dsl.hpp):
    A     = 0x01    B     = 0x02    X     = 0x04    Y     = 0x08
    UP    = 0x10    DOWN  = 0x40    LEFT  = 0x80    RIGHT = 0x20
    LB    = 0x100   RB    = 0x200   LT    = 0x400   RT    = 0x800
    START = 0x1000  SELECT = 0x2000
    L3    = 0x4000  R3    = 0x8000

FSM Transitions (from config.yaml):
    Passive -> FixStand:        LT (hold 2s) + UP
    FixStand -> MilitaryMarch:  RB + X
    MilitaryMarch -> Dance_102: LT (hold 2s) + DOWN
    MilitaryMarch -> Gangnam:   LT (hold 2.5s) + LEFT
    MilitaryMarch -> 08Clip01:  RB + Y
    MilitaryMarch -> CR7:       RB + A
"""

import argparse
import socket
import struct
import time

INJECTOR_HOST = "127.0.0.1"
INJECTOR_PORT = 15001

# Button constants
A = 0x01
B = 0x02
X = 0x04
Y = 0x08
UP = 0x10
DOWN = 0x40
LEFT = 0x80
RB = 0x200
LT = 0x400


def send_button(buttons: int, duration: float = 0.1):
    """Send a button press via UDP to the JoystickInjector."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = struct.pack("<I", buttons)
    end_time = time.time() + duration
    while time.time() < end_time:
        sock.sendto(data, (INJECTOR_HOST, INJECTOR_PORT))
        time.sleep(0.02)  # 50Hz
    sock.close()


def transition_passive_to_fixstand():
    """Passive -> FixStand: hold LT for 2s, then press UP."""
    print("  Passive -> FixStand: holding LT (2s)...")
    send_button(LT, duration=2.0)
    time.sleep(0.5)
    print("  Pressing UP...")
    send_button(UP, duration=0.3)
    time.sleep(2.0)  # wait for stand-up


def transition_fixstand_to_military():
    """FixStand -> MilitaryMarch: RB + X."""
    print("  FixStand -> MilitaryMarch: RB + X...")
    send_button(RB | X, duration=0.3)
    time.sleep(2.0)


def run_military_march():
    """Full chain: Passive -> FixStand -> MilitaryMarch."""
    print("[1/2] Passive -> FixStand")
    transition_passive_to_fixstand()
    print("[2/2] FixStand -> MilitaryMarch")
    transition_fixstand_to_military()
    print("=== MilitaryMarch active ===")


def run_dance_102():
    """Full chain: -> MilitaryMarch -> Dance_102 (LT hold 2s + DOWN)."""
    run_military_march()
    time.sleep(1.0)
    print("[3/3] MilitaryMarch -> Dance_102: holding LT (2s)...")
    send_button(LT, duration=2.0)
    time.sleep(0.5)
    print("  Pressing DOWN...")
    send_button(DOWN, duration=0.3)
    print("=== Dance_102 active ===")


def run_gangnam_style():
    """Full chain: -> MilitaryMarch -> Gangnam_Style (LT hold 2.5s + LEFT)."""
    run_military_march()
    time.sleep(1.0)
    print("[3/3] MilitaryMarch -> Gangnam_Style: holding LT (2.5s)...")
    send_button(LT, duration=2.5)
    time.sleep(0.5)
    print("  Pressing LEFT...")
    send_button(LEFT, duration=0.3)
    print("=== Gangnam_Style active ===")


def run_08clip01():
    """Full chain: -> MilitaryMarch -> 08Clip01Track1 (RB + Y)."""
    run_military_march()
    time.sleep(1.0)
    print("[3/3] MilitaryMarch -> 08Clip01Track1: RB + Y...")
    send_button(RB | Y, duration=0.3)
    print("=== 08Clip01Track1 active ===")


def run_cr7_youtube_run():
    """Full chain: -> MilitaryMarch -> CR7YoutubeRun (RB + A)."""
    run_military_march()
    time.sleep(1.0)
    print("[3/3] MilitaryMarch -> CR7YoutubeRun: RB + A...")
    send_button(RB | A, duration=0.3)
    print("=== CR7YoutubeRun active ===")


POLICIES = {
    "military_march": run_military_march,
    "dance_102": run_dance_102,
    "gangnam_style": run_gangnam_style,
    "08clip01": run_08clip01,
    "cr7_youtube_run": run_cr7_youtube_run,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger g1_ctrl FSM transitions")
    parser.add_argument("--policy", required=True, choices=POLICIES.keys(),
                        help="Policy to activate")
    args = parser.parse_args()

    print(f"=== Triggering {args.policy} ===")
    POLICIES[args.policy]()
    print("Done. Robot should be executing the policy.")
    print("In MuJoCo: press 8 to lower robot, then 9 to release elastic band.")
