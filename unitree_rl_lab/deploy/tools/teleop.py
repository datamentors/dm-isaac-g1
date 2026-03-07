#!/usr/bin/env python3
"""
teleop.py - UDP joystick emulator for g1_ctrl sim2sim.

Sends plain-text commands to JoystickInjector (127.0.0.1:15001)
to trigger FSM state transitions without a physical gamepad.

Usage:
    python teleop.py                     # interactive menu
    python teleop.py --auto-mimic        # auto: Passive -> FixStand -> Velocity -> Mimic
    python teleop.py --fixstand          # just transition to FixStand
    python teleop.py --velocity          # transition to Velocity (via FixStand)
"""

import argparse
import socket
import time

ADDR = ("127.0.0.1", 15001)


def send(msg: str):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(msg.encode("utf-8"), ADDR)
    s.close()


def tap(btn: str, t: float = 0.12):
    send(f"{btn}=1")
    time.sleep(t)
    send(f"{btn}=0")


def hold(btn: str, on: bool):
    send(f"{btn}={'1' if on else '0'}")


def release_all():
    send("rb=0 lb=0 a=0 b=0 x=0 y=0 up=0 down=0 left=0 right=0 start=0 back=0 f1=0 f2=0")
    send("set lx 0 ly 0 rx 0 ry 0 lt 0 rt 0")


def to_passive():
    """Any state -> Passive (LT + B)"""
    print("-> Passive (LT + B)")
    release_all()
    time.sleep(0.2)
    hold("lt", True)
    time.sleep(0.15)
    tap("b", 0.12)
    time.sleep(0.15)
    hold("lt", False)
    time.sleep(0.35)


def to_fixstand():
    """Passive -> FixStand (LT + up)"""
    print("-> FixStand (LT + up)")
    release_all()
    time.sleep(0.2)
    hold("lt", True)
    time.sleep(0.15)
    tap("up", 0.12)
    time.sleep(0.15)
    hold("lt", False)
    time.sleep(0.35)


def to_velocity():
    """FixStand -> Velocity (RB + X)"""
    print("-> Velocity (RB + X)")
    hold("rb", True)
    time.sleep(0.15)
    tap("x", 0.12)
    time.sleep(0.15)
    hold("rb", False)
    time.sleep(0.35)


def to_mimic():
    """Velocity -> Mimic_08Clip01Track1 (RB + Y)"""
    print("-> Mimic (RB + Y)")
    hold("rb", True)
    time.sleep(0.15)
    tap("y", 0.12)
    time.sleep(0.15)
    hold("rb", False)
    time.sleep(0.35)


def auto_mimic(stand_wait: float = 5.0):
    """Full auto sequence: Passive -> FixStand -> (wait) -> Velocity -> Mimic"""
    print("=== Auto-Mimic Sequence ===")
    to_fixstand()
    print(f"  Waiting {stand_wait}s for stand-up...")
    time.sleep(stand_wait)
    to_velocity()
    time.sleep(0.5)
    to_mimic()
    print("=== Mimic should be running now ===")


def interactive():
    """Interactive menu"""
    print("\n=== g1_ctrl Teleop (UDP JoystickInjector) ===")
    print("Commands:")
    print("  1 - Passive (LT + B)")
    print("  2 - FixStand (LT + up)")
    print("  3 - Velocity (RB + X)")
    print("  4 - Mimic (RB + Y)")
    print("  a - Auto: Passive -> FixStand -> Velocity -> Mimic")
    print("  r - Release all buttons")
    print("  q - Quit")
    print()

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "1":
            to_passive()
        elif cmd == "2":
            to_fixstand()
        elif cmd == "3":
            to_velocity()
        elif cmd == "4":
            to_mimic()
        elif cmd == "a":
            auto_mimic()
        elif cmd == "r":
            release_all()
            print("Released all")
        elif cmd == "q":
            break
        else:
            print("Unknown command")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="g1_ctrl teleop via UDP JoystickInjector")
    parser.add_argument("--auto-mimic", action="store_true", help="Auto sequence to start mimic")
    parser.add_argument("--fixstand", action="store_true", help="Transition to FixStand")
    parser.add_argument("--velocity", action="store_true", help="Transition to Velocity (via FixStand)")
    parser.add_argument("--stand-wait", type=float, default=5.0, help="Seconds to wait for stand-up")
    args = parser.parse_args()

    if args.auto_mimic:
        auto_mimic(args.stand_wait)
    elif args.fixstand:
        to_fixstand()
    elif args.velocity:
        to_fixstand()
        time.sleep(args.stand_wait)
        to_velocity()
    else:
        interactive()
