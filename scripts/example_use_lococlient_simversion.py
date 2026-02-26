#!/usr/bin/env python3
"""Unitree G1 Locomotion Client — Simulation REPL Interface.

Interactive command-line tool for controlling the Unitree G1 robot's locomotion
state machine via the Unitree SDK2 LocoClient. Designed for use with the G1
simulator (Isaac Sim / Mujoco / Gazebo) but also works on the real robot when
the DDS network interface is pointed at the hardware.

Locomotion Modes:
-----------------
The G1 locomotion controller exposes a finite-state machine (FSM) with these
primary modes:

    ID  Name             FSM ID   Description
    --  ----             ------   -----------
    0   ZeroTorque       —        Release all joint actuators (motors off).
    1   Damp             —        Passive damping on all joints (soft stop).
    2   LockedStanding   4        Stand in a fixed upright posture (no balance).
    3   RunningMode      801      Full bipedal locomotion with balance control.

IDs 0 and 1 use dedicated SDK methods; IDs 2 and 3 map to SetFsmId() calls
with the FSM identifiers shown above.

Response Sniffer:
-----------------
A background DDS subscriber (``ResponseSniffer``) automatically attaches to
the locomotion service's response topic and prints every reply that the robot
controller publishes. This is invaluable for debugging command delivery,
latency, and error codes without a separate terminal.

DDS Configuration:
------------------
The Unitree SDK2 uses DDS (Data Distribution Service) for robot communication.
The script accepts optional CLI arguments to configure:

    - **Network interface** (e.g. ``lo`` for loopback / simulation,
      ``eth0`` for hardware).
    - **DDS domain** (integer, default ``1``). Domain ``0`` is commonly used
      in simulation environments; domain ``1`` is the hardware default.
    - **Service name** (default ``"sport"``). The locomotion RPC service
      registered in the robot's DDS participant.

Usage:
    # Default DDS config (no arguments) — suitable for local simulation
    python3 scripts/example_use_lococlient_simversion.py

    # Specify network interface only (domain=1, service="sport")
    python3 scripts/example_use_lococlient_simversion.py lo

    # Specify interface and domain (e.g. domain=0 for Isaac Sim)
    python3 scripts/example_use_lococlient_simversion.py lo 0

    # Specify interface and service name (legacy positional form)
    python3 scripts/example_use_lococlient_simversion.py lo sport

    # Full: interface, domain, and service name
    python3 scripts/example_use_lococlient_simversion.py lo 0 sport

REPL Commands:
    0-3             Select a locomotion mode by numeric ID.
    ZeroTorque …    Select a locomotion mode by name (case-insensitive).
    fsm <id>        Send an arbitrary FSM ID (e.g. ``fsm 801``).
    list            Print the available locomotion modes.
    help            Show full usage information.
    exit / quit     Terminate the REPL.

Requirements:
    - unitree_sdk2py  (Unitree SDK 2 Python bindings)
    - A running G1 simulation or physical robot reachable over the
      configured DDS domain and network interface.
"""

import sys
import time
from dataclasses import dataclass

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

# Channel-name helper to resolve DDS topic names, plus the IDL response type
# used by the background response sniffer.
from unitree_sdk2py.core.channel_name import GetServerChannelName, ChannelType
from unitree_sdk2py.idl.unitree_api.msg.dds_ import Response_ as Response

# ---------------------------------------------------------------------------
# Locomotion mode definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestOption:
    """A named locomotion mode with its corresponding numeric selector ID.

    Attributes:
        name: Human-readable label shown in the REPL menu.
        id:   Integer selector used for user input and internal dispatch.
    """
    name: str
    id: int


# Available locomotion modes. The ``id`` field is the REPL selector; the
# actual FSM IDs sent to the robot differ for LockedStanding (4) and
# RunningMode (801) — see the dispatch logic in ``main()``.
OPTION_LIST = [
    TestOption(name="ZeroTorque",      id=0),
    TestOption(name="Damp",            id=1),
    TestOption(name="LockedStanding",  id=2),
    TestOption(name="RunningMode",     id=3),
]

# Fast lookup tables: name (lowercase) → TestOption, numeric id → TestOption
NAME2OPT = {opt.name.lower(): opt for opt in OPTION_LIST}
ID2OPT   = {opt.id: opt for opt in OPTION_LIST}

USAGE = """Usage:
  python3 example_use_lococlient_simversion.py [networkInterface] [domain_or_service] [service_name]

Examples:
  # Default DDS config (no arguments)
  python3 example_use_lococlient_simversion.py

  # Specify only network interface, keep default domain=1 and service="sport"
  python3 example_use_lococlient_simversion.py lo

  # Specify network interface and domain (e.g. domain=0)
  python3 example_use_lococlient_simversion.py lo 0

  # Old style: specify network interface and service (domain stays 1)
  python3 example_use_lococlient_simversion.py lo sport

  # Full: interface, domain, and service name
  python3 example_use_lococlient_simversion.py lo 0 sport

Inside the REPL:
  - Enter an ID (0/1/2/3) or a name (ZeroTorque, Damp, LockedStanding, RunningMode)
  - 'fsm <id>' : set an arbitrary FSM ID directly (e.g., 'fsm 801')
  - 'list'     : show available options
  - 'help'     : show this help
  - 'exit'/'quit'
"""


def print_options():
    """Display the numbered list of locomotion modes to stdout."""
    print("\nAvailable options:")
    for opt in OPTION_LIST:
        print(f"  {opt.id}: {opt.name}")
    print()


def resolve_option(user_input: str):
    """Resolve user input to a ``TestOption``, or ``None`` if unrecognised.

    Accepts either a numeric ID string (e.g. ``"2"``) or a case-insensitive
    mode name (e.g. ``"runningmode"``).

    Args:
        user_input: Raw string from the REPL prompt.

    Returns:
        The matching ``TestOption``, or ``None``.
    """
    s = user_input.strip().lower()
    if s.isdigit():
        return ID2OPT.get(int(s))
    return NAME2OPT.get(s)


# ---------------------------------------------------------------------------
# DDS Response Sniffer
# ---------------------------------------------------------------------------

class ResponseSniffer:
    """Background DDS subscriber that prints every locomotion service response.

    Subscribes to the *server-side SEND channel* for the given service name,
    which carries ``Response_`` messages published by the robot's locomotion
    controller after each command is processed.

    This is purely a diagnostic aid — it does not affect command dispatch.

    Args:
        service_name: The DDS locomotion service to monitor (default ``"sport"``).
    """

    def __init__(self, service_name: str = "sport"):
        self.service_name = service_name
        self.sub = None

    def start(self):
        """Create the DDS subscriber and begin receiving responses."""
        # The server's SEND channel is where responses are published.
        rsp_topic = GetServerChannelName(self.service_name, ChannelType.SEND)
        self.sub = ChannelSubscriber(rsp_topic, Response)
        self.sub.Init(self._cb, 64)  # queue depth of 64 messages
        print(f"[Sniffer] Subscribed to responses: {rsp_topic}")

    def _cb(self, rsp: Response):
        """DDS callback — extract and print key fields from the response.

        Defensively accesses nested attributes because the exact IDL layout
        may vary across SDK versions.
        """
        try:
            h = rsp.header
            rid = int(getattr(getattr(h, "identity", object()), "id", -1))
            api = int(getattr(getattr(h, "identity", object()), "api_id", -1))
            code = int(getattr(getattr(h, "status", object()), "code", -1))
            msg  = getattr(getattr(h, "status", object()), "message", "")
        except Exception:
            rid, api, code, msg = -1, -1, -1, ""

        data = ""
        try:
            data = str(getattr(rsp, "data", ""))[:200]
        except Exception:
            pass

        print(f"[Sniffer][RX] id={rid} api_id={api} code={code}"
              f"{' msg='+msg if msg else ''}"
              f"{' data='+data if data else ''}")


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments, initialise DDS, and run the interactive REPL.

    Argument parsing supports three positional parameters (all optional):

        1. ``networkInterface`` — DDS network interface name (e.g. ``lo``,
           ``eth0``). When omitted, ``ChannelFactoryInitialize()`` uses the
           SDK default.
        2. ``domain_or_service`` — If numeric, treated as the DDS domain ID
           (e.g. ``0``). If non-numeric, treated as the service name for
           backwards compatibility with older invocations.
        3. ``service_name`` — Explicit service name when the second argument
           is a domain ID.
    """
    # Defaults matching the Unitree G1 hardware convention
    domain = 1
    service_name = "sport"

    if len(sys.argv) == 1:
        # No arguments: rely on SDK-internal defaults (typically loopback).
        print("[INFO] No network interface or domain provided.")
        print("[INFO] Calling ChannelFactoryInitialize() with default settings.")
        ChannelFactoryInitialize()
    else:
        net_if = sys.argv[1]  # e.g. 'lo', 'eth0'

        # Second argument can be either a numeric domain or a service name.
        if len(sys.argv) >= 3:
            arg2 = sys.argv[2]
            if arg2.isdigit():
                domain = int(arg2)
                if len(sys.argv) >= 4:
                    service_name = sys.argv[3]
            else:
                # Legacy form: treat arg2 as the service name, keep domain=1.
                service_name = arg2

        print(f"[INFO] Initializing DDS with domain={domain}, iface={net_if}")
        ChannelFactoryInitialize(domain, net_if)

    # Attach the response sniffer before issuing any commands so we can
    # observe the controller's replies from the very first request.
    sniffer = ResponseSniffer(service_name)
    sniffer.start()

    # Initialise the locomotion RPC client with a generous 10 s timeout
    # (simulation may be slower than real-time).
    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()

    print("Connected. Type 'list' to see available options, or 'help' for help.")
    print_options()

    try:
        while True:
            try:
                user_input = input("Enter command ID/name or 'fsm <id>': ").strip()
            except EOFError:
                print("\nEOF received. Exiting.")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            # --- Meta commands ---
            if cmd in ("exit", "quit"):
                print("Exiting.")
                break
            if cmd in ("help", "?"):
                print(USAGE)
                continue
            if cmd == "list":
                print_options()
                continue

            # --- Direct FSM ID override ---
            if cmd.startswith("fsm "):
                parts = cmd.split()
                if len(parts) != 2:
                    print("Usage: fsm <integer_id>")
                    continue
                try:
                    fsm_id = int(parts[1])
                except ValueError:
                    print("Usage: fsm <integer_id>")
                    continue
                print(f"-> SetFsmId({fsm_id}) ...")
                try:
                    code = client.SetFsmId(fsm_id)
                    print(f"   => return code: {code}")
                except Exception as e:
                    print(f"Error while setting FSM {fsm_id}: {e}")
                time.sleep(0.2)
                continue

            # --- Named / numbered mode selection ---
            opt = resolve_option(user_input)
            if not opt:
                print("Unrecognized option. Type 'list' to see valid choices, or use 'fsm <id>'.")
                continue

            print(f"-> Executing {opt.name} (id={opt.id}) ...")
            try:
                if opt.id == 0:
                    # ZeroTorque: disable all actuators immediately.
                    client.ZeroTorque()
                elif opt.id == 1:
                    # Damp: apply passive joint damping (safe shutdown posture).
                    client.Damp()
                elif opt.id == 2:
                    # LockedStanding: FSM 4 — rigid upright pose, no active balance.
                    code = client.SetFsmId(4)
                    print(f"   => SetFsmId(4) return code: {code}")
                elif opt.id == 3:
                    # RunningMode: FSM 801 — full bipedal locomotion with balance.
                    code = client.SetFsmId(801)
                    print(f"   => SetFsmId(801) return code: {code}")
                else:
                    print(f"Unknown ID {opt.id}. Nothing done.")
                    continue
            except Exception as e:
                print(f"Error while executing {opt.name}: {e}")

            # Small pause to allow the sniffer callback to print the response
            # before the next prompt appears.
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
