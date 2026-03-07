from joystick_adapter import press, release, tap
import time

print("Resetting to Passive")

# Velocity → Passive (LT + B)
press("LT")
time.sleep(0.2)
tap("B")
time.sleep(0.2)
release("LT")

time.sleep(2)

print("Entering FixStand")

# Passive → FixStand (LT + UP)
press("LT")
time.sleep(0.2)
tap("UP")
time.sleep(0.2)
release("LT")

time.sleep(4)

print("Starting Velocity (walk)")

# FixStand → Velocity (RB + X)
press("RB")
time.sleep(0.2)
tap("X")
time.sleep(0.2)
release("RB")

print("Robot should now be walking")