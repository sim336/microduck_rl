#!/usr/bin/env python3
"""BAM pendulum-bench recorder for the Feetech HD-1910-C001.

Produces raw trajectory logs in the JSON schema consumed by Rhoban BAM
(https://github.com/Rhoban/bam): process with `python -m bam.process`, fit
with `python -m bam.fit`.

Methodology adapted from the published Waveshare ST3025 bench
(https://github.com/i1Cps/duck_mini_pro_headless/tree/main/bam_bench), which
used the same STS3215 voltage-control platform and reached M6 holdout
MAE ~20 mrad. Key practices carried over:

  - zero reference is the servo's HOME'd center (2047), NOT gravity rest:
    gearbox friction+armature hold the bare arm wherever it is left;
  - goal-sync before torque-on (a stale goal register would snap the arm);
  - pure-P (I=0, D=0) watchdog against self-oscillation on a light arm;
  - temperature gate between trajectories (thermal drift corrupts fits);
  - loop-rate self-test >= 150 Hz (fits assume ~200 Hz telemetry);
  - BENCH telemetry cross-checked against independent reads before moving;
  - multiple kp values (4/8/16/32) — control-law params need multi-kp data.

BENCH SETUP (one dedicated HD-1910 on the pendulum rig, NOT the robot):
  - servo mounted rigidly, shaft horizontal, arm + weights attached
  - arm swings freely ~+/-110 deg around straight-down
  - USB-TTL half-duplex adapter (Feetech), 1 Mbps, ID 1 (default)
  - regulated supply at the target voltage (6 / 7.4 / 8.4 V), logged per entry

NOTE: HD-1910 is a pre-sale part; the register map below follows the
STS3215/HL platform and MUST be verified against the real unit (use the
Feetech host tool) before trusting telemetry. Current feedback is read
only if the register exists on HD-1910.

Usage — one invocation per mounting condition (mass x length):

  python3 scripts/hd1910_bench_record.py \
      --port /dev/ttyUSB0 --id 1 \
      --mass 0.30 --arm-mass 0.05 --length 0.12 \
      --kps 4 8 16 32 --logdir ~/bam_logs/hd1910 \
      --vin 7.4
"""
import argparse
import json
import math
import os
import time
from datetime import datetime

import numpy as np

from bam.trajectory import trajectories

RAD_PER_STEP = (2.0 * math.pi) / 4096.0  # HD-1910 12-bit encoder
ZERO_STEPS = 2047                          # servo center after HOME
MAX_ANGLE = 1.92                           # rad (~110 deg), BAM full amplitude
WATCHDOG_MARGIN = 0.26                     # rad past --max-angle
TEMP_PAUSE_C = 55
TEMP_RESUME_C = 48
MIN_LOOP_HZ = 150
MOTOR_NAME = "hd1910"

# --- Feetech TTL register addresses (STS3215 platform — VERIFY ON HD-1910) ---
REG_P_ID = 21
REG_LOCK = 55
REG_TORQUE_LIMIT = 48
REG_PRESENT_POS = 56
REG_PRESENT_SPEED = 58
REG_PRESENT_LOAD = 60
REG_PRESENT_VOLTAGE = 62
REG_PRESENT_TEMP = 63
REG_PRESENT_CURRENT = 69   # HL-series 2-byte current; verify on HD-1910


class FeetechBus:
    """Minimal TTL half-duplex reader/writer for one servo (pypot-backed).

    Swap this class for rustypot or a raw serial driver if pypot's
    FeetechSTS3215IO does not speak the HD-1910 register map.
    """

    def __init__(self, port: str, sid: int):
        try:
            from pypot.feetech import FeetechSTS3215IO
        except ImportError:
            raise SystemExit(
                "pypot not installed — `pip install pypot` or replace "
                "FeetechBus with your own TTL driver."
            )
        self.sid = sid
        self.io = FeetechSTS3215IO(port)

    def read_register(self, sid, addr, n):
        # pypot exposes per-register getters; fall back to raw if absent.
        return self.io.read_register(sid, addr, n)

    def write_register(self, sid, addr, data):
        self.io.write_register(sid, addr, data)

    def set_gains(self, sid, kp, ki=0, kd=0):
        self.io.set_P_coefficient({sid: kp})
        self.io.set_D_coefficient({sid: kd})

    def set_goal(self, sid, steps, speed=0, acc=0):
        self.io.set_goal_position({sid: steps})

    def enable(self, sid, on: bool):
        if on:
            self.io.enable_torque([sid])
        else:
            self.io.disable_torque([sid])

    def telemetry(self, sid):
        """Read (pos, vel, load, volt, temp, current) in raw units; None on error."""
        try:
            pos = self.io.get_present_position([sid])[0]
            vel = self.io.get_present_speed([sid])[0]
            load = self.io.get_present_load([sid])[0]
            volt = self.io.get_present_voltage([sid])[0]
            temp = self.io.get_present_temperature([sid])[0]
            cur = None
            try:
                cur = self.io.get_present_current([sid])[0]
            except Exception:
                pass  # HD-1910 current register may differ — best effort
            return pos, vel, load, volt, temp, cur
        except Exception:
            return None


def steps_of(angle_rad, zero_steps=ZERO_STEPS):
    s = int(round(zero_steps + angle_rad / RAD_PER_STEP))
    return max(30, min(4065, s))


def loop_rate_test(bus, sid, n=100):
    t0 = time.monotonic()
    good = 0
    for _ in range(n):
        if bus.telemetry(sid) is not None:
            good += 1
    dt = time.monotonic() - t0
    hz = n / dt if dt > 0 else 0
    print(f"loop-rate self-test: {hz:.0f} Hz ({good}/{n} good reads)")
    return hz, good


def temp_gate(bus, sid):
    t = bus.telemetry(sid)
    if t is None or t[4] < TEMP_PAUSE_C:
        return
    print(f"  [temp gate] servo at {t[4]}C — pausing until <{TEMP_RESUME_C}C ...")
    while t is not None and t[4] > TEMP_RESUME_C:
        time.sleep(10)
        t = bus.telemetry(sid)


def record_trajectory(bus, sid, traj_name, meta, max_angle=MAX_ANGLE):
    traj = trajectories[traj_name]
    duration = traj.duration
    entries = []
    torque_enable = None
    aborted = False

    # Goal-sync before torque-on (stale goal would snap the arm).
    here = bus.telemetry(sid)
    if here is not None:
        bus.set_goal(sid, here[0])
    bus.enable(sid, True)
    a0, en0 = traj(0.0)
    bus.set_goal(sid, steps_of(a0))
    time.sleep(1.5)
    if not en0:
        bus.enable(sid, False)
    torque_enable = en0

    start = time.monotonic()
    while True:
        t = time.monotonic() - start
        if t >= duration:
            break
        goal, enable = traj(t)
        goal = max(-max_angle, min(max_angle, goal))
        if enable != torque_enable:
            bus.enable(sid, enable)
            torque_enable = enable
        if enable:
            bus.set_goal(sid, steps_of(goal))

        t0 = time.monotonic() - start
        tel = bus.telemetry(sid)
        t1 = time.monotonic() - start
        if tel is None:
            continue
        pos, vel, load, volt, temp, cur = tel
        entries.append({
            "timestamp": (t0 + t1) / 2.0,
            "position": (pos - ZERO_STEPS) * RAD_PER_STEP,
            "speed": vel * RAD_PER_STEP,
            "load": float(load),
            "current": float(cur) if cur is not None else 0.0,
            "input_volts": volt / 10.0,
            "temp": float(temp),
            "goal_position": float(goal),
            "torque_enable": bool(torque_enable),
        })
        # Pure-P watchdog: D=0 can self-oscillate and pump amplitude.
        if abs(entries[-1]["position"]) > max_angle + WATCHDOG_MARGIN:
            bus.enable(sid, False)
            torque_enable = False
            aborted = True
            print(f"  !! WATCHDOG: position {entries[-1]['position']:+.2f} rad "
                  f"exceeded {max_angle + WATCHDOG_MARGIN:.2f} — torque cut")
            break

    # Bleed energy passively before re-taking control.
    bus.enable(sid, False)
    torque_enable = False
    deadline = time.monotonic() + 8.0
    still = 0
    while time.monotonic() < deadline and still < 5:
        tel = bus.telemetry(sid)
        if tel is not None and abs(tel[1]) * RAD_PER_STEP < 0.05:
            still += 1
        else:
            still = 0
        time.sleep(0.05)

    if aborted:
        return {**meta, "trajectory": traj_name, "entries": entries,
                "aborted": True}

    # Gentle return to hanging zero.
    here = bus.telemetry(sid)
    if here is not None:
        bus.set_goal(sid, here[0])
    bus.enable(sid, True)
    cur = (here[0] - ZERO_STEPS) * RAD_PER_STEP if here is not None else 0.0
    while abs(cur) > 0.02:
        cur -= math.copysign(min(0.01, abs(cur)), cur)
        bus.set_goal(sid, steps_of(cur))
        time.sleep(0.01)
    bus.set_goal(sid, ZERO_STEPS)

    return {**meta, "trajectory": traj_name, "entries": entries, "aborted": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="TTL serial port")
    ap.add_argument("--id", type=int, default=1, help="bench servo id")
    ap.add_argument("--mass", type=float, required=True, help="attached mass incl. fasteners [kg]")
    ap.add_argument("--arm-mass", type=float, default=0.05, help="arm assembly mass [kg]")
    ap.add_argument("--length", type=float, required=True, help="shaft axis -> mass CoM [m]")
    ap.add_argument("--vin", type=float, default=7.4, help="supply voltage [V]")
    ap.add_argument("--kps", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--trajectories", nargs="+", default=list(trajectories))
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="logs per (kp, trajectory)")
    ap.add_argument("--max-angle", type=float, default=MAX_ANGLE)
    ap.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    args = ap.parse_args()

    for name in args.trajectories:
        if name not in trajectories:
            raise SystemExit(f"unknown trajectory {name!r} (have: {list(trajectories)})")
    os.makedirs(os.path.expanduser(args.logdir), exist_ok=True)

    bus = FeetechBus(args.port, args.id)
    sid = args.id
    try:
        tel = bus.telemetry(sid)
        if tel is None:
            raise SystemExit(f"servo {sid} not responding on {args.port} — "
                             "check wiring, 1 Mbps baud, and that the unit is "
                             "not already claimed by another process.")
        print(f"telemetry sane: pos {tel[0]}, {tel[3] / 10:.1f} V, {tel[4]} C")

        hz, good = loop_rate_test(bus, sid)
        if hz < MIN_LOOP_HZ:
            print(f"WARNING: loop rate {hz:.0f} Hz < {MIN_LOOP_HZ} Hz — "
                  f"fits will use dt=0.01 instead of 0.005; proceeding.")

        vin = args.vin
        print(f"bus voltage: {vin:.1f} V (regulated supply)")

        span = int(args.max_angle / RAD_PER_STEP) + 150
        if not (span <= ZERO_STEPS <= 4095 - span):
            raise SystemExit(f"--max-angle {args.max_angle} exceeds the range "
                             "around servo center — use <= 1.92.")

        n_planned = len(args.kps) * len(args.trajectories) * args.repeat
        print(f"\nplan: kps {args.kps} x {args.trajectories} x{args.repeat} = "
              f"{n_planned} logs (mass {args.mass} kg, arm {args.arm_mass} kg, "
              f"length {args.length} m, clamp +-{args.max_angle:.2f} rad)")
        print("first movement: slow move to servo center for the plumb check, "
              "then trajectory 1 (lift_and_drop cuts torque at t=2s by design).")
        if not args.yes:
            ans = input("Arm free, swing clear, weights secured, arm pointing "
                        "STRAIGHT DOWN? [y/N] ").strip().lower()
            if ans != "y":
                raise SystemExit("aborted before any movement.")

        here = bus.telemetry(sid)
        if here is None:
            raise SystemExit("could not read position")
        bus.set_goal(sid, here[0])           # goal-sync, no snap
        bus.enable(sid, True)
        bus.set_goal(sid, ZERO_STEPS)
        time.sleep(2.5)
        if not args.yes:
            ans = input("Servo is at center — arm pointing STRAIGHT DOWN (plumb)? "
                        "[y/N] ").strip().lower()
            if ans != "y":
                bus.enable(sid, False)
                raise SystemExit("zero miscalibrated — re-HOME with the arm "
                                 "plumb, then rerun.")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_logs = 0
        for kp in args.kps:
            bus.set_gains(sid, kp, 0, 0)
            print(f"--- kp={kp} ---")
            for traj_name in args.trajectories:
                for rep in range(args.repeat):
                    temp_gate(bus, sid)
                    meta = {
                        "mass": args.mass,
                        "arm_mass": args.arm_mass,
                        "length": args.length,
                        "kp": kp,
                        "vin": vin,
                        "motor": MOTOR_NAME,
                    }
                    print(f"  recording {traj_name} (rep {rep + 1}/{args.repeat}) ...")
                    log = record_trajectory(bus, sid, traj_name, meta,
                                            max_angle=args.max_angle)
                    was_aborted = log.pop("aborted", False)
                    fn = (("ABORTED_" if was_aborted else "")
                          + f"{MOTOR_NAME}_m{args.mass}_L{args.length}_kp{kp}_"
                            f"{traj_name}_r{rep + 1}_{stamp}.json")
                    path = os.path.join(os.path.expanduser(args.logdir), fn)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(log, f)
                    if was_aborted:
                        raise SystemExit(f"watchdog abort — torque off, partial "
                                         f"log at {path}.\nInspect the rig "
                                         "before rerunning.")
                    n_logs += 1
                    print(f"  wrote {path} ({len(log['entries'])} entries)")

        print(f"\ndone: {n_logs} logs to {args.logdir}.")
    finally:
        try:
            bus.enable(sid, False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
