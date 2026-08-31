#!/usr/bin/env python3
"""A policy server that replays a recorded episode. No model, no GPU.

Speaks the same websocket protocol as openpi's own policy server, so the same
client drives either one -- swap the address, change nothing else. That is the
point of it: you can exercise the whole inference path (observation assembly,
transport, action clamping, the arm actually moving) before a checkpoint
exists, and later bisect "is it the policy or is it the plumbing" by pointing
the client at this instead.

    ros2 run pika_nero_teleop replay_policy_server.py \\
        --episode /root/pika_ros/data/episode0 --port 8000

    # then, in another shell
    ros2 run pika_nero_teleop policy_client.py --host localhost --port 8000

It is OPEN LOOP. The observation is read and discarded; actions come from the
recording regardless of where the arm actually is. That makes it a test of the
plumbing, not of control -- if the arm has drifted from where the episode was
recorded, the replay will happily command the original trajectory anyway. Start
from READY, which is where every episode begins.

Protocol, matching openpi's WebsocketPolicyServer:
  on connect  -> server sends a msgpack metadata dict
  each round  -> client sends an observation dict
                 server replies {"actions": (horizon, 8), "server_timing": {...}}

Depends only on `openpi-client` and `websockets` -- deliberately NOT on openpi
itself, so this runs on the robot machine without jax or a GPU:

    pip install websockets
    pip install -e <openpi>/packages/openpi-client
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback

import numpy as np

try:
    from openpi_client import msgpack_numpy
except ImportError:
    sys.exit("openpi-client is not installed:\n"
             "  pip install -e <openpi>/packages/openpi-client")
import websockets.asyncio.server as _server
import websockets.frames

ARM_DIR = "arm/jointState/master"        # the IK command = the recorded action
JAW_DIR = "gripper/encoder/pikaSensor"   # the Sense trigger = the gripper action


def load_episode(path):
    """Recorded episode directory -> (N, 8) actions, matching the dataset layout.

    Reads the raw capture rather than the converted LeRobot dataset so this has
    no dependency on lerobot/torch. Columns are the same 8: seven arm joints in
    radians then jaw opening in metres. See training/DATASET_CARD.md.
    """
    arm = os.path.join(path, ARM_DIR)
    jaw = os.path.join(path, JAW_DIR)
    for d in (arm, jaw):
        if not os.path.isdir(d):
            raise SystemExit(f"not a recorded episode: {d} is missing")

    def stamped(d):
        return sorted((float(f.rsplit(".", 1)[0]), os.path.join(d, f))
                      for f in os.listdir(d) if f[0].isdigit())

    arm_s, jaw_s = stamped(arm), stamped(jaw)
    jaw_t = np.array([t for t, _ in jaw_s])

    actions = []
    for t, f in arm_s:
        q = json.load(open(f))["position"]
        if len(q) < 7:
            continue
        # Nearest jaw sample in time. The jaw stream runs ~125 Hz against the
        # arm's 30, so "nearest" is within ~4 ms -- the same subsampling
        # data_sync.py does when building the dataset.
        j = json.load(open(jaw_s[int(np.argmin(np.abs(jaw_t - t)))][1]))
        actions.append(list(q[:7]) + [float(j.get("distance", 0.0))])
    if not actions:
        raise SystemExit(f"no usable samples in {path}")
    return np.asarray(actions, dtype=np.float32)


class ReplayPolicy:
    """Hands back the next `horizon` recorded actions, ignoring the observation."""

    def __init__(self, actions, horizon):
        self.actions = actions
        self.horizon = horizon
        self.cursor = 0

    def infer(self, obs):
        if self.cursor >= len(self.actions):
            # Past the end: hold the final pose rather than stopping dead, so
            # the client sees a well-formed chunk and the arm does not jerk.
            chunk = np.repeat(self.actions[-1:], self.horizon, axis=0)
        else:
            chunk = self.actions[self.cursor:self.cursor + self.horizon]
            if len(chunk) < self.horizon:
                pad = np.repeat(chunk[-1:], self.horizon - len(chunk), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
        self.cursor += self.horizon
        return {"actions": chunk.astype(np.float32)}


async def serve(policy, host, port, metadata):
    packer = msgpack_numpy.Packer()

    async def handler(ws):
        logging.info("client connected from %s", ws.remote_address)
        await ws.send(packer.pack(metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                t0 = time.monotonic()
                out = policy.infer(obs)
                out["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
                await ws.send(packer.pack(out))
            except websockets.ConnectionClosed:
                logging.info("client disconnected")
                # Rewind so the next client replays from the start.
                policy.cursor = 0
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                               reason="internal error")
                raise

    async with _server.serve(handler, host, port, compression=None, max_size=None):
        logging.info("replay server on ws://%s:%d", host, port)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="a recorded episode directory")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--action-horizon", type=int, default=30,
                    help="actions per reply; match the policy config (default 30)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    actions = load_episode(args.episode)
    logging.info("loaded %d actions (%.1f s at 30 Hz) from %s",
                 len(actions), len(actions) / 30.0, args.episode)
    logging.info("first action: %s", np.round(actions[0], 4).tolist())

    policy = ReplayPolicy(actions, args.action_horizon)
    metadata = {"server": "replay", "episode": os.path.basename(args.episode.rstrip("/")),
                "frames": len(actions), "action_horizon": args.action_horizon,
                "action_dim": int(actions.shape[1])}
    try:
        asyncio.run(serve(policy, args.host, args.port, metadata))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
