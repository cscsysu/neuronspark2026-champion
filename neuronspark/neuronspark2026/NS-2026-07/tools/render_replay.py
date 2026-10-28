#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve()
PUBLIC_ROOT = HERE.parents[1] if (HERE.parents[1] / "simulator" / "dexsim_core.py").exists() else None
if PUBLIC_ROOT is not None:
    sys.path.insert(0, str(PUBLIC_ROOT / "simulator"))
else:
    sys.path.insert(0, str(HERE.parent))

from dexsim_core import AgentSubprocessProxy, DexSimEnv, read_jsonl, sanitize_task_info, score_rollout  # noqa: E402

WIDTH = 640
HEIGHT = 420
FPS = 8


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _world_to_px(x: float, y: float) -> tuple[int, int]:
    scale = min(WIDTH * 0.62, HEIGHT * 0.82) / 2.2
    cx = int(WIDTH * 0.39)
    cy = int(HEIGHT * 0.52)
    return int(cx + x * scale), int(cy - y * scale)


def _blank() -> bytearray:
    buf = bytearray([246, 244, 238] * WIDTH * HEIGHT)
    for gx in [i / 5 for i in range(-5, 6)]:
        x0, y0 = _world_to_px(gx, -1.0)
        x1, y1 = _world_to_px(gx, 1.0)
        _line(buf, x0, y0, x1, y1, (226, 223, 214))
    for gy in [i / 5 for i in range(-5, 6)]:
        x0, y0 = _world_to_px(-1.0, gy)
        x1, y1 = _world_to_px(1.0, gy)
        _line(buf, x0, y0, x1, y1, (226, 223, 214))
    return buf


def _clip_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel))) for channel in color)


def _set(buf: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        i = (y * WIDTH + x) * 3
        buf[i:i + 3] = bytes(_clip_color(color))


def _rect(buf: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    color = _clip_color(color)
    xa, xb = sorted((max(0, x0), min(WIDTH - 1, x1)))
    ya, yb = sorted((max(0, y0), min(HEIGHT - 1, y1)))
    for y in range(ya, yb + 1):
        row = (y * WIDTH + xa) * 3
        buf[row:row + (xb - xa + 1) * 3] = bytes(color) * (xb - xa + 1)


def _circle(buf: bytearray, cx: int, cy: int, r: int, color: tuple[int, int, int], fill: bool = True) -> None:
    rr = r * r
    inner = max(0, r - 2) ** 2
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if d <= rr and (fill or d >= inner):
                _set(buf, x, y, color)


def _line(buf: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set(buf, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _arrow(buf: bytearray, x0: int, y0: int, dx: float, dy: float, color: tuple[int, int, int]) -> None:
    norm = math.sqrt(dx * dx + dy * dy)
    if norm < 1e-8:
        return
    ux, uy = dx / norm, dy / norm
    x1 = int(x0 + ux * 48)
    y1 = int(y0 - uy * 48)
    _line(buf, x0, y0, x1, y1, color)
    _circle(buf, x1, y1, 4, color, True)


def _bar(buf: bytearray, x: int, y: int, width: int, value: float, color: tuple[int, int, int]) -> None:
    value = _clamp(value, 0.0, 1.0)
    _rect(buf, x, y, x + width, y + 9, (218, 214, 204))
    _rect(buf, x, y, x + int(width * value), y + 9, color)


def _primitive_color(primitive: str) -> tuple[int, int, int]:
    return {
        "brace": (31, 108, 99),
        "stabilize": (31, 108, 99),
        "push": (213, 93, 43),
        "drag": (176, 74, 119),
        "pivot": (74, 92, 180),
        "roll": (58, 137, 190),
        "lift_edge": (181, 116, 37),
        "tap": (81, 139, 58),
        "wait": (116, 116, 116),
        "finish": (24, 124, 66),
    }.get(primitive, (90, 90, 90))


def _frame_png_data(frame: dict[str, Any], task: dict[str, Any]) -> bytes:
    buf = _blank()
    target = task["target_pose"]
    tx, ty = _world_to_px(float(target["x"]), float(target["y"]))
    _circle(buf, tx, ty, 24, (38, 125, 92), False)
    _line(buf, tx, ty, int(tx + 24 * math.cos(float(target["theta"]))), int(ty - 24 * math.sin(float(target["theta"]))), (38, 125, 92))

    pose = frame["pose"]
    ox, oy = _world_to_px(float(pose["x"]), float(pose["y"]))
    contact = float(frame.get("contact", 0.0))
    damage = float(frame.get("damage", 0.0))
    slip = float(frame.get("slip_risk", 0.0))
    base = (
        int(80 + 120 * damage),
        int(116 - 44 * damage),
        int(168 - 88 * damage),
    )
    radius = int(17 + 15 * float(task.get("object", {}).get("size", 0.12)))
    _circle(buf, ox, oy, radius + int(10 * contact), (169, 197, 186), False)
    _circle(buf, ox, oy, radius, base, True)
    _line(buf, ox, oy, int(ox + 28 * math.cos(float(pose["theta"]))), int(oy - 28 * math.sin(float(pose["theta"]))), (25, 31, 38))

    action = frame.get("action") or {}
    primitive = str(action.get("primitive", "wait"))
    direction = action.get("direction") or [0.0, 0.0]
    _arrow(buf, ox, oy, float(direction[0]), float(direction[1]), _primitive_color(primitive))

    panel_x = 500
    _rect(buf, panel_x - 20, 0, WIDTH - 1, HEIGHT - 1, (238, 235, 226))
    _bar(buf, panel_x, 42, 105, contact, (51, 134, 106))
    _bar(buf, panel_x, 72, 105, slip, (210, 99, 47))
    _bar(buf, panel_x, 102, 105, damage, (174, 53, 70))
    _bar(buf, panel_x, 132, 105, min(1.0, float(action.get("force", 0.0))), (79, 107, 178))
    step_value = min(1.0, float(frame.get("step", 0)) / max(1.0, float(task.get("max_steps", 64))))
    _bar(buf, panel_x, 162, 105, step_value, (93, 93, 93))

    if frame.get("possible_drop"):
        _circle(buf, panel_x + 20, 210, 12, (176, 43, 52), True)
    if frame.get("resource_warning"):
        _circle(buf, panel_x + 58, 210, 12, (196, 131, 38), True)
    if frame.get("over_force_warning"):
        _circle(buf, panel_x + 96, 210, 12, (80, 80, 80), True)
    return bytes(buf)


def _write_ppm(path: Path, rgb: bytes) -> None:
    path.write_bytes(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + rgb)


def _encode_mp4(frames_dir: Path, output_mp4: Path, fps: int = FPS) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"created": False, "reason": "ffmpeg_not_found"}
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.ppm"),
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    return {
        "created": proc.returncode == 0 and output_mp4.exists(),
        "returncode": proc.returncode,
        "stderr": proc.stderr[-800:],
    }


def _zip_frames(frames_dir: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for frame in sorted(frames_dir.glob("frame_*.ppm")):
            zf.write(frame, frame.name)


def _redacted_task_id(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]


def run_trace(submission_dir: str | Path, task: dict[str, Any], variant: dict[str, Any] | None, redact: bool = False) -> dict[str, Any]:
    agent = AgentSubprocessProxy(submission_dir)
    env = DexSimEnv(task, variant)
    frames: list[dict[str, Any]] = []
    error = None
    try:
        agent.reset(sanitize_task_info(task))
        while not env.done and env.step_count < env.max_steps:
            obs = env.observation()
            try:
                action = agent.act(obs)
            except Exception as exc:
                action = {"primitive": "wait", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
                env.invalid_reason = "act_error"
                env.done = True
                error = f"act_error: {type(exc).__name__}"
            result = env.step(action)
            state = env.state()
            frames.append({
                "step": env.step_count,
                "pose": {
                    "x": round(float(state["final_pose"]["x"]), 5),
                    "y": round(float(state["final_pose"]["y"]), 5),
                    "theta": round(float(state["final_pose"]["theta"]), 5),
                },
                "action": action,
                "contact": round(float(state["min_contact"]), 5),
                "slip_risk": round(float(env.last_slip_risk), 5),
                "damage": round(float(state["damage"]), 5),
                "possible_drop": bool(state["drop_events"]),
                "resource_warning": bool(state["resource_violation"]),
                "over_force_warning": bool(obs.get("violation_flags", {}).get("over_force_warning")),
            })
            if result.get("invalid"):
                error = str(result["invalid"])
                break
    finally:
        agent.close()
    state = env.state()
    metrics = score_rollout(task, state)
    return {
        "task_id": _redacted_task_id(str(task["id"])) if redact else task["id"],
        "redacted": redact,
        "task_type": task["task_type"],
        "object_nominal": task.get("object_nominal", {}),
        "target_pose": task["target_pose"],
        "frame_count": len(frames),
        "metrics": metrics,
        "error": error,
        "frames": frames,
        "artifact_policy": (
            "diagnostic replay; no variant id, hard-slice label, private physics parameter, "
            "optimal action, server path, or full hidden set transcript"
        ),
    }


def write_replay(trace: dict[str, Any], task: dict[str, Any], output_dir: str | Path, prefix: str = "replay", make_video: bool = True) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"{prefix}_summary.json"
    summary = dict(trace)
    summary["frames"] = trace["frames"][:80]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    result: dict[str, Any] = {"summary": str(summary_path), "frame_count": len(trace["frames"])}
    with tempfile.TemporaryDirectory(prefix="ns202607_replay_") as tmp:
        frames_dir = Path(tmp)
        for i, frame in enumerate(trace["frames"][:80]):
            _write_ppm(frames_dir / f"frame_{i:04d}.ppm", _frame_png_data(frame, task))
        frames_zip = out / f"{prefix}_frames.zip"
        _zip_frames(frames_dir, frames_zip)
        result["frames_zip"] = str(frames_zip)
        if make_video:
            mp4 = out / f"{prefix}.mp4"
            result["mp4"] = _encode_mp4(frames_dir, mp4)
            if result["mp4"].get("created"):
                result["mp4"]["path"] = str(mp4)
    return result


def write_diagnostic_replay(
    submission_dir: str | Path,
    tasks: list[dict[str, Any]],
    artifact_dir: str | Path,
    prefix: str = "diagnostic_replay",
    redact: bool = True,
    make_video: bool = True,
) -> dict[str, Any]:
    if not tasks:
        raise RuntimeError("no tasks available for replay")
    task = tasks[0]
    variants = task.get("private_rollouts") or task.get("public_variants") or [{}]
    trace = run_trace(submission_dir, task, variants[0], redact=redact)
    return write_replay(trace, task, artifact_dir, prefix=prefix, make_video=make_video)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", help="Directory containing agent.py")
    parser.add_argument("--tasks", default="tasks/valid_tasks.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--output-dir", default="replay_output")
    parser.add_argument("--prefix", default="replay")
    parser.add_argument("--redact", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)

    task_path = Path(args.tasks)
    if not task_path.is_absolute():
        task_path = Path.cwd() / task_path
    tasks = read_jsonl(task_path)
    if not tasks:
        raise SystemExit("no tasks found")
    task = tasks[max(0, min(args.index, len(tasks) - 1))]
    variants = task.get("public_variants") or [{}]
    variant = variants[max(0, min(args.variant, len(variants) - 1))]
    trace = run_trace(args.submission_dir, task, variant, redact=args.redact)
    artifact = write_replay(trace, task, args.output_dir, prefix=args.prefix, make_video=not args.no_video)
    payload = {
        "task_file": str(task_path),
        "task_id": trace["task_id"],
        "task_type": trace["task_type"],
        "metrics": trace["metrics"],
        "artifact": artifact,
        "note": "Headless 2D public replay for debugging. Official hidden replay is redacted and bounded.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
