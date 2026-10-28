"""Final submission solver for NS-2026-09.

Key improvement: portal_used event only triggers on ACTUAL teleportation,
not just walking onto a P tile. This matches the gold label behavior observed
in train data (4 cases of A ending on P with portal_used=False).
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GRID_SIZE = 12
EVENT_KEYS = [
    "goal_reached",
    "collision",
    "hazard",
    "box_on_goal",
    "key_collected",
    "portal_used",
]
ENTITY_KEYS = ["A", "B", "O", "K"]
DIRS = {
    "U": (0, -1),
    "D": (0, 1),
    "L": (-1, 0),
    "R": (1, 0),
}
ARROW_DIRS = {
    "^": (0, -1),
    "v": (0, 1),
    "<": (-1, 0),
    ">": (1, 0),
}
DYNAMIC_GRID_CHARS = set("ABOKD")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def timeline_bucket(first_step: int | None, horizon: int) -> str:
    if first_step is None:
        return "never"
    if first_step <= horizon / 3:
        return "early"
    if first_step <= 2 * horizon / 3:
        return "mid"
    return "late"


def parse_entities(grid: list[str]) -> tuple[list[list[str]], dict[str, tuple[int, int]]]:
    base = [list(row) for row in grid]
    entities: dict[str, tuple[int, int]] = {}
    for y, row in enumerate(base):
        for x, ch in enumerate(row):
            if ch in ENTITY_KEYS:
                entities[ch] = (x, y)
                base[y][x] = "."
    return base, entities


def render_grid(base: list[list[str]], entities: dict[str, tuple[int, int]]) -> list[str]:
    out = [row[:] for row in base]
    for ch in ["K", "B", "O", "A"]:
        if ch in entities:
            x, y = entities[ch]
            out[y][x] = ch
    return ["".join(row) for row in out]


@dataclass(frozen=True)
class Profile:
    push_box: bool = False
    push_orb: bool = False
    door_requires_key: bool = True
    portal_mode: str = "pair"
    orb_portal: bool = False
    arrow_mode: str = "none"
    box_arrow_mode: str = "none"
    orb_arrow_mode: str = "none"
    ice_mode: str = "one"
    box_ice_mode: str = "none"
    orb_ice_mode: str = "none"
    orb_terrain_timing: str = "after"
    terrain_order: str = "after_action"
    orb_mode: str = "none"
    orb_period: int = 1
    orb_phase: int = 0
    orb_bounce: bool = False
    wait_is_action: bool = True


class Simulator:
    """World simulator with fixed portal_used event logic."""

    def __init__(self, grid: list[str], profile: Profile):
        self.base, self.entities = parse_entities(grid)
        self.profile = profile
        self.has_key = False
        self.events = {key: False for key in EVENT_KEYS}
        self.first_step: dict[str, int | None] = {key: None for key in EVENT_KEYS}
        self.event_order: list[str] = []
        self.orb_dir: str | None = None
        self.last_dir: str | None = None
        self.last_entity_dir: dict[str, str] = {}
        self._in_terrain = False
        if profile.orb_mode.startswith("drift_"):
            self.orb_dir = profile.orb_mode.split("_", 1)[1]

    def cell(self, x: int, y: int) -> str:
        return self.base[y][x]

    def entity_at(self, x: int, y: int, exclude: str | None = None) -> str | None:
        for ch, pos in self.entities.items():
            if ch != exclude and pos == (x, y):
                return ch
        return None

    def mark_event(self, key: str, step: int) -> None:
        if not self.events[key]:
            self.events[key] = True
            self.first_step[key] = step
            if len(self.event_order) < 3:
                self.event_order.append(key)

    def passable_static(self, x: int, y: int) -> bool:
        ch = self.cell(x, y)
        if ch == "#":
            return False
        if ch == "D" and self.profile.door_requires_key and not self.has_key:
            return False
        return True

    def try_move_entity(self, entity: str, dx: int, dy: int, step: int, *,
                        can_push: bool, collision_event: bool) -> bool:
        x, y = self.entities[entity]
        nx, ny = x + dx, y + dy
        if not self.passable_static(nx, ny):
            if collision_event:
                self.mark_event("collision", step)
            return False
        hit = self.entity_at(nx, ny, exclude=entity)
        if hit is not None:
            if entity == "A" and hit == "K":
                pass
            elif hit == "K" and entity in {"B", "O"}:
                pass
            elif hit in {"B", "O"} and (entity == "A" or can_push):
                push_allowed = (hit == "B" and self.profile.push_box) or (
                    hit == "O" and self.profile.push_orb
                )
                if push_allowed and self.try_move_entity(
                    hit, dx, dy, step, can_push=False, collision_event=False
                ):
                    pass
                else:
                    if collision_event:
                        self.mark_event("collision", step)
                    return False
            else:
                if collision_event:
                    self.mark_event("collision", step)
                return False
        self.entities[entity] = (nx, ny)
        for name, (vx, vy) in DIRS.items():
            if (dx, dy) == (vx, vy):
                self.last_entity_dir[entity] = name
                break
        self.handle_cell_events(entity, step)
        if entity in {"B", "O"} and self.profile.terrain_order != "none" and not self._in_terrain:
            if entity == "O" and self.profile.orb_terrain_timing != "after":
                return True
            arrow_mode = self.profile.box_arrow_mode if entity == "B" else self.profile.orb_arrow_mode
            ice_mode = self.profile.box_ice_mode if entity == "B" else self.profile.orb_ice_mode
            if arrow_mode != "none" or ice_mode != "none":
                self.apply_terrain(entity, step)
        return True

    def handle_cell_events(self, entity: str, step: int) -> None:
        """Handle events when entity arrives at a cell.

        KEY FIX: portal_used is NOT marked here. It is only marked in
        apply_portal when actual teleportation occurs. This matches the
        gold label behavior where walking onto P without teleporting
        does NOT trigger portal_used.
        """
        x, y = self.entities[entity]
        ch = self.cell(x, y)
        if entity == "A":
            if "K" in self.entities and self.entities["K"] == (x, y):
                del self.entities["K"]
                self.has_key = True
                self.mark_event("key_collected", step)
            if ch == "G":
                self.mark_event("goal_reached", step)
            elif ch == "H":
                self.mark_event("hazard", step)
            # NOTE: portal_used NOT marked here - only in apply_portal
        elif entity == "B":
            if ch == "G":
                self.mark_event("box_on_goal", step)
            elif ch == "H":
                self.mark_event("hazard", step)
        elif entity == "O":
            if ch == "H":
                self.mark_event("hazard", step)
            elif ch == "G":
                self.mark_event("goal_reached", step)

    def portal_targets(self) -> list[tuple[int, int]]:
        targets = []
        for y, row in enumerate(self.base):
            for x, ch in enumerate(row):
                if ch == "P":
                    targets.append((x, y))
        return targets

    def apply_portal(self, entity: str, step: int) -> bool:
        if self.profile.portal_mode == "none" or (entity != "A" and not self.profile.orb_portal):
            return False
        x, y = self.entities[entity]
        if self.cell(x, y) != "P":
            return False
        portals = self.portal_targets()
        if len(portals) < 2:
            return False
        try:
            idx = portals.index((x, y))
        except ValueError:
            return False
        if self.profile.portal_mode == "pair":
            target = portals[1 - idx] if len(portals) == 2 else portals[(idx + 1) % len(portals)]
        else:
            target = portals[(idx + 1) % len(portals)]
        if self.entity_at(*target, exclude=entity) is None and self.passable_static(*target):
            self.entities[entity] = target
            if entity == "A":
                self.mark_event("portal_used", step)
            self.handle_cell_events(entity, step)
            return True
        return False

    def apply_terrain_once(self, entity: str, step: int) -> bool:
        x, y = self.entities[entity]
        ch = self.cell(x, y)
        if entity == "A":
            arrow_mode = self.profile.arrow_mode
        elif entity == "B":
            arrow_mode = self.profile.box_arrow_mode
        else:
            arrow_mode = self.profile.orb_arrow_mode
        if ch in ARROW_DIRS and arrow_mode != "none":
            dx, dy = ARROW_DIRS[ch]
            return self.try_move_entity(entity, dx, dy, step, can_push=True, collision_event=(entity == "A"))
        if ch == "I":
            if entity == "A":
                ice_mode = self.profile.ice_mode
                direction = self.last_dir
            elif entity == "B":
                ice_mode = self.profile.box_ice_mode
                direction = self.last_entity_dir.get(entity)
            else:
                ice_mode = self.profile.orb_ice_mode
                direction = self.last_entity_dir.get(entity)
            if ice_mode == "none":
                return self.apply_portal(entity, step)
            if direction in DIRS:
                dx, dy = DIRS[direction]
                return self.try_move_entity(entity, dx, dy, step, can_push=True, collision_event=(entity == "A"))
        return self.apply_portal(entity, step)

    def apply_terrain(self, entity: str, step: int) -> None:
        if self.profile.terrain_order == "none":
            return
        limit = 1
        if entity == "A":
            arrow_mode = self.profile.arrow_mode
            ice_mode = self.profile.ice_mode
        elif entity == "B":
            arrow_mode = self.profile.box_arrow_mode
            ice_mode = self.profile.box_ice_mode
        else:
            arrow_mode = self.profile.orb_arrow_mode
            ice_mode = self.profile.orb_ice_mode
        if arrow_mode == "chain" or ice_mode == "slide":
            limit = 20
        seen = set()
        old_flag = self._in_terrain
        self._in_terrain = True
        try:
            for _ in range(limit):
                state = (entity, self.entities.get(entity), getattr(self, "last_dir", None), self.last_entity_dir.get(entity))
                if state in seen:
                    break
                seen.add(state)
                moved = self.apply_terrain_once(entity, step)
                if not moved:
                    break
        finally:
            self._in_terrain = old_flag

    def move_orb(self, step: int) -> None:
        if "O" not in self.entities or self.orb_dir not in DIRS:
            return
        if (step + self.profile.orb_phase) % self.profile.orb_period != 0:
            return
        if self.profile.orb_terrain_timing == "before" and self.profile.terrain_order != "none":
            self.apply_terrain("O", step)
        dx, dy = DIRS[self.orb_dir]
        ok = self.try_move_entity("O", dx, dy, step, can_push=False, collision_event=False)
        if not ok and self.profile.orb_bounce:
            opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}[self.orb_dir]
            self.orb_dir = opposite

    def step(self, action: str, step: int) -> None:
        if self.profile.terrain_order == "before_action_after_action":
            self.apply_terrain("A", step)
        if action in DIRS:
            self.last_dir = action
            dx, dy = DIRS[action]
            self.try_move_entity("A", dx, dy, step, can_push=True, collision_event=True)
        if self.profile.terrain_order in {"after_action", "before_action_after_action"}:
            self.apply_terrain("A", step)
        self.move_orb(step)

    def run(self, actions: list[str]) -> dict[str, Any]:
        for step, action in enumerate(actions, 1):
            self.step(action, step)
        grid = render_grid(self.base, self.entities)
        timeline = {key: timeline_bucket(self.first_step[key], len(actions)) for key in EVENT_KEYS}
        order = (self.event_order + ["none", "none", "none"])[:3]
        terminal = infer_terminal(self.events)
        return {
            "final_grid": grid,
            "events": dict(self.events),
            "event_timeline": timeline,
            "event_order": order,
            "terminal": terminal,
        }


def infer_terminal(events: dict[str, bool]) -> str:
    if events["goal_reached"]:
        return "goal"
    if events["hazard"]:
        return "hazard"
    if events["collision"]:
        return "blocked"
    return "timeout"


def simulate(grid: list[str], actions: list[str], profile: Profile) -> dict[str, Any]:
    return Simulator(grid, profile).run(actions)


def label_from_probe(ep: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_grid": ep["observed_final_full_grid"],
        "events": ep["observed_final_events"],
        "event_timeline": ep["observed_final_event_timeline"],
        "event_order": ep["observed_final_event_order"],
        "terminal": ep["observed_final_terminal"],
    }


def entity_position(grid: list[str], ch: str) -> tuple[tuple[int, int], ...] | None:
    pos = tuple((x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == ch)
    return pos or None


def label_match_score(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    score = 0.0
    pg = pred["final_grid"]
    gg = gold["final_grid"]
    cells = 0
    ok = 0
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if pg[y][x] in DYNAMIC_GRID_CHARS or gg[y][x] in DYNAMIC_GRID_CHARS:
                cells += 1
                ok += pg[y][x] == gg[y][x]
    if cells:
        score += 8.0 * ok / cells
    for ch in ENTITY_KEYS:
        pp = entity_position(pg, ch)
        gp = entity_position(gg, ch)
        if pp is not None or gp is not None:
            score += 2.0 if pp == gp else 0.0
    score += 0.5 * sum(pred["events"][k] == gold["events"][k] for k in EVENT_KEYS)
    score += 8.0 if pred["event_order"] == gold["event_order"] else 0.0
    score += 0.25 if pred["terminal"] == gold["terminal"] else 0.0
    score += 0.25 * sum(pred["event_timeline"][k] == gold["event_timeline"][k] for k in EVENT_KEYS)
    score += 0.25 if pg == gg else 0.0
    return score


def default_profiles() -> list[Profile]:
    profiles: list[Profile] = []
    for push_box, push_orb in itertools.product([True, False], repeat=2):
        for orb_arrow_mode, orb_ice_mode, box_arrow_mode in itertools.product(
            ["none", "one"], ["none", "one"], ["none", "one"]
        ):
            timings = ["after"] if orb_arrow_mode == "none" and orb_ice_mode == "none" else ["after", "before"]
            base = {
                "push_box": push_box,
                "push_orb": push_orb,
                "door_requires_key": True,
                "portal_mode": "pair",
                "arrow_mode": "none",
                "ice_mode": "one",
                "terrain_order": "after_action",
                "orb_portal": False,
                "box_ice_mode": "none",
                "orb_arrow_mode": orb_arrow_mode,
                "orb_ice_mode": orb_ice_mode,
                "box_arrow_mode": box_arrow_mode,
            }
            for timing in timings:
                profiles.append(Profile(**base, orb_terrain_timing=timing, orb_mode="none"))
                for direction in "UDLR":
                    profiles.append(Profile(**base, orb_terrain_timing=timing, orb_mode=f"drift_{direction}"))
    return profiles


def choose_profile(row: dict[str, Any], profiles: list[Profile]) -> Profile:
    best = profiles[0]
    best_score = -1e9
    for profile in profiles:
        total = 0.0
        for ep in row["context_episodes"]:
            pred = simulate(row["initial_full_grid"], ep["actions"], profile)
            total += label_match_score(pred, label_from_probe(ep))
        if total > best_score:
            best_score = total
            best = profile
    return best


def predict_rows(rows: list[dict[str, Any]], profiles: list[Profile]) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        profile = choose_profile(row, profiles)
        for query in row["queries"]:
            pred = simulate(query["initial_full_grid"], query["future_actions"], profile)
            results.append({
                "id": query["query_id"],
                "final_grid": pred["final_grid"],
                "events": pred["events"],
                "event_timeline": pred["event_timeline"],
                "event_order": pred["event_order"],
                "terminal": pred["terminal"],
            })
    return results


def valid_golds(rows):
    return {query["query_id"]: query["label"] for row in rows for query in row["queries"]}


def score_predictions(preds, golds):
    n = len(golds)
    dyn_ok = dyn_total = ent_ok = ent_total = event_ok = 0
    timeline_ok = timeline_total = order_ok = terminal_ok = 0
    for pred in preds:
        gold = golds[pred["id"]]
        pg = pred["final_grid"]
        gg = gold["final_grid"]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if pg[y][x] in DYNAMIC_GRID_CHARS or gg[y][x] in DYNAMIC_GRID_CHARS:
                    dyn_total += 1
                    dyn_ok += pg[y][x] == gg[y][x]
        for ch in ENTITY_KEYS:
            pp = entity_position(pg, ch)
            gp = entity_position(gg, ch)
            if pp is not None or gp is not None:
                ent_total += 1
                ent_ok += pp == gp
        event_ok += all(pred["events"][key] == gold["events"][key] for key in EVENT_KEYS)
        for key in EVENT_KEYS:
            if gold["event_timeline"][key] != "never":
                timeline_total += 1
                timeline_ok += pred["event_timeline"][key] == gold["event_timeline"][key]
        order_ok += pred["event_order"] == gold["event_order"]
        terminal_ok += pred["terminal"] == gold["terminal"]
    metrics = {
        "dynamic_grid_accuracy": dyn_ok / dyn_total if dyn_total else 1.0,
        "entity_accuracy": ent_ok / ent_total if ent_total else 1.0,
        "event_vector_accuracy": event_ok / n,
        "event_timeline_accuracy": timeline_ok / timeline_total if timeline_total else 1.0,
        "event_order_accuracy": order_ok / n,
        "terminal_accuracy": terminal_ok / n,
    }
    score = (100 + 225 * metrics["dynamic_grid_accuracy"] + 225 * metrics["entity_accuracy"]
             + 250 * metrics["event_vector_accuracy"] + 250 * metrics["event_timeline_accuracy"]
             + 250 * metrics["event_order_accuracy"] + 200 * metrics["terminal_accuracy"])
    return score, metrics


def cmd_eval(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.tasks)
    profiles = default_profiles()
    print(f"Profile count: {len(profiles)}")
    preds = predict_rows(rows, profiles)
    score, metrics = score_predictions(preds, valid_golds(rows))
    print(f"score={score:.6f}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.tasks)
    profiles = default_profiles()
    print(f"Profile count: {len(profiles)}")
    preds = predict_rows(rows, profiles)
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(preds)} predictions to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--tasks", default="valid.jsonl")
    eval_p.set_defaults(func=cmd_eval)
    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--tasks", default="test.jsonl")
    pred_p.add_argument("--out", default="results.json")
    pred_p.set_defaults(func=cmd_predict)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
