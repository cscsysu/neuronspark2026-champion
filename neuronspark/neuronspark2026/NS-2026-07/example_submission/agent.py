from __future__ import annotations

import math


def _norm(vec):
    x = float(vec[0])
    y = float(vec[1])
    n = math.sqrt(x * x + y * y)
    if n < 1e-9:
        return [0.0, 0.0]
    return [x / n, y / n]


class Agent:
    """A legal closed-loop baseline for public sanity evaluation.

    It intentionally does not use task ids, lookup tables, public variants, or
    simulator internals. The controller uses a small tool/resource budget, then
    aligns orientation and actively moves toward the target.
    """

    def reset(self, task_info):
        self.task_type = task_info["task_type"]
        self.tool_axis = task_info.get("tool_goal", {}).get("axis", [1.0, 0.0])
        self.reserved = task_info.get("resource_constraint", {}).get("reserve_fingers", [])
        self.scene = task_info.get("scene_context", {})
        self.obstacles = self.scene.get("obstacles", []) if isinstance(self.scene, dict) else []
        obj = task_info.get("object", {})
        self.fragile = float(obj.get("fragility", 0.0)) > 0.58
        self.taps_done = 0
        self._batch_agents = None

    def reset_batch(self, task_infos):
        self._batch_agents = [Agent() for _ in task_infos]
        for agent, task_info in zip(self._batch_agents, task_infos):
            agent.reset(task_info)

    def act_batch(self, observations):
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")
        actions = []
        for obs in observations:
            idx = int(obs.get("batch_index", len(actions)))
            actions.append(self._batch_agents[idx].act(obs))
        return actions

    def _cap(self, force, damage_risk):
        if self.fragile and damage_risk > 0.10:
            return min(force, 0.68)
        return force

    def _avoid_obstacles(self, direction, observation):
        pose = observation.get("object_pose_estimate", {})
        ox = float(pose.get("x", 0.0))
        oy = float(pose.get("y", 0.0))
        adjust_x = float(direction[0])
        adjust_y = float(direction[1])
        for obstacle in self.obstacles:
            try:
                cx = float(obstacle.get("x", 0.0))
                cy = float(obstacle.get("y", 0.0))
                radius = float(obstacle.get("radius", 0.08))
            except Exception:
                continue
            dx = ox - cx
            dy = oy - cy
            dist2 = dx * dx + dy * dy
            if dist2 < (radius + 0.18) ** 2 and dist2 > 1e-8:
                scale = 0.20 / max(0.08, math.sqrt(dist2))
                adjust_x += dx * scale
                adjust_y += dy * scale
        return _norm([adjust_x, adjust_y])

    def act(self, observation):
        step = int(observation["step"])
        state = observation["low_dim_state"]["values"]
        dx = float(state[0])
        dy = float(state[1])
        dtheta = float(state[2])
        dist = math.sqrt(dx * dx + dy * dy)
        contact = float(observation["contact_summary"]["coverage"])
        slip_risk = float(observation["contact_summary"]["slip_risk"])
        damage_risk = float(observation["contact_summary"]["damage_risk"])

        if dist < 0.085 and abs(dtheta) < 0.34:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        if step < 1 or contact < 0.04 or slip_risk > 0.92 or damage_risk > 0.98:
            return {"primitive": "brace", "finger": "palm", "force": 0.30, "direction": [0.0, 0.0]}

        if self.task_type in {"tool_use", "resource_sequence"} and self.taps_done < 8 and step % 3 == 0:
            self.taps_done += 1
            finger = self.reserved[0] if self.task_type == "resource_sequence" and self.reserved else "index"
            return {
                "primitive": "tap",
                "finger": finger,
                "force": self._cap(0.85, damage_risk),
                "direction": _norm(self.tool_axis),
            }

        if abs(dtheta) > 0.22:
            return {
                "primitive": "pivot",
                "finger": "thumb",
                "force": self._cap(0.95, damage_risk),
                "direction": [1.0 if dtheta >= 0.0 else -1.0, 0.0],
            }

        return {
            "primitive": "push",
            "finger": "palm",
            "force": self._cap(0.75, damage_risk),
            "direction": self._avoid_obstacles(_norm([dx, dy]), observation),
        }
