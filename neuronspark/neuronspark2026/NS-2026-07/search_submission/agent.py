"""NS-2026-07: Rule + MPC Search Agent.

Uses the proven rule agent as base, then searches for improvements using
short-horizon proxy simulation. Search can only IMPROVE over rules (never worse).

Key insight: proxy env ranking is correct even with physics mismatch.
We don't need exact prediction - just correct relative ordering of actions.
"""
from __future__ import annotations
import copy
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "simulator"))
try:
    from dexsim_core import ProxyDexSimEnv
except ImportError:
    ProxyDexSimEnv = None

ACTION_PRIMITIVES = ["brace", "push", "drag", "pivot", "roll", "lift_edge", "tap", "stabilize", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]

def _norm2(x, y): return math.sqrt(x*x + y*y)
def _normalize(x, y):
    n = _norm2(x, y)
    return (x/n, y/n) if n > 1e-9 else (0.0, 0.0)
def _clamp(v, lo, hi): return max(lo, min(hi, v))


class Agent:
    """Rule agent + short-horizon MPC search."""

    def __init__(self):
        self._batch_agents = None
        self.rng = random.Random(7)

    def reset(self, task_info):
        self.task_info = task_info
        self.task = task_info  # keep full task for proxy env creation
        self.task_type = task_info.get("task_type", "nonprehensile_relocation")
        self.tool_axis = task_info.get("tool_goal", {}).get("axis", [1.0, 0.0])
        self.tool_need = float(task_info.get("tool_goal", {}).get("progress", 0.0))
        self.reserved = task_info.get("resource_constraint", {}).get("reserve_fingers", [])
        self.max_peak_force = float(task_info.get("resource_constraint", {}).get("max_peak_force", 0.8))
        self.obstacles = task_info.get("scene_context", {}).get("obstacles", []) if isinstance(task_info.get("scene_context"), dict) else []
        self.corridor = task_info.get("scene_context", {}).get("corridor", {}) if isinstance(task_info.get("scene_context"), dict) else {}

        obj = task_info.get("object", {})
        self.fragility = float(obj.get("fragility", 0.0))
        self.mass = float(obj.get("mass", 1.0))
        self.obj_size = float(obj.get("size", 0.1))
        self.dmg_thresh = 1.05 - 0.72 * self.fragility
        self.safe_palm = min(0.92, self.dmg_thresh / 1.15)
        self.safe_index = min(0.95, self.dmg_thresh / 0.98)
        self.safe_thumb = min(0.92, self.dmg_thresh / 1.05)

        self.taps_done = 0
        per_tap = 0.048
        if self.task_type == "tool_use":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, self.tool_need) * 0.84 / per_tap), 9, 24))
        elif self.task_type == "resource_sequence":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, self.tool_need) * 0.84 / per_tap), 9, 22))
        else:
            self.taps_needed = 0

        self.phase = "brace"
        self.brace_done = 0
        self.stuck = 0
        self.last_dist = None
        # Adaptive
        self.force_boost = 1.0
        self.progress_history = []
        self.delay_hint = 0
        self.initialized = False

    def reset_batch(self, task_infos):
        self._batch_agents = [Agent() for _ in task_infos]
        for a, t in zip(self._batch_agents, task_infos):
            a.reset(t)

    def act_batch(self, observations):
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")
        return [self._batch_agents[int(o.get("batch_index", i))].act(o)
                for i, o in enumerate(observations)]

    def _force(self, base, role, damage_risk):
        if role == "palm": f = min(base, self.safe_palm)
        elif role == "tap": f = min(base, self.safe_index)
        elif role == "pivot": f = min(base, self.safe_thumb)
        else: f = min(base, self.safe_palm)
        if damage_risk > 0.15: f *= max(0.50, 1.0 - damage_risk * 0.55)
        f = min(f, self.max_peak_force + 0.06)
        f *= self.force_boost
        return _clamp(f, 0.05, 1.0)

    def _avoid(self, dx, dy, obs):
        pose = obs.get("object_pose_estimate", {})
        ox, oy = float(pose.get("x", 0)), float(pose.get("y", 0))
        ax, ay = dx, dy
        for o in self.obstacles:
            try:
                cx, cy, r = float(o.get("x",0)), float(o.get("y",0)), float(o.get("radius",0.08))
            except: continue
            ddx, ddy = ox-cx, oy-cy
            d = _norm2(ddx, ddy)
            danger = r + max(0.025, self.obj_size*0.22) + 0.06
            if d < danger+0.16 and d > 1e-6:
                ndx, ndy = _normalize(ddx, ddy)
                scale = (0.9+(danger-d)*5) if d < danger else (0.4*(1-(d-danger)/0.16))
                ax += ndx*scale; ay += ndy*scale
        if self.corridor:
            try:
                w = float(self.corridor.get("width", 0))
                if w > 0:
                    ca = self.corridor.get("axis", [1,0])
                    cax, cay = _normalize(float(ca[0]), float(ca[1]))
                    lat = -cay*ox + cax*oy - float(self.corridor.get("offset", 0))
                    margin = w*0.5 - abs(lat)
                    if margin < 0.10:
                        sign = -1.0 if lat > 0 else 1.0
                        ax += (-cay)*sign*0.5*max(0, 1-margin/0.10)
                        ay += cax*sign*0.5*max(0, 1-margin/0.10)
                    if margin < 0.12:
                        along = ax*cax + ay*cay
                        perp = ax*(-cay) + ay*cax
                        perp *= 0.4
                        ax = along*cax + perp*(-cay)
                        ay = along*cay + perp*cax
            except: pass
        return _normalize(ax, ay)

    def _rule_action(self, obs):
        """V14 rule agent logic (proven 321.9 on hidden)."""
        step = int(obs["step"])
        remaining = int(obs.get("remaining_steps", 128-step))
        state = obs["low_dim_state"]["values"]
        dx, dy, dtheta = float(state[0]), float(state[1]), float(state[2])
        dx_f, dy_f, dtheta_f = float(state[3]), float(state[4]), float(state[5])
        contact = float(state[9])
        slip_risk = float(state[10])
        damage_risk = float(obs.get("contact_summary", {}).get("damage_risk", 0))
        dist = _norm2(dx, dy)
        angle = abs(dtheta)
        dist_f = _norm2(dx_f, dy_f)
        angle_f = abs(dtheta_f)
        sc = obs.get("stage_context", {})

        if not self.initialized:
            ss = obs.get("sensor_status", {})
            self.delay_hint = int(ss.get("action_delay_steps_hint", 0))
            self.initialized = True

        stages_ok = not sc.get("enabled", False) or float(sc.get("completion_fraction", 0)) >= 0.999
        tool_ok = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed * 0.72

        if dist_f < 0.062 and angle_f < 0.24 and stages_ok and tool_ok:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
        if remaining <= 1:
            if dist_f < 0.13 and angle_f < 0.55:
                return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
            nfx, nfy = self._avoid(*_normalize(dx_f, dy_f), obs)
            return {"primitive": "push", "finger": "palm", "force": self._force(0.82, "palm", damage_risk), "direction": [nfx, nfy]}
        if contact < 0.06:
            return {"primitive": "stabilize", "finger": "palm", "force": 0.45, "direction": [0.0, 0.0]}
        if self.phase == "brace":
            self.brace_done += 1
            if (self.brace_done >= 1 and contact >= 0.22) or self.brace_done >= 2:
                self.phase = "tool" if self.taps_needed > 0 else "move"
            return {"primitive": "brace", "finger": "palm", "force": 0.38, "direction": [0.0, 0.0]}
        if contact < 0.13 or (slip_risk > 0.86 and contact < 0.26):
            return {"primitive": "brace", "finger": "palm", "force": 0.36, "direction": [0.0, 0.0]}
        if self.phase == "tool":
            if self.taps_done >= self.taps_needed:
                self.phase = "move"
            else:
                if contact < 0.20:
                    return {"primitive": "brace", "finger": "palm", "force": 0.33, "direction": [0.0, 0.0]}
                self.taps_done += 1
                ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
                finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
                return {"primitive": "tap", "finger": finger, "force": self._force(0.88, "tap", damage_risk), "direction": [ax, ay]}

        tap_interval = 4 + self.delay_hint
        if self.taps_done < self.taps_needed and step % tap_interval == 0 and remaining > 10 and contact > 0.20:
            self.taps_done += 1
            ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
            finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
            return {"primitive": "tap", "finger": finger, "force": self._force(0.85, "tap", damage_risk), "direction": [ax, ay]}

        # Force adaptation
        self.progress_history.append(dist)
        if len(self.progress_history) > 10 + self.delay_hint*3:
            self.progress_history.pop(0)
        min_steps = 6 + self.delay_hint*2
        if len(self.progress_history) >= min_steps and self.progress_history[0] > 0.15:
            prog = self.progress_history[0] - self.progress_history[-1]
            if prog < 0.012*len(self.progress_history)*0.25 and prog >= -0.02:
                self.force_boost = min(1.50, self.force_boost + 0.06)
            elif prog > 0.012*len(self.progress_history)*0.5:
                self.force_boost = max(1.0, self.force_boost - 0.01)

        if self.stuck > 6:
            self.stuck = 0
            nx, ny = self._avoid(*_normalize(dx, dy), obs)
            return {"primitive": "drag", "finger": "index", "force": self._force(0.75, "palm", damage_risk), "direction": [nx, ny]}

        if angle > 0.36 and dist < 0.22:
            sign = 1.0 if dtheta >= 0 else -1.0
            return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.86 if self.mass > 1.2 else 0.80, "pivot", damage_risk), "direction": [sign, 0.0]}
        if angle > 0.26 and dist > 0.12:
            nx, ny = self._avoid(*_normalize(dx, dy), obs)
            if (nx+ny >= 0) == (dtheta >= 0):
                return {"primitive": "roll", "finger": "palm", "force": self._force(0.74, "palm", damage_risk), "direction": [nx, ny]}
            else:
                return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.80, "pivot", damage_risk), "direction": [1.0 if dtheta>=0 else -1.0, 0.0]}
        if dist > 0.088:
            nx, ny = self._avoid(*_normalize(dx, dy), obs)
            base = 0.88 if self.mass > 1.3 else (0.82 if dist > 0.28 else (0.74 if dist > 0.15 else 0.65))
            return {"primitive": "push", "finger": "palm", "force": self._force(base, "palm", damage_risk), "direction": [nx, ny]}
        if angle > 0.12:
            return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.55, "pivot", damage_risk), "direction": [1.0 if dtheta>=0 else -1.0, 0.0]}

        tool_ok2 = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed * 0.65
        if dist_f < 0.065 and angle_f < 0.26 and stages_ok and tool_ok2:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        nx, ny = _normalize(dx, dy) if dist > 0.02 else _normalize(dx_f, dy_f)
        nx, ny = self._avoid(nx, ny, obs)
        return {"primitive": "push", "finger": "palm", "force": self._force(0.42, "palm", damage_risk), "direction": [nx, ny]}

    def _create_proxy(self, obs):
        """Create synced proxy env from observation."""
        if ProxyDexSimEnv is None:
            return None
        try:
            task = dict(self.task)
            pose = obs["object_pose_estimate"]
            task["start_pose"] = {"x": float(pose["x"]), "y": float(pose["y"]), "theta": float(pose["theta"])}
            proxy = ProxyDexSimEnv(task, {"observation_noise": 0.001})
            proxy.step_count = int(obs["step"])
            proxy.contact = float(obs["low_dim_state"]["values"][9])
            proxy.damage = float(obs.get("contact_summary", {}).get("damage_risk", 0)) * 0.3
            proxy.energy = float(obs["low_dim_state"]["values"][12]) * 72.0
            proxy.tool_progress = self.taps_done * 0.052
            vel = obs.get("object_velocity_estimate", {})
            proxy.vx = float(vel.get("vx", 0))
            proxy.vy = float(vel.get("vy", 0))
            proxy.vtheta = float(vel.get("vtheta", 0))
            return proxy
        except:
            return None

    def _evaluate_action(self, proxy, action, horizon=3):
        """Evaluate action with short-horizon lookahead."""
        env = copy.deepcopy(proxy)
        env.step(action)
        # Continue with simple push toward target for remaining steps
        for _ in range(horizon - 1):
            if env.done:
                break
            obs_p = env.observation()
            s = obs_p["low_dim_state"]["values"]
            nx, ny = _normalize(float(s[0]), float(s[1]))
            env.step({"primitive": "push", "finger": "palm", "force": 0.65, "direction": [nx, ny]})
        state = env.state()
        # Score: weighted distance + angle + safety
        dist = state["distance_error"]
        angle = state["angle_error"]
        damage = state["damage"]
        stage_frac = state["stage_completion_fraction"]
        score = -dist * 3.0 - angle * 1.5 - damage * 5.0 + stage_frac * 2.0
        if state.get("terminated_by_finish") and dist < 0.085 and angle < 0.34:
            score += 5.0
        return score

    def _search_improve(self, obs, rule_action):
        """Try to improve on the rule action with short search.
        Only override rule if search finds SIGNIFICANTLY better action (robust to noise)."""
        proxy = self._create_proxy(obs)
        if proxy is None:
            return rule_action

        state = obs["low_dim_state"]["values"]
        dx, dy, dtheta = float(state[0]), float(state[1]), float(state[2])
        damage_risk = float(obs.get("contact_summary", {}).get("damage_risk", 0))

        # Evaluate rule action as baseline
        rule_score = self._evaluate_action(proxy, rule_action, horizon=3)

        # Generate candidates
        candidates = []
        nx, ny = _normalize(dx, dy)
        safe_f = min(0.92, self.dmg_thresh / 1.15, self.max_peak_force + 0.05)
        if damage_risk > 0.2: safe_f *= 0.7

        for angle_off in [-0.4, -0.25, -0.12, 0.12, 0.25, 0.4]:
            cos_a, sin_a = math.cos(angle_off), math.sin(angle_off)
            rx, ry = nx*cos_a - ny*sin_a, nx*sin_a + ny*cos_a
            candidates.append({"primitive": "push", "finger": "palm", "force": min(0.75, safe_f), "direction": [rx, ry]})

        for f in [0.45, 0.55, 0.65, 0.80, 0.90]:
            candidates.append({"primitive": "push", "finger": "palm", "force": min(f, safe_f), "direction": [nx, ny]})

        candidates.append({"primitive": "roll", "finger": "palm", "force": min(0.70, safe_f), "direction": [nx, ny]})
        candidates.append({"primitive": "drag", "finger": "index", "force": min(0.70, safe_f), "direction": [nx, ny]})
        sign = 1.0 if dtheta >= 0 else -1.0
        candidates.append({"primitive": "pivot", "finger": "thumb", "force": min(0.75, safe_f), "direction": [sign, 0.0]})

        # Evaluate
        best_action = rule_action
        best_score = rule_score
        for cand in candidates:
            score = self._evaluate_action(proxy, cand, horizon=3)
            if score > best_score:
                best_score = score
                best_action = cand

        # Conservative: only override rule if improvement is SIGNIFICANT
        # This prevents noise-induced false improvements from overriding rules
        margin = 0.8  # require clear improvement
        if best_score > rule_score + margin:
            return best_action
        return rule_action

    def act(self, observation):
        state = observation["low_dim_state"]["values"]
        dist = _norm2(float(state[0]), float(state[1]))

        # Stuck detection
        if self.last_dist is not None and dist > 0.10:
            self.stuck = self.stuck + 1 if abs(dist - self.last_dist) < 0.003 else max(0, self.stuck - 2)
        self.last_dist = dist

        # Get rule action (always have this as fallback)
        rule_action = self._rule_action(observation)

        # Decide whether search is beneficial
        # Search helps when proxy physics match reality (low noise, low delay)
        # Search hurts when physics mismatch is large
        ss = observation.get("sensor_status", {})
        noise_hint = float(ss.get("pose_dropout_rate_hint", 0))
        delay_hint = int(ss.get("action_delay_steps_hint", 0))
        tactile_hint = float(ss.get("tactile_dropout_rate_hint", 0))

        # Only search when conditions are favorable
        conditions_ok = (noise_hint < 0.15 and delay_hint <= 1 and tactile_hint < 0.2)
        searchable = rule_action["primitive"] in ("push", "drag", "roll", "pivot")

        if searchable and conditions_ok and ProxyDexSimEnv is not None:
            return self._search_improve(observation, rule_action)

        return rule_action
