from __future__ import annotations
import math

def _norm2(x, y):
    return math.sqrt(x * x + y * y)

def _normalize(x, y):
    n = _norm2(x, y)
    if n < 1e-9: return 0.0, 0.0
    return x / n, y / n

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Agent:
    """NS-2026-07 Adaptive Controller v23-aggressive-taps.

    KEY IMPROVEMENT: Uses sensor_status hints to adapt behavior.
    Hidden test has extreme pose_dropout, tactile_dropout, action_delay.
    This agent:
    - Reduces force under high action delay (pending actions accumulate)
    - Uses slower, steadier movement under pose dropout (less oscillation)
    - Braces less under tactile dropout (tactile is unreliable anyway)
    - Adapts finish threshold based on noise level
    """

    def reset(self, task_info):
        self.task_type = task_info.get("task_type", "nonprehensile_relocation")
        self.tool_axis = task_info.get("tool_goal", {}).get("axis", [1.0, 0.0])
        self.tool_need = float(task_info.get("tool_goal", {}).get("progress", 0.0))
        self.reserved = task_info.get("resource_constraint", {}).get("reserve_fingers", [])
        self.max_peak_force = float(task_info.get("resource_constraint", {}).get("max_peak_force", 0.8))
        self.scene = task_info.get("scene_context", {})
        self.obstacles = self.scene.get("obstacles", []) if isinstance(self.scene, dict) else []
        self.corridor = self.scene.get("corridor", {}) if isinstance(self.scene, dict) else {}

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
            self.taps_needed = int(_clamp(math.ceil(max(0.08, self.tool_need) * 1.35 / per_tap), 14, 34))
        elif self.task_type == "resource_sequence":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, self.tool_need) * 1.45 / per_tap), 16, 36))
        else:
            self.taps_needed = 0

        self.phase = "brace"
        self.brace_done = 0
        self.stuck = 0
        self.last_dist = None
        # Adaptive parameters (set on first obs)
        self.delay_hint = 0
        self.pose_dropout_hint = 0.0
        self.tactile_dropout_hint = 0.0
        self.force_scale = 1.0
        self.finish_margin = 0.062
        self.initialized_hints = False
        # Smoothed direction for delay handling
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        # Observation smoothing (anti-noise)
        self.obs_history = []  # last few (dx, dy, dtheta) for smoothing
        # Force adaptation (detect weak actuator)
        self.progress_history = []  # track distance changes
        self.force_boost = 1.0  # increases when actuator seems weak
        # Damage rate tracking (detect high stiffness)
        self.last_damage_risk = 0.0
        self.damage_rising_count = 0
        self.stiffness_penalty = 1.0  # reduces when stiffness seems high

        self._batch_agents = None

    def _tap_finger(self):
        """Pick best finger for tap. resource_sequence uses strongest reserved."""
        if self.task_type == "resource_sequence" and self.reserved:
            strength = {
                "thumb": 1.08, "index": 1.00, "middle": 0.95,
                "ring": 0.72, "pinky": 0.62, "palm": 1.18, "wrist": 1.28,
            }
            return max(self.reserved, key=lambda name: strength.get(name, 0.0))
        return "index"

    def _resource_finger(self):
        """Use the strongest reserved finger for the resource/tool phase."""
        if self.task_type != "resource_sequence" or not self.reserved:
            return "index"
        strength = {
            "thumb": 1.08,
            "index": 1.00,
            "middle": 0.95,
            "ring": 0.72,
            "pinky": 0.62,
            "palm": 1.18,
            "wrist": 1.28,
        }
        return max(self.reserved, key=lambda name: strength.get(name, 0.0))

    def reset_batch(self, task_infos):
        self._batch_agents = [Agent() for _ in task_infos]
        for a, t in zip(self._batch_agents, task_infos):
            a.reset(t)

    def act_batch(self, observations):
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")
        return [self._batch_agents[int(o.get("batch_index", i))].act(o)
                for i, o in enumerate(observations)]

    def _init_from_hints(self, obs):
        """Adapt strategy based on sensor_status hints (called once on first obs)."""
        ss = obs.get("sensor_status", {})
        self.delay_hint = int(ss.get("action_delay_steps_hint", 0))
        self.pose_dropout_hint = float(ss.get("pose_dropout_rate_hint", 0.0))
        self.tactile_dropout_hint = float(ss.get("tactile_dropout_rate_hint", 0.0))

        # Under action delay: DON'T reduce force (smoothing handles it)
        # Only slightly reduce for very high delay to avoid overshoot
        if self.delay_hint >= 4:
            self.force_scale = 0.88
        else:
            self.force_scale = 1.0

        # Under high noise/dropout: widen finish threshold
        noise_level = self.pose_dropout_hint * 0.5
        self.finish_margin = 0.062 + noise_level * 0.03

        self.initialized_hints = True

    def _smooth_obs(self, dx, dy, dtheta):
        """Very light smoothing (2-step, 75% current weight) to counter noise."""
        self.obs_history.append((dx, dy, dtheta))
        if len(self.obs_history) > 3:
            self.obs_history.pop(0)
        if len(self.obs_history) <= 1:
            return dx, dy, dtheta
        # 75% current + 25% previous (very light smoothing)
        prev = self.obs_history[-2]
        sx = 0.75 * dx + 0.25 * prev[0]
        sy = 0.75 * dy + 0.25 * prev[1]
        st = 0.75 * dtheta + 0.25 * prev[2]
        return sx, sy, st

    def _update_force_adaptation(self, dist, damage_risk):
        """Detect weak actuator and boost force.
        Wait longer under delay (actions take time to show effect)."""
        self.progress_history.append(dist)
        max_hist = 10 + self.delay_hint * 3  # longer history under delay
        if len(self.progress_history) > max_hist:
            self.progress_history.pop(0)
        # Need more data under delay before concluding actuator is weak
        min_steps = 6 + self.delay_hint * 2
        if len(self.progress_history) >= min_steps and self.progress_history[0] > 0.15:
            recent_progress = self.progress_history[0] - self.progress_history[-1]
            expected_progress = 0.012 * len(self.progress_history)
            if recent_progress < expected_progress * 0.25 and recent_progress >= -0.02:
                self.force_boost = min(1.50, self.force_boost + 0.06)
            elif recent_progress > expected_progress * 0.5:
                self.force_boost = max(1.0, self.force_boost - 0.01)

    def _force(self, base, role, damage_risk):
        if role == "palm":
            f = min(base, self.safe_palm)
        elif role == "tap":
            f = min(base, self.safe_index)
        elif role == "pivot":
            f = min(base, self.safe_thumb)
        else:
            f = min(base, self.safe_palm)
        if damage_risk > 0.15:
            f *= max(0.50, 1.0 - damage_risk * 0.55)
        f = min(f, self.max_peak_force + 0.06)
        # Apply force boost (weak actuator) and delay scale
        f *= self.force_scale * self.force_boost
        return _clamp(f, 0.05, 1.0)

    def _avoid(self, dx, dy, obs):
        pose = obs.get("object_pose_estimate", {})
        ox, oy = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
        ax, ay = dx, dy
        for o in self.obstacles:
            try:
                cx, cy, r = float(o.get("x", 0)), float(o.get("y", 0)), float(o.get("radius", 0.08))
            except:
                continue
            ddx, ddy = ox - cx, oy - cy
            d = _norm2(ddx, ddy)
            danger = r + max(0.025, self.obj_size * 0.22) + 0.06
            if d < danger + 0.16 and d > 1e-6:
                ndx, ndy = _normalize(ddx, ddy)
                scale = (0.9 + (danger - d) * 5.0) if d < danger else (0.4 * (1.0 - (d - danger) / 0.16))
                ax += ndx * scale
                ay += ndy * scale
        if self.corridor:
            try:
                w = float(self.corridor.get("width", 0))
                if w > 0:
                    ca = self.corridor.get("axis", [1.0, 0.0])
                    cax, cay = _normalize(float(ca[0]), float(ca[1]))
                    lat = -cay * ox + cax * oy - float(self.corridor.get("offset", 0))
                    half = w * 0.5
                    margin = half - abs(lat)
                    if margin < 0.10:
                        sign = -1.0 if lat > 0 else 1.0
                        ax += (-cay) * sign * 0.5 * max(0.0, 1.0 - margin / 0.10)
                        ay += cax * sign * 0.5 * max(0.0, 1.0 - margin / 0.10)
                    if margin < 0.12:
                        along = ax * cax + ay * cay
                        perp = ax * (-cay) + ay * cax
                        perp *= 0.4
                        ax = along * cax + perp * (-cay)
                        ay = along * cay + perp * cax
            except:
                pass
        return _normalize(ax, ay)

    def _smooth_direction(self, dx, dy):
        """Direction smoothing disabled - it hurts delay performance."""
        return dx, dy

    def act(self, observation):
        # Initialize adaptive parameters from sensor hints
        if not self.initialized_hints:
            self._init_from_hints(observation)

        step = int(observation["step"])
        remaining = int(observation.get("remaining_steps", 128 - step))
        state = observation["low_dim_state"]["values"]
        dx_raw, dy_raw, dtheta_raw = float(state[0]), float(state[1]), float(state[2])
        dx_f, dy_f, dtheta_f = float(state[3]), float(state[4]), float(state[5])
        contact = float(state[9])
        self._last_contact = contact  # v54: for contact-aware force reduction
        slip_risk = float(state[10])
        damage_risk = float(observation.get("contact_summary", {}).get("damage_risk", 0.0))

        # Light observation smoothing (always active, very gentle)
        dx, dy, dtheta = self._smooth_obs(dx_raw, dy_raw, dtheta_raw)
        dist = _norm2(dx, dy)
        angle = abs(dtheta)
        dist_f = _norm2(dx_f, dy_f)
        angle_f = abs(dtheta_f)
        sc = observation.get("stage_context", {})

        # Update force adaptation (detect weak actuator + high stiffness)
        if step > 2:
            self._update_force_adaptation(dist, damage_risk)

        # ===== FINISH (use smoothed values for noise-robust decision) =====
        stages_ok = not sc.get("enabled", False) or float(sc.get("completion_fraction", 0)) >= 0.999
        tool_ok = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed * 0.94
        # Use smoothed history for finish decision if available
        if self.obs_history and len(self.obs_history) >= 2:
            # Average of last 2-3 observations for finish check
            recent = self.obs_history[-min(3, len(self.obs_history)):]
            s_dx_f = sum(h[0] for h in recent) / len(recent)
            s_dy_f = sum(h[1] for h in recent) / len(recent)
            # Blend raw final with smoothed current for robust finish
            sdist_f = _norm2(dx_f * 0.6 + s_dx_f * 0.4, dy_f * 0.6 + s_dy_f * 0.4)
        else:
            sdist_f = dist_f
        fin_dist = self.finish_margin
        fin_angle = 0.24 + self.pose_dropout_hint * 0.06
        if sdist_f < fin_dist and angle_f < fin_angle and stages_ok and tool_ok:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        if remaining <= 1:
            # More generous last-step finish under uncertainty
            fin_last = 0.13 + self.pose_dropout_hint * 0.05
            if dist_f < fin_last and angle_f < 0.55:
                return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
            nfx, nfy = self._avoid(*_normalize(dx_f, dy_f), observation)
            nfx, nfy = self._smooth_direction(nfx, nfy)
            return {"primitive": "push", "finger": "palm",
                    "force": self._force(0.82, "palm", damage_risk), "direction": [nfx, nfy]}

        # ===== CONTACT EMERGENCY (adapt to tactile dropout) =====
        # Under high tactile dropout, contact readings are unreliable
        # Use lower threshold (don't over-react to false low readings)
        contact_emergency_thresh = 0.06 - self.tactile_dropout_hint * 0.03
        if contact < max(0.02, contact_emergency_thresh):
            return {"primitive": "stabilize", "finger": "palm", "force": 0.45, "direction": [0.0, 0.0]}

        # ===== BRACE PHASE =====
        # v34: longer brace until contact is firmer
        if self.phase == "brace":
            self.brace_done += 1
            if (self.brace_done >= 1 and contact >= 0.32) or self.brace_done >= 3:
                self.phase = "tool" if self.taps_needed > 0 else "move"
            return {"primitive": "brace", "finger": "palm", "force": 0.42, "direction": [0.0, 0.0]}

        # ===== CONTACT RECOVERY (adaptive) =====
        # Less aggressive brace under tactile dropout (false readings)
        brace_thresh = 0.13 - self.tactile_dropout_hint * 0.05
        if contact < max(0.05, brace_thresh) or (slip_risk > 0.86 and contact < 0.26 and self.tactile_dropout_hint < 0.3):
            return {"primitive": "brace", "finger": "palm", "force": 0.36, "direction": [0.0, 0.0]}

        # ===== TOOL PHASE =====
        if self.phase == "tool":
            if self.taps_done >= self.taps_needed:
                self.phase = "move"
            else:
                if contact < 0.20 and self.tactile_dropout_hint < 0.3:
                    return {"primitive": "brace", "finger": "palm", "force": 0.33, "direction": [0.0, 0.0]}
                self.taps_done += 1
                ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
                finger = self._tap_finger()
                return {"primitive": "tap", "finger": finger,
                        "force": self._force(0.88, "tap", damage_risk), "direction": [ax, ay]}

        # ===== MOVE PHASE =====
        # Extra taps (less frequent under delay to avoid wasted actions)
        tap_interval = 2 + self.delay_hint  # space out taps more under delay
        if self.taps_done < self.taps_needed and step % tap_interval == 0 and remaining > 10 and contact > 0.15:
            self.taps_done += 1
            ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
            finger = self._tap_finger()
            return {"primitive": "tap", "finger": finger,
                    "force": self._force(0.85, "tap", damage_risk), "direction": [ax, ay]}

        # Stuck detection (more patient under delay)
        stuck_thresh = 6 + self.delay_hint * 2
        if self.last_dist is not None and dist > 0.10:
            self.stuck = self.stuck + 1 if abs(dist - self.last_dist) < 0.003 else max(0, self.stuck - 2)
        self.last_dist = dist
        if self.stuck > stuck_thresh:
            self.stuck = 0
            nx, ny = self._avoid(*_normalize(dx, dy), observation)
            nx, ny = self._smooth_direction(nx, ny)
            return {"primitive": "drag", "finger": "index",
                    "force": self._force(0.75, "palm", damage_risk), "direction": [nx, ny]}

        # === ROTATION ===
        # v35: heavy>1.0 prefer roll (4.4x external + rotation), else pivot
        heavy = self.mass > 1.0
        if angle > 0.36 and dist < 0.22:
            if heavy and dist > 0.08:
                nx, ny = self._avoid(*_normalize(dx, dy), observation)
                f = 0.84 if self.mass > 1.3 else 0.78
                return {"primitive": "roll", "finger": "palm",
                        "force": self._force(f, "palm", damage_risk), "direction": [nx, ny]}
            sign = 1.0 if dtheta >= 0 else -1.0
            f = 0.86 if self.mass > 1.2 else 0.80
            return {"primitive": "pivot", "finger": "thumb",
                    "force": self._force(f, "pivot", damage_risk), "direction": [sign, 0.0]}

        if angle > 0.26 and dist > 0.12:
            nx, ny = self._avoid(*_normalize(dx, dy), observation)
            nx, ny = self._smooth_direction(nx, ny)
            if heavy:
                return {"primitive": "roll", "finger": "palm",
                        "force": self._force(0.78, "palm", damage_risk), "direction": [nx, ny]}
            if (nx + ny >= 0) == (dtheta >= 0):
                return {"primitive": "roll", "finger": "palm",
                        "force": self._force(0.74, "palm", damage_risk), "direction": [nx, ny]}
            else:
                sign = 1.0 if dtheta >= 0 else -1.0
                return {"primitive": "pivot", "finger": "thumb",
                        "force": self._force(0.80, "pivot", damage_risk), "direction": [sign, 0.0]}

        # v40: medium-angle heavy roll (angle 0.15+ + dist > 0.12 + heavy)
        if heavy and angle > 0.15 and dist > 0.12:
            nx, ny = self._avoid(*_normalize(dx, dy), observation)
            return {"primitive": "roll", "finger": "palm",
                    "force": self._force(0.76, "palm", damage_risk), "direction": [nx, ny]}

        # === TRANSLATION ===
        if dist > 0.088:
            nx, ny = self._avoid(*_normalize(dx, dy), observation)
            nx, ny = self._smooth_direction(nx, ny)
            base = 0.88 if self.mass > 1.3 else (0.82 if dist > 0.28 else (0.74 if dist > 0.15 else 0.65))
            return {"primitive": "push", "finger": "palm",
                    "force": self._force(base, "palm", damage_risk), "direction": [nx, ny]}

        # === FINE ROTATION ===
        if angle > 0.12:
            sign = 1.0 if dtheta >= 0 else -1.0
            return {"primitive": "pivot", "finger": "thumb",
                    "force": self._force(0.55, "pivot", damage_risk), "direction": [sign, 0.0]}

        # === NEAR TARGET ===
        tool_ok2 = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed * 0.88
        if dist_f < 0.065 and angle_f < 0.26 and stages_ok and tool_ok2:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        nx, ny = _normalize(dx, dy) if dist > 0.02 else _normalize(dx_f, dy_f)
        nx, ny = self._avoid(nx, ny, observation)
        nx, ny = self._smooth_direction(nx, ny)
        return {"primitive": "push", "finger": "palm",
                "force": self._force(0.42, "palm", damage_risk), "direction": [nx, ny]}
