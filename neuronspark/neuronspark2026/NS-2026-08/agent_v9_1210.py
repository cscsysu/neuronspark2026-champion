# NS-2026-08 agent v1 — original (best so far). CPU-only, stdlib only.
from collections import defaultdict, deque

HELPFUL = {
    "release_instability": {"ROLLBACK_RELEASE", "CIRCUIT_BREAK"},
    "config_shift": {"ROLLBACK_CONFIG", "SET_TIMEOUT"},
    "heap_growth": {"RESTART", "SCALE"},
    "traffic_surge": {"SCALE", "THROTTLE"},
    "storage_wait": {"REBUILD_INDEX", "SET_TIMEOUT", "RESTART"},
    "consumer_lag": {"SCALE", "DRAIN_QUEUE"},
    "cache_collapse": {"WARM_CACHE", "THROTTLE", "CIRCUIT_BREAK"},
    "dependency_timeout": {"CIRCUIT_BREAK", "SET_TIMEOUT"},
}
ACTION_HINT_MODE = {
    "rollback_release": "release_instability",
    "rollback_config": "config_shift",
    "restart_scale": "heap_growth",
    "scale_throttle": "traffic_surge",
    "storage_index": "storage_wait",
    "queue_drain": "consumer_lag",
    "cache_warm": "cache_collapse",
    "timeout_circuit": "dependency_timeout",
}
ACTION_COOLDOWN = {
    "SCALE": 4, "ROLLBACK_RELEASE": 12, "ROLLBACK_CONFIG": 10, "RESTART": 8,
    "CIRCUIT_BREAK": 6, "SET_TIMEOUT": 5, "WARM_CACHE": 5, "REBUILD_INDEX": 16,
    "DRAIN_QUEUE": 7, "THROTTLE": 6, "DIAGNOSTIC_PROBE": 5,
}
ACTION_COST = {
    "SCALE": 1.3, "ROLLBACK_RELEASE": 4.2, "ROLLBACK_CONFIG": 3.4, "RESTART": 2.7,
    "CIRCUIT_BREAK": 2.9, "SET_TIMEOUT": 1.8, "WARM_CACHE": 2.4, "REBUILD_INDEX": 5.7,
    "DRAIN_QUEUE": 3.2, "THROTTLE": 2.8, "DIAGNOSTIC_PROBE": 1.6, "NOOP": 0.0,
}


ROLE = {
    "svc_00": "edge", "svc_01": "api", "svc_02": "api", "svc_03": "api",
    "svc_04": "api", "svc_05": "store", "svc_06": "store",
    "svc_07": "cache", "svc_08": "queue", "svc_09": "worker",
}

# Empirical from public data: which (mode, mode.service) combos are legal.
LEGAL_MODE_SERVICE = {
    "release_instability": {"svc_01", "svc_02", "svc_03", "svc_04", "svc_09"},
    "config_shift": {"svc_00", "svc_01", "svc_02", "svc_03", "svc_04"},
    "heap_growth": {"svc_00", "svc_02", "svc_03", "svc_04", "svc_09"},
    "traffic_surge": {"svc_00", "svc_02", "svc_03"},
    "storage_wait": {"svc_02", "svc_03", "svc_04"},
    "consumer_lag": {"svc_09"},
    "cache_collapse": {"svc_00", "svc_01", "svc_02"},
    "dependency_timeout": {"svc_00", "svc_02", "svc_03", "svc_04"},
}


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


class Agent:
    def reset(self, task_info):
        self.task_info = task_info or {}
        self.services = {s["alias"]: s for s in self.task_info.get("services", [])}
        self.aliases = list(self.services.keys())
        self.topology = self.task_info.get("topology", []) or []
        self.traffic = list(self.task_info.get("traffic_forecast", []) or [])
        self.horizon = int(self.task_info.get("horizon", 48))

        self.neighbors = {a: set() for a in self.aliases}
        for e in self.topology:
            f, t = e.get("from"), e.get("to")
            if f in self.neighbors and t:
                self.neighbors[f].add(t)
            if t in self.neighbors and f:
                self.neighbors[t].add(f)

        self.last_used_step = {}
        self.last_action_only = {}

        self.dispatched = defaultdict(int)

        self.history = {a: deque(maxlen=8) for a in self.aliases}
        self.events_seen = defaultdict(list)
        self.channel_count = defaultdict(int)

        self.channel_mode = {}

        self.replicas = {a: int(s.get("base_replicas", 1)) for a, s in self.services.items()}

        self.probes_pending = []
        self.probed_targets = {}

        self.svc_mode_score = {a: defaultdict(float) for a in self.aliases}
        self.dep_hint = {}    # svc -> mode -> {source_hint: weight}

        self.t = -1

    def _cd_ok(self, atype, target, now, slack=1):
        cd = ACTION_COOLDOWN.get(atype, 4)
        key = f"{atype}:{target}"
        last = self.last_used_step.get(key, -10**9)
        return now > last + cd + slack - 1

    def _record_action(self, atype, target, now):
        self.last_used_step[f"{atype}:{target}"] = now
        self.last_action_only[atype] = now

    def _update_history(self, observation):
        for h in observation.get("health", []):
            svc = h.get("service")
            if svc not in self.history:
                continue
            self.history[svc].append({
                "L": _safe_int(h.get("latency_bucket")),
                "E": _safe_int(h.get("error_bucket")),
                "S": _safe_int(h.get("saturation_bucket")),
                "Q": _safe_int(h.get("queue_bucket")),
                "stale": _safe_int(h.get("stale_steps")),
                "missing": tuple(h.get("missing_fields") or []),
            })

    def _service_pressure(self, svc):
        h = self.history.get(svc)
        if not h:
            return 0.0
        n = min(3, len(h))
        last = list(h)[-n:]
        return sum(x["L"] + x["E"] + x["S"] + x["Q"] for x in last) / max(1, n)

    def _delta(self, svc):
        h = list(self.history.get(svc, []))
        if len(h) < 2:
            return 0.0
        cur = h[-1]["L"] + h[-1]["E"] + h[-1]["S"] + h[-1]["Q"]
        prev = max(x["L"] + x["E"] + x["S"] + x["Q"] for x in h[:-1])
        return cur - prev

    def _ingest_probes(self, observation):
        for p in observation.get("probe_results", []):
            target = p.get("target", "")
            hint = p.get("action_hint", "investigate")
            mode = ACTION_HINT_MODE.get(hint)
            confb = _safe_int(p.get("confidence_bucket", 0))
            pressureb = _safe_int(p.get("pressure_bucket", 0))
            source = p.get("source_hint", target)
            noisy = bool(p.get("noisy"))
            w = (confb + 1) * (pressureb + 1) * (0.4 if noisy else 1.0)
            # source_hint = mode.dependency (or service if no dep).
            # Probe target may be mode.service, mode.dependency, or a neighbor.
            # We score both — but bias toward target since it carries the most direct evidence.
            if mode and source in self.svc_mode_score:
                self.svc_mode_score[source][mode] += w
            if mode and target in self.svc_mode_score:
                self.svc_mode_score[target][mode] += w * 0.5
            # Cross-link: probe identifies (target, mode) — source is the probable dep.
            if mode and source and target and source != target:
                self.dep_hint.setdefault(target, {}).setdefault(mode, {})
                self.dep_hint[target][mode][source] = self.dep_hint[target][mode].get(source, 0) + w
            # Channel→mode learning: events.service == mode.service (deterministic in checker).
            # We need to identify which service is the "mode.service".  When probe says
            # (mode m, source_hint=Y), the actual mode.service is *some* service whose anomaly
            # was caused by Y.  Without ground-truth tying, we use the probe's *target* as a hint
            # since target is usually the mode.service when mode_target_relation is highest.
            if mode and target:
                for (ch, esvc), times in self.events_seen.items():
                    if ch == "ch_bg" or ch in self.channel_mode:
                        continue
                    if esvc == target and len(times) >= 1:
                        self.channel_mode[ch] = mode

    def _ingest_events(self, observation):
        for ev in observation.get("events", []):
            ch = ev.get("channel")
            svc = ev.get("service")
            if not ch or ch == "ch_bg":
                continue
            self.events_seen[(ch, svc)].append(self.t)
            self.channel_count[ch] += 1
            # If we already know this channel maps to a mode, attribute to svc.
            mode = self.channel_mode.get(ch)
            if mode and svc in self.svc_mode_score:
                self.svc_mode_score[svc][mode] += 1.4
            # Repeated events on the same (ch, svc): the svc is likely mode.service for ch.
            # Constrain the search to LEGAL_MODE_SERVICE.
            if not mode and len(self.events_seen[(ch, svc)]) >= 3:
                # Find candidate modes whose LEGAL_MODE_SERVICE includes svc.
                candidates = [m for m, legal in LEGAL_MODE_SERVICE.items() if svc in legal]
                # Pick the one with highest probe-score on svc.
                scores = self.svc_mode_score.get(svc, {})
                best = None
                best_score = 0
                for cm in candidates:
                    s = scores.get(cm, 0)
                    if s > best_score:
                        best, best_score = cm, s
                if best and best_score > 1.0:
                    self.channel_mode[ch] = best

    def _channel_mode_guess(self, ch):
        if ch in self.channel_mode:
            return self.channel_mode[ch]
        tally = defaultdict(float)
        for (c, svc), times in self.events_seen.items():
            if c != ch:
                continue
            scores = self.svc_mode_score.get(svc, {})
            for m, s in scores.items():
                tally[m] += s
        if not tally:
            return None
        best = max(tally.items(), key=lambda kv: kv[1])
        if best[1] > 1.0:
            self.channel_mode[ch] = best[0]
            return best[0]
        return None

    def _hypothesize(self):
        hyps = []
        for svc, scores in self.svc_mode_score.items():
            for mode, sc in scores.items():
                if sc <= 0:
                    continue
                # Filter out impossible (mode, svc) combinations from public-data prior.
                if svc not in LEGAL_MODE_SERVICE.get(mode, set()):
                    continue
                dep = self._dep_for(mode, svc)
                hyps.append((sc, mode, svc, dep))
        # consumer_lag: anchor on svc_09 — relatively reliable since deps are constant.
        p09 = self._service_pressure("svc_09")
        h09 = self.history.get("svc_09")
        if h09:
            qmax = max(x["Q"] for x in h09)
            if qmax >= 4 or p09 >= 7:
                hyps.append(((qmax + p09) * 0.85, "consumer_lag", "svc_09", "svc_08"))
        # cache_collapse: svc_07 anomaly anchor (svc_07 is dep in 100% of cache_collapse cases).
        p07 = self._service_pressure("svc_07")
        if p07 >= 7:
            best_caller, best_v = None, 0
            for c in ("svc_00", "svc_01", "svc_02"):
                v = self._service_pressure(c)
                if v > best_v:
                    best_caller, best_v = c, v
            # Only fire if caller is also somewhat anomalous.
            if best_v >= 5:
                hyps.append((p07 * 0.85, "cache_collapse", best_caller or "svc_00", "svc_07"))
        # storage_wait: store anomaly + caller anomaly.
        for store in ("svc_05", "svc_06"):
            ps = self._service_pressure(store)
            if ps >= 7:
                best_caller, best_v = None, 0
                for c in ("svc_03", "svc_04", "svc_02"):
                    if store in self.neighbors.get(c, ()):
                        v = self._service_pressure(c)
                        if v >= 5.5 and v > best_v:
                            best_caller, best_v = c, v
                if best_caller is not None:
                    hyps.append((ps * 0.85, "storage_wait", best_caller, store))
        # traffic_surge: traffic_bucket >=4 AND svc_00 itself anomalous (causal anchor).
        if self._traffic_bucket_now() >= 4 and self._service_pressure("svc_00") >= 6:
            hyps.append((5.0, "traffic_surge", "svc_00", "svc_00"))
        return sorted(hyps, key=lambda x: x[0], reverse=True)

    def _dep_for(self, mode, svc):
        # If we have a probe-derived dep hint for this (svc, mode), prefer it.
        hints = (self.dep_hint.get(svc) or {}).get(mode) or {}
        if hints:
            best = max(hints.items(), key=lambda kv: kv[1])
            if best[1] > 0:
                return best[0]
        if mode == "cache_collapse":
            return "svc_07"
        if mode == "consumer_lag":
            return "svc_08"
        if mode == "storage_wait":
            for n in self.neighbors.get(svc, ()):
                if self.services.get(n, {}).get("role") == "store":
                    return n
            return "svc_05"
        if mode == "dependency_timeout":
            for n in self.neighbors.get(svc, ()):
                role = self.services.get(n, {}).get("role")
                if role in ("store", "cache", "api"):
                    return n
            return svc
        return svc

    def _traffic_bucket_now(self):
        return getattr(self, "_last_traffic_bucket", 0)

    def _propose_for_mode(self, mode, svc, dep, t):
        out = []
        m_svc = svc if svc in self.services else "svc_00"
        d_svc = dep if dep in self.services else m_svc
        if mode == "release_instability":
            out.append({"type": "ROLLBACK_RELEASE", "target": m_svc})
            out.append({"type": "CIRCUIT_BREAK", "target": "svc_00"})
        elif mode == "config_shift":
            out.append({"type": "ROLLBACK_CONFIG", "target": m_svc})
            # SET_TIMEOUT on m_svc; if not strongly identified, also fall back to svc_00.
            out.append({"type": "SET_TIMEOUT", "target": m_svc, "value": 750})
        elif mode == "heap_growth":
            out.append({"type": "RESTART", "target": m_svc})
            base = self.services.get(m_svc, {}).get("base_replicas", 1)
            mx = self.services.get(m_svc, {}).get("max_replicas", base)
            out.append({"type": "SCALE", "target": m_svc, "value": min(int(mx), int(base) + 3)})
        elif mode == "traffic_surge":
            for tgt in ("svc_00", m_svc):
                base = self.services.get(tgt, {}).get("base_replicas", 1)
                mx = self.services.get(tgt, {}).get("max_replicas", base)
                bonus = 4 if tgt == "svc_00" else 3
                out.append({"type": "SCALE", "target": tgt, "value": min(int(mx), int(base) + bonus)})
            out.append({"type": "THROTTLE", "target": "svc_00", "value": 12})
        elif mode == "storage_wait":
            out.append({"type": "REBUILD_INDEX", "target": d_svc})
            out.append({"type": "SET_TIMEOUT", "target": m_svc, "value": 1000})
        elif mode == "consumer_lag":
            out.append({"type": "DRAIN_QUEUE", "target": "svc_08"})
            base = self.services.get("svc_09", {}).get("base_replicas", 1)
            mx = self.services.get("svc_09", {}).get("max_replicas", base)
            out.append({"type": "SCALE", "target": "svc_09", "value": min(int(mx), int(base) + 4)})
        elif mode == "cache_collapse":
            out.append({"type": "WARM_CACHE", "target": "svc_07"})
            out.append({"type": "THROTTLE", "target": "svc_00", "value": 12})
        elif mode == "dependency_timeout":
            out.append({"type": "CIRCUIT_BREAK", "target": m_svc})
            out.append({"type": "SET_TIMEOUT", "target": d_svc, "value": 1000})
        return out

    def _is_useful_now(self, action, mode, svc, dep):
        atype = action.get("type")
        if atype in HELPFUL.get(mode, set()):
            tgt = action.get("target")
            if tgt in {svc, dep, "svc_00", "svc_07", "svc_08", "svc_09"}:
                return True
        return False

    def _action_signature(self, mode, svc, dep, action):
        return (mode, svc, dep, action.get("type"), action.get("target"))

    def _maybe_probe(self, t):
        candidates = []
        for svc in self.aliases:
            press = self._service_pressure(svc)
            scores = self.svc_mode_score.get(svc, {})
            known = max(scores.values()) if scores else 0
            need = press - known * 0.4
            last = self.probed_targets.get(svc, -10)
            if t - last < 4:
                continue
            if press >= 3.0 and need > 1.2:
                candidates.append((need, svc))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        target = candidates[0][1]
        if not self._cd_ok("DIAGNOSTIC_PROBE", target, t):
            return None
        return {"type": "DIAGNOSTIC_PROBE", "target": target}

    def act(self, observation):
        self.t = _safe_int(observation.get("time", 0))
        self._last_traffic_bucket = _safe_int(observation.get("traffic_bucket", 0))
        self._update_history(observation)
        self._ingest_probes(observation)
        self._ingest_events(observation)

        t = self.t
        recent_actions = observation.get("recent_actions", [])

        gh = _safe_int(observation.get("global_health_bucket", 0))
        any_high = any(self._service_pressure(s) >= 6 for s in self.aliases)

        # Track gh transitions to detect "incident just emerged".
        prev_gh = getattr(self, "_prev_gh", 0)
        gh_jumped = gh >= 2 and gh > prev_gh
        self._prev_gh = gh

        # Pre-probe: probe FIRST when situation is bad and probe info is weak.
        hyps = self._hypothesize()
        max_probe_score = max(
            (max(scores.values()) if scores else 0)
            for scores in self.svc_mode_score.values()
        ) if self.svc_mode_score else 0
        # Probe opportunistically on incident emergence.
        if gh_jumped or ((gh >= 2 or any_high) and max_probe_score < 5.0):
            probe = self._maybe_probe(t)
            if probe is not None:
                self._record_action("DIAGNOSTIC_PROBE", probe["target"], t)
                self.probed_targets[probe["target"]] = t
                return probe

        for sc, mode, svc, dep in hyps:
            if sc < 4.2:
                continue
            actions = self._propose_for_mode(mode, svc, dep, t)
            for idx, a in enumerate(actions):
                if not self._is_useful_now(a, mode, svc, dep):
                    continue
                atype = a["type"]
                tgt = a.get("target", svc)
                sig = self._action_signature(mode, svc, dep, a)
                # Higher cap for cheap maintenance actions which keep helping over time.
                if atype in ("ROLLBACK_RELEASE", "ROLLBACK_CONFIG", "REBUILD_INDEX"):
                    limit = 1
                elif atype in ("RESTART", "WARM_CACHE", "DRAIN_QUEUE"):
                    limit = 2
                else:
                    limit = 3
                if self.dispatched[sig] >= limit:
                    continue
                if not self._cd_ok(atype, tgt, t):
                    continue
                if atype == "SCALE":
                    cur = self.replicas.get(tgt, 1)
                    if cur >= a["value"]:
                        continue
                # Secondary actions (idx > 0): need higher score to fire,
                # OR primary action must already have been dispatched for this hypothesis.
                if idx > 0:
                    primary = actions[0]
                    primary_sig = self._action_signature(mode, svc, dep, primary)
                    if self.dispatched[primary_sig] == 0 and sc < 5.0:
                        continue
                if atype == "SET_TIMEOUT":
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    primary = actions[0]
                    primary_sig = self._action_signature(mode, svc, dep, primary)
                    if self.dispatched[primary_sig] == 0 and prob < 6:
                        continue
                # WARM_CACHE: only fire when svc_07 itself shows actual anomaly
                # (avoids assumption-driven wrong calls in non-cache_collapse tasks).
                if atype == "WARM_CACHE":
                    p07 = self._service_pressure("svc_07")
                    prob_cc = self.svc_mode_score.get(svc, {}).get("cache_collapse", 0)
                    if p07 < 6.5 and prob_cc < 6:
                        continue
                # DRAIN_QUEUE: only fire when svc_08/svc_09 shows actual queue anomaly.
                if atype == "DRAIN_QUEUE":
                    h09 = list(self.history.get("svc_09", []))
                    h08 = list(self.history.get("svc_08", []))
                    q09 = max((x["Q"] for x in h09[-3:]), default=0)
                    q08 = max((x["Q"] for x in h08[-3:]), default=0)
                    prob_cl = self.svc_mode_score.get(svc, {}).get("consumer_lag", 0)
                    if max(q09, q08) < 4 and prob_cl < 3:
                        continue
                # THROTTLE: helpful in traffic_surge & cache_collapse only — needs traffic
                # bucket high or cache pressure obvious.
                if atype == "THROTTLE":
                    p07 = self._service_pressure("svc_07")
                    if self._last_traffic_bucket < 4 and p07 < 6.5 and sc < 4:
                        continue
                # Risk gate: high-cost ops only when we have strong probe evidence
                # (or strong symptom evidence for inherently safe mappings).
                if atype == "REBUILD_INDEX":
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    dep_p = self._service_pressure(dep)
                    if prob < 5 and dep_p < 8:
                        continue
                if atype == "ROLLBACK_RELEASE":
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    if prob < 4:
                        continue
                if atype == "ROLLBACK_CONFIG":
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    if prob < 4:
                        continue
                if atype == "RESTART":
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    if prob < 4 and self._service_pressure(svc) < 7.5:
                        continue
                self._record_action(atype, tgt, t)
                self.dispatched[sig] += 1
                if atype == "SCALE":
                    self.replicas[tgt] = a["value"]
                return a

        # If still no action and gh high, last-resort probe.
        if gh >= 3:
            probe = self._maybe_probe(t)
            if probe is not None:
                self._record_action("DIAGNOSTIC_PROBE", probe["target"], t)
                self.probed_targets[probe["target"]] = t
                return probe

        # Pre-emptive scale: if traffic forecast will surge in the next few minutes,
        # scale svc_00 ahead of time. SCALE svc_00 is helpful in traffic_surge & some heap_growth.
        if self.traffic and t < self.horizon - 1:
            future_max = max((self.traffic[k] for k in range(t+1, min(t+5, self.horizon))), default=1.0)
            if future_max >= 1.25:
                tgt = "svc_00"
                base = self.services.get(tgt, {}).get("base_replicas", 1)
                mx = self.services.get(tgt, {}).get("max_replicas", base)
                want = min(int(mx), int(base) + 3)
                if self.replicas.get(tgt, base) < want and self._cd_ok("SCALE", tgt, t):
                    self._record_action("SCALE", tgt, t)
                    self.replicas[tgt] = want
                    return {"type": "SCALE", "target": tgt, "value": want}

        if self._last_traffic_bucket >= 4:
            tgt = "svc_00"
            base = self.services.get(tgt, {}).get("base_replicas", 1)
            mx = self.services.get(tgt, {}).get("max_replicas", base)
            want = min(int(mx), int(base) + 3)
            if self.replicas.get(tgt, base) < want and self._cd_ok("SCALE", tgt, t):
                self._record_action("SCALE", tgt, t)
                self.replicas[tgt] = want
                return {"type": "SCALE", "target": tgt, "value": want}

        return {"type": "NOOP"}
