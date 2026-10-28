# NS-2026-08 agent v4: probe-first then helpful-multi-mode actions.
# CPU-only, stdlib only.
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
ROLE = {
    "svc_00": "edge", "svc_01": "api", "svc_02": "api", "svc_03": "api",
    "svc_04": "api", "svc_05": "store", "svc_06": "store",
    "svc_07": "cache", "svc_08": "queue", "svc_09": "worker",
}


def _i(x, d=0):
    try:
        return int(x)
    except Exception:
        return d


class Agent:
    def reset(self, task_info):
        self.task_info = task_info or {}
        self.services = {s["alias"]: s for s in self.task_info.get("services", [])}
        self.aliases = list(self.services.keys())
        self.topology = self.task_info.get("topology", []) or []
        self.horizon = int(self.task_info.get("horizon", 48))
        self.neigh = {a: set() for a in self.aliases}
        for e in self.topology:
            f, t = e.get("from"), e.get("to")
            if f in self.neigh and t:
                self.neigh[f].add(t)
            if t in self.neigh and f:
                self.neigh[t].add(f)
        self.last_used = {}
        self.bad_hist = {a: deque(maxlen=10) for a in self.aliases}
        self.svc_mode = {a: defaultdict(float) for a in self.aliases}
        self.dep_hint = {a: defaultdict(float) for a in self.aliases}  # mode -> dep service
        self.probed_at = {a: -100 for a in self.aliases}
        self.replicas = {a: int(s.get("base_replicas", 1)) for a, s in self.services.items()}
        self.dispatched = defaultdict(int)
        self.t = -1
        self.last_traffic_b = 0
        self.last_global_b = 0

    def _cd_ok(self, atype, target, now):
        cd = ACTION_COOLDOWN.get(atype, 4)
        last = self.last_used.get((atype, target), -10**9)
        return now >= last + cd

    def _record(self, atype, target, now):
        self.last_used[(atype, target)] = now

    def _bad(self, h):
        return _i(h.get("latency_bucket")) + _i(h.get("error_bucket")) + _i(h.get("saturation_bucket")) + _i(h.get("queue_bucket"))

    def _update_history(self, observation):
        for h in observation.get("health", []):
            svc = h.get("service")
            if svc not in self.bad_hist:
                continue
            self.bad_hist[svc].append(self._bad(h))

    def _smooth(self, svc, n=3):
        h = list(self.bad_hist.get(svc, []))
        if not h:
            return 0.0
        last = h[-min(n, len(h)):]
        return sum(last) / len(last)

    def _anom(self, svc, thr=7.5):
        return self._smooth(svc, 3) >= thr

    def _ingest_probes(self, observation):
        for p in observation.get("probe_results", []):
            target = p.get("target", "")
            hint = p.get("action_hint", "investigate")
            mode = ACTION_HINT_MODE.get(hint)
            confb = _i(p.get("confidence_bucket", 0))
            pressb = _i(p.get("pressure_bucket", 0))
            source = p.get("source_hint", target)
            noisy = bool(p.get("noisy"))
            if not mode:
                continue
            w = (confb + 1) * (pressb + 1)
            if noisy:
                w *= 0.4
            # Source service is the *cause* (mode.service or its dep).
            if source in self.svc_mode:
                self.svc_mode[source][mode] += w
            # Probe target may be the affected service.
            if target in self.svc_mode and target != source:
                self.svc_mode[target][mode] += w * 0.4
                # Hint that source might be the dep
                self.dep_hint[target][source] += w

    # ---------- Hypothesis ranking ----------
    def _hypothesize(self):
        hyps = []
        # Probe-driven: scale up & enumerate all probable (mode, svc, dep) combos.
        for svc, scores in self.svc_mode.items():
            for mode, sc in scores.items():
                if sc < 1.0:
                    continue
                # Determine dep
                dep = self._best_dep(mode, svc)
                hyps.append((sc * 1.5, mode, svc, dep))
        # Symptom-driven: only safe / well-anchored modes.
        # consumer_lag - svc_09 anomaly
        if self._anom("svc_09", 7.0) or self._anom("svc_08", 7.0):
            sc = max(self._smooth("svc_09"), self._smooth("svc_08"))
            hyps.append((sc * 0.9, "consumer_lag", "svc_09", "svc_08"))
        # cache_collapse - svc_07 anomaly
        if self._anom("svc_07", 7.0):
            best_caller, best_v = "svc_00", self._smooth("svc_00")
            for c in ("svc_01", "svc_02"):
                v = self._smooth(c)
                if self._anom(c, 6.0) and v > best_v:
                    best_caller, best_v = c, v
            sc = self._smooth("svc_07")
            hyps.append((sc * 0.9, "cache_collapse", best_caller, "svc_07"))
        # storage_wait
        for store in ("svc_05", "svc_06"):
            if self._anom(store, 7.0):
                best_caller, best_v = None, 0
                for c in ("svc_03", "svc_04", "svc_02"):
                    if store in self.neigh.get(c, ()):
                        v = self._smooth(c)
                        if self._anom(c, 6.0) and v > best_v:
                            best_caller, best_v = c, v
                caller = best_caller or "svc_03"
                sc = self._smooth(store)
                hyps.append((sc * 0.85, "storage_wait", caller, store))
        # traffic_surge
        if self.last_traffic_b >= 4:
            hyps.append((6.0, "traffic_surge", "svc_00", "svc_00"))
        # Merge by (mode, svc, dep)
        merged = {}
        for sc, m, s, d in hyps:
            k = (m, s, d)
            merged[k] = max(merged.get(k, 0), sc)
        out = sorted([(v, k[0], k[1], k[2]) for k, v in merged.items()], key=lambda x: -x[0])
        return out

    def _best_dep(self, mode, svc):
        if mode == "cache_collapse":
            return "svc_07"
        if mode == "consumer_lag":
            return "svc_08"
        if mode == "storage_wait":
            for n in self.neigh.get(svc, ()):
                if ROLE.get(n) == "store":
                    return n
            return "svc_05"
        if mode == "dependency_timeout":
            # Use dep_hint if any
            hints = self.dep_hint.get(svc, {})
            if hints:
                return max(hints.items(), key=lambda kv: kv[1])[0]
            for n in self.neigh.get(svc, ()):
                if ROLE.get(n) in ("store", "cache", "api"):
                    return n
            return svc
        return svc

    def _propose(self, mode, svc, dep):
        out = []
        s = svc if svc in self.services else "svc_00"
        d = dep if dep in self.services else s
        if mode == "release_instability":
            out.append({"type": "ROLLBACK_RELEASE", "target": s})
            out.append({"type": "CIRCUIT_BREAK", "target": s})
        elif mode == "config_shift":
            out.append({"type": "ROLLBACK_CONFIG", "target": s})
            out.append({"type": "SET_TIMEOUT", "target": s, "value": 750})
        elif mode == "heap_growth":
            out.append({"type": "RESTART", "target": s})
            base = self.services.get(s, {}).get("base_replicas", 1)
            mx = self.services.get(s, {}).get("max_replicas", base)
            out.append({"type": "SCALE", "target": s, "value": min(int(mx), int(base) + 3)})
        elif mode == "traffic_surge":
            for tgt in ("svc_00", s):
                base = self.services.get(tgt, {}).get("base_replicas", 1)
                mx = self.services.get(tgt, {}).get("max_replicas", base)
                bonus = 4 if tgt == "svc_00" else 3
                out.append({"type": "SCALE", "target": tgt, "value": min(int(mx), int(base) + bonus)})
            out.append({"type": "THROTTLE", "target": "svc_00", "value": 12})
        elif mode == "storage_wait":
            out.append({"type": "REBUILD_INDEX", "target": d})
            out.append({"type": "SET_TIMEOUT", "target": s, "value": 1000})
        elif mode == "consumer_lag":
            out.append({"type": "DRAIN_QUEUE", "target": "svc_08"})
            base = self.services.get("svc_09", {}).get("base_replicas", 1)
            mx = self.services.get("svc_09", {}).get("max_replicas", base)
            out.append({"type": "SCALE", "target": "svc_09", "value": min(int(mx), int(base) + 4)})
        elif mode == "cache_collapse":
            out.append({"type": "WARM_CACHE", "target": "svc_07"})
            out.append({"type": "THROTTLE", "target": "svc_00", "value": 12})
        elif mode == "dependency_timeout":
            out.append({"type": "CIRCUIT_BREAK", "target": s})
            out.append({"type": "SET_TIMEOUT", "target": d, "value": 1000})
        return out

    def _useful(self, action, mode, svc, dep):
        atype = action.get("type")
        if atype not in HELPFUL.get(mode, set()):
            return False
        tgt = action.get("target")
        return tgt in {svc, dep, "svc_00", "svc_07", "svc_08", "svc_09"}

    # ---------- Diagnostic probe scheduling ----------
    def _maybe_probe(self, t, force=False):
        cands = []
        for a in self.aliases:
            avg = self._smooth(a, 3)
            if avg < 6 and not force:
                continue
            sm = self.svc_mode.get(a, {})
            top = max(sm.values()) if sm else 0
            if top > 7 and not force:
                continue
            if t - self.probed_at.get(a, -100) < 6:
                continue
            cands.append((avg, a))
        if not cands:
            return None
        cands.sort(reverse=True)
        target = cands[0][1]
        if not self._cd_ok("DIAGNOSTIC_PROBE", target, t):
            return None
        return target

    def act(self, observation):
        self.t = _i(observation.get("time", 0))
        t = self.t
        self.last_traffic_b = _i(observation.get("traffic_bucket", 0))
        self.last_global_b = _i(observation.get("global_health_bucket", 0))
        self._update_history(observation)
        self._ingest_probes(observation)

        any_anom = any(self._anom(a, 7.0) for a in self.aliases)
        bad_count = sum(1 for a in self.aliases if self._smooth(a) >= 7.0)

        # Quiet phase: NOOP.
        if self.last_global_b == 0 and self.last_traffic_b <= 2 and not any_anom:
            return {"type": "NOOP"}

        hyps = self._hypothesize()

        # Top action selection (high-conf actions): ordered by score, with risk gate.
        for sc, mode, svc, dep in hyps:
            if sc < 3.0:
                continue
            for a in self._propose(mode, svc, dep):
                if not self._useful(a, mode, svc, dep):
                    continue
                atype = a["type"]
                tgt = a.get("target", svc)
                key = (mode, svc, dep, atype, tgt)
                limit = 1 if atype in ("ROLLBACK_RELEASE", "ROLLBACK_CONFIG", "REBUILD_INDEX", "RESTART") else 2
                if self.dispatched[key] >= limit:
                    continue
                if not self._cd_ok(atype, tgt, t):
                    continue
                if atype == "SCALE":
                    if self.replicas.get(tgt, 1) >= a["value"]:
                        continue
                # Risk gate: high-cost ops need higher confidence.
                if atype == "REBUILD_INDEX" and sc < 5.0:
                    continue
                if atype == "ROLLBACK_RELEASE" and sc < 5.0:
                    continue
                if atype == "ROLLBACK_CONFIG" and sc < 4.5:
                    continue
                if atype == "RESTART" and sc < 4.5:
                    continue
                self._record(atype, tgt, t)
                self.dispatched[key] += 1
                if atype == "SCALE":
                    self.replicas[tgt] = a["value"]
                return a

        # If something is wrong but no firm hypothesis: PROBE.
        top_score = hyps[0][0] if hyps else 0.0
        if (self.last_global_b >= 2 or any_anom) and top_score < 4.0:
            tgt = self._maybe_probe(t)
            if tgt is not None:
                self._record("DIAGNOSTIC_PROBE", tgt, t)
                self.probed_at[tgt] = t
                return {"type": "DIAGNOSTIC_PROBE", "target": tgt}

        # Lower-bar action: try a low-risk action even if score is mid-low.
        for sc, mode, svc, dep in hyps:
            if sc < 1.5:
                continue
            for a in self._propose(mode, svc, dep):
                if not self._useful(a, mode, svc, dep):
                    continue
                atype = a["type"]
                if atype in ("ROLLBACK_RELEASE", "ROLLBACK_CONFIG", "REBUILD_INDEX", "RESTART"):
                    continue   # too risky without strong signal
                tgt = a.get("target", svc)
                key = (mode, svc, dep, atype, tgt)
                if self.dispatched[key] >= 2:
                    continue
                if not self._cd_ok(atype, tgt, t):
                    continue
                if atype == "SCALE":
                    if self.replicas.get(tgt, 1) >= a["value"]:
                        continue
                self._record(atype, tgt, t)
                self.dispatched[key] += 1
                if atype == "SCALE":
                    self.replicas[tgt] = a["value"]
                return a

        # Last-resort probe if global remains bad.
        if self.last_global_b >= 3:
            tgt = self._maybe_probe(t, force=True)
            if tgt is not None:
                self._record("DIAGNOSTIC_PROBE", tgt, t)
                self.probed_at[tgt] = t
                return {"type": "DIAGNOSTIC_PROBE", "target": tgt}

        return {"type": "NOOP"}
