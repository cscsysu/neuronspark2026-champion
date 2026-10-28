class Agent:
    def reset(self, task_info):
        self.task_info = task_info or {}
        self.services = {s.get("alias"): s for s in self.task_info.get("services", []) if s.get("alias")}
        self.roles = {a: s.get("role", "") for a, s in self.services.items()}
        self.base = {a: int(s.get("base_replicas", 1)) for a, s in self.services.items()}
        self.maxrep = {a: int(s.get("max_replicas", self.base.get(a, 1))) for a, s in self.services.items()}
        self.neighbors = {a: set() for a in self.services}
        for e in self.task_info.get("topology", []):
            u, v = e.get("from"), e.get("to")
            if u in self.neighbors and v in self.neighbors:
                self.neighbors[u].add(v)
                self.neighbors[v].add(u)
        self.used = set()
        self.pending_keys = set()
        self.pending_tag = {}
        self.cool = {}
        self.pending = []
        self.probed_at = {}
        self.last_generic_probe = -99
        self.channel_hint = {}
        self.channel_source = {}
        self.recent_events = []
        self.event_counts = {}
        self.event_actioned = set()
        self.event_timeouted = set()
        self.last_health = {}
        self.bad_seen = {}
        self.probe_targets = set()
        self.last_t = -1

    def _ok_target(self, target):
        if target in self.services:
            return target
        return "svc_00" if "svc_00" in self.services else next(iter(self.services), "svc_00")

    def _scale_value(self, target, extra):
        target = self._ok_target(target)
        return min(self.maxrep.get(target, 1), self.base.get(target, 1) + extra)

    def _cool_ok(self, action, t):
        typ = action.get("type", "NOOP")
        target = action.get("target", "")
        return t >= self.cool.get((typ, target), -1)

    def _mark(self, action, t):
        typ = action.get("type", "NOOP")
        target = action.get("target", "")
        cd = {
            "SCALE": 4, "ROLLBACK_RELEASE": 12, "ROLLBACK_CONFIG": 10,
            "RESTART": 8, "CIRCUIT_BREAK": 6, "SET_TIMEOUT": 5,
            "WARM_CACHE": 5, "REBUILD_INDEX": 16, "DRAIN_QUEUE": 7,
            "THROTTLE": 6, "DIAGNOSTIC_PROBE": 5,
        }.get(typ, 0)
        if typ != "NOOP":
            self.cool[(typ, target)] = t + cd

    def _make(self, typ, target=None, value=None):
        out = {"type": typ}
        if target is not None:
            out["target"] = self._ok_target(target)
        if value is not None:
            out["value"] = int(value)
        return out

    def _unique_key(self, action, tag=None):
        typ = action.get("type")
        target = action.get("target", "")
        value = action.get("value", "")
        if typ in {"SET_TIMEOUT", "THROTTLE"}:
            value = ""
        return (typ, target, value)

    def _enqueue(self, actions, tag):
        high_priority = str(tag).startswith(("probe:", "event:", "unresolved:"))
        insert_at = None
        if high_priority:
            insert_at = len(self.pending)
            for idx, (old_key, _old_action) in enumerate(self.pending):
                if str(self.pending_tag.get(old_key, "")).startswith("fallback:"):
                    insert_at = idx
                    break
        added = 0
        for a in actions:
            if not a or a.get("type") == "NOOP":
                continue
            key = self._unique_key(a, tag)
            if key not in self.used and key not in self.pending_keys:
                item = (key, a)
                if insert_at is None:
                    self.pending.append(item)
                else:
                    self.pending.insert(insert_at + added, item)
                    added += 1
                self.pending_keys.add(key)
                self.pending_tag[key] = tag

    def _service_score(self, h):
        return (
            1.20 * h.get("error_bucket", 0)
            + 1.00 * h.get("latency_bucket", 0)
            + 0.85 * h.get("saturation_bucket", 0)
            + 0.95 * h.get("queue_bucket", 0)
        )

    def _refresh_health(self, observation):
        for h in observation.get("health", []) or []:
            svc = h.get("service")
            if svc in self.services:
                prev = self.last_health.get(svc, {})
                merged = dict(prev)
                for k in ("latency_bucket", "error_bucket", "saturation_bucket", "queue_bucket"):
                    if k in h:
                        merged[k] = h.get(k, 0)
                merged["service"] = svc
                merged["stale_steps"] = h.get("stale_steps", 0)
                self.last_health[svc] = merged
                if self._service_score(merged) >= 7.0 or h.get("queue_bucket", 0) >= 3:
                    self.bad_seen.setdefault(svc, int(observation.get("time", 0)))

    def _worst_services(self, n=4):
        rows = sorted(
            self.last_health.values(),
            key=lambda h: (self._service_score(h), h.get("queue_bucket", 0), h.get("error_bucket", 0)),
            reverse=True,
        )
        return [r.get("service") for r in rows[:n] if r.get("service") in self.services]

    def _worst_neighbor(self, target, role=None):
        candidates = list(self.neighbors.get(target, ()))
        if role:
            candidates = [c for c in candidates if self.roles.get(c) == role]
        if not candidates:
            candidates = [a for a, r in self.roles.items() if (not role or r == role)]
        if not candidates:
            return self._ok_target(target)
        return max(candidates, key=lambda s: self._service_score(self.last_health.get(s, {"service": s})))

    def _storage_for_service(self, svc):
        stores = [n for n in self.neighbors.get(svc, ()) if self.roles.get(n) == "store"]
        if not stores:
            stores = [a for a, r in self.roles.items() if r == "store"]
        if not stores:
            return svc
        return max(stores, key=lambda s: self._service_score(self.last_health.get(s, {"service": s})))

    def _hint_actions(self, hint, source, target, t):
        source = self._ok_target(source or target)
        target = self._ok_target(target or source)
        if hint == "rollback_release":
            return [self._make("ROLLBACK_RELEASE", source), self._make("CIRCUIT_BREAK", "svc_00")]
        if hint == "rollback_config":
            return [self._make("ROLLBACK_CONFIG", source), self._make("SET_TIMEOUT", source, 750)]
        if hint == "restart_scale":
            return [
                self._make("RESTART", source),
                self._make("SCALE", source, self._scale_value(source, 3)),
            ]
        if hint == "scale_throttle":
            actions = [self._make("SCALE", "svc_00", self._scale_value("svc_00", 4))]
            if source != "svc_00":
                actions.append(self._make("SCALE", source, self._scale_value(source, 3)))
            actions.append(self._make("THROTTLE", "svc_00", 10))
            return actions
        if hint == "storage_index":
            dep = source if self.roles.get(source) == "store" else self._storage_for_service(target)
            if target in self.neighbors.get(dep, set()) and self.roles.get(target) != "store":
                svc = target
            else:
                api_neighbors = [n for n in self.neighbors.get(dep, set()) if self.roles.get(n) in {"api", "edge"}]
                if api_neighbors:
                    svc = max(api_neighbors, key=lambda s: self._service_score(self.last_health.get(s, {"service": s})))
                else:
                    svc = self._worst_neighbor(dep)
            return [self._make("REBUILD_INDEX", dep), self._make("SET_TIMEOUT", svc, 1000)]
        if hint == "queue_drain":
            return [
                self._make("DRAIN_QUEUE", "svc_08"),
                self._make("SCALE", "svc_09", self._scale_value("svc_09", 4)),
            ]
        if hint == "cache_warm":
            return [self._make("WARM_CACHE", "svc_07"), self._make("THROTTLE", "svc_00", 12)]
        if hint == "timeout_circuit":
            svc = target
            if self.roles.get(target) in {"store", "cache", "queue"}:
                svc = "svc_00"
            elif source != target and source in self.services and target not in self.neighbors.get(source, set()):
                svc = "svc_00"
            return [self._make("CIRCUIT_BREAK", svc), self._make("SET_TIMEOUT", source, 1000)]
        return []

    def _probe(self, target, t):
        target = self._ok_target(target)
        if t - self.probed_at.get(target, -99) < 5:
            return None
        a = self._make("DIAGNOSTIC_PROBE", target)
        self.probed_at[target] = t
        return a

    def _pop_pending(self, t):
        keep = []
        chosen = None
        for key, action in self.pending:
            self.pending_keys.discard(key)
            if key in self.used:
                continue
            if chosen is None and self._cool_ok(action, t):
                chosen = (key, action)
            else:
                keep.append((key, action))
        self.pending = keep
        self.pending_keys = {key for key, _ in keep}
        if chosen:
            self.used.add(chosen[0])
            self._mark(chosen[1], t)
            return chosen[1]
        return None

    def _fallback_actions(self, observation, t):
        gb = int(observation.get("global_health_bucket", 0) or 0)
        tb = int(observation.get("traffic_bucket", 0) or 0)
        worst = self._worst_services(3)
        worst0 = worst[0] if worst else "svc_00"
        h08 = self.last_health.get("svc_08", {})
        h09 = self.last_health.get("svc_09", {})
        h07 = self.last_health.get("svc_07", {})
        if (h08.get("queue_bucket", 0) >= 4 or h09.get("queue_bucket", 0) >= 4) and gb >= 2:
            return [self._make("DRAIN_QUEUE", "svc_08"), self._make("SCALE", "svc_09", self._scale_value("svc_09", 4))]
        if tb >= 4 and gb >= 3:
            acts = [self._make("SCALE", "svc_00", self._scale_value("svc_00", 4))]
            if worst0 != "svc_00" and self.roles.get(worst0) in {"edge", "api"}:
                acts.append(self._make("SCALE", worst0, self._scale_value(worst0, 3)))
            return acts
        if h07 and self._service_score(h07) >= 8.5 and gb >= 3:
            return [self._make("WARM_CACHE", "svc_07")]
        if gb >= 5:
            return [self._make("SCALE", "svc_00", self._scale_value("svc_00", 4))]
        return []

    def act(self, observation):
        t = int(observation.get("time", self.last_t + 1))
        self.last_t = t
        self._refresh_health(observation)

        real_events = []
        for e in observation.get("events", []) or []:
            ch = e.get("channel", "")
            if ch == "ch_bg" or e.get("token") == "tok_bg":
                continue
            svc = self._ok_target(e.get("service"))
            age = int(e.get("age", 0) or 0)
            real_events.append((ch, svc, age))
            if ch:
                self.channel_source[ch] = svc
                self.recent_events.append((t, ch, svc))
                key = (ch, svc)
                prev = self.event_counts.get(key, (0, t))
                self.event_counts[key] = (prev[0] + 1, min(prev[1], t))
        self.recent_events = [row for row in self.recent_events if t - row[0] <= 8]

        for p in observation.get("probe_results", []) or []:
            hint = p.get("action_hint", "")
            pressure = int(p.get("pressure_bucket", 0) or 0)
            conf = int(p.get("confidence_bucket", 0) or 0)
            target = self._ok_target(p.get("target"))
            source = self._ok_target(p.get("source_hint", target))
            if pressure >= 1 and conf >= 3 and hint and hint != "investigate":
                self._enqueue(self._hint_actions(hint, source, target, t), "probe:" + hint + ":" + source + ":" + target)
                for _, ch, svc in reversed(self.recent_events):
                    if svc in {target, source}:
                        duplicate_hint = False
                        for old_ch, old_hint in self.channel_hint.items():
                            if old_ch != ch and old_hint == hint:
                                duplicate_hint = True
                                break
                        if duplicate_hint:
                            continue
                        self.channel_hint.setdefault(ch, hint)

        low_pending = (not self.pending) or all(
            str(self.pending_tag.get(key, "")).startswith("fallback:") for key, _ in self.pending
        )
        gb_now = int(observation.get("global_health_bucket", 0) or 0)
        if low_pending and gb_now >= 1:
            for ch, svc, _age in real_events:
                key = (ch, svc)
                if ch not in self.channel_hint and key not in self.event_actioned:
                    pr = self._probe(svc, t)
                    if pr is not None:
                        self._mark(pr, t)
                        return pr

        action = self._pop_pending(t)
        if action is not None:
            return action

        for ch, svc, _age in real_events:
            if ch in self.channel_hint:
                hint = self.channel_hint[ch]
                self._enqueue(self._hint_actions(hint, svc, svc, t), "event:" + ch + ":" + svc)

        action = self._pop_pending(t)
        if action is not None:
            return action

        for ch, svc, age in real_events:
            key = (ch, svc)
            if ch in self.channel_hint or key in self.event_actioned:
                continue
            count, first_t = self.event_counts.get(key, (1, t))
            score = self._service_score(self.last_health.get(svc, {"service": svc}))
            actions = []
            final_action = False
            if svc == "svc_09" and (count >= 1 or age >= 3):
                if any(h == "queue_drain" for h in self.channel_hint.values()) or ("DRAIN_QUEUE", "svc_08", "") in self.used:
                    actions = [self._make("ROLLBACK_RELEASE", "svc_09"), self._make("CIRCUIT_BREAK", "svc_00")]
                    final_action = True
            elif self.roles.get(svc) in {"api", "edge"} and count >= 2 and score >= 11.0 and t - first_t >= 2:
                if key not in self.event_timeouted:
                    actions = [self._make("SET_TIMEOUT", svc, 1000)]
                    self.event_timeouted.add(key)
                elif count >= 3 and t - first_t >= 5:
                    actions = [self._make("ROLLBACK_CONFIG", svc)]
                    final_action = True
            if actions:
                if final_action:
                    self.event_actioned.add(key)
                self._enqueue(actions, "unresolved:" + ch + ":" + svc)
                action = self._pop_pending(t)
                if action is not None:
                    return action

        gb = int(observation.get("global_health_bucket", 0) or 0)
        actions = self._fallback_actions(observation, t)
        if actions:
            self._enqueue(actions, "fallback:" + str(t))
            action = self._pop_pending(t)
            if action is not None:
                return action

        if gb >= 1:
            for ch, svc, _age in real_events:
                if ch not in self.channel_hint:
                    pr = self._probe(svc, t)
                    if pr is not None:
                        self._mark(pr, t)
                        return pr

        if gb >= 1 and (not self.channel_hint or gb >= 4):
            targets = []
            targets.extend(self._worst_services(4))
            for special in ("svc_07", "svc_08", "svc_09", "svc_00"):
                if special in self.services and special not in targets:
                    targets.append(special)
            for target in targets:
                pr = self._probe(target, t)
                if pr is not None:
                    self._mark(pr, t)
                    return pr

        return {"type": "NOOP"}
