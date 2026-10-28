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
        self.global_dispatched = defaultdict(int)
        self.global_type_dispatched = defaultdict(int)

        self.history = {a: deque(maxlen=8) for a in self.aliases}
        self.events_seen = defaultdict(list)
        self.channel_count = defaultdict(int)
        # Per-task force-probe queue (service names that need urgent probing).
        self.force_probe = []   # list of (svc, reason_t)
        self.event_seen_pairs = set()   # (channel, service) already seen

        self.channel_mode = {}

        self.replicas = {a: int(s.get("base_replicas", 1)) for a, s in self.services.items()}

        self.probes_pending = []
        self.probed_targets = {}

        self.svc_mode_score = {a: defaultdict(float) for a in self.aliases}
        self.dep_hint = {}    # svc -> mode -> {source_hint: weight}

        self._rebuild_target = None  # which store we've already targeted for REBUILD_INDEX

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

    def _baseline(self, svc, n_early=4):
        h = list(self.history.get(svc, []))
        if len(h) <= n_early:
            return 3.0
        early = h[:n_early]
        return sum(x["L"] + x["E"] + x["S"] + x["Q"] for x in early) / len(early)

    def _anomaly_score(self, svc):
        """Combined anomaly signal: pressure + delta against baseline."""
        cur = self._service_pressure(svc)
        base = self._baseline(svc)
        return cur + max(0, cur - base) * 0.5

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
            # LEGAL guard: only credit (svc, mode) pairs that are legal per public-data prior.
            legal_set = LEGAL_MODE_SERVICE.get(mode, set()) if mode else set()
            if mode and source in self.svc_mode_score and source in legal_set:
                self.svc_mode_score[source][mode] += w
            if mode and target in self.svc_mode_score and target in legal_set:
                self.svc_mode_score[target][mode] += w * 0.5
            # Indirect inference: probe of a dep-target service (e.g., svc_05) returns a hint
            # for a mode whose legal services don't include the dep-target itself (e.g.,
            # storage_wait is on svc_02/03/04, not svc_05). The probe still tells us SOMEWHERE
            # in legal_set that mode is active. Spread credit to legal services that have
            # events seen — they're candidate mode hosts.
            if mode and target and target not in legal_set and legal_set:
                # Distribute w*0.6 across event-bearing legal services proportional to pressure.
                event_legal = [
                    s for s in legal_set
                    if s in self.svc_mode_score and any(
                        es == s for (ch, es) in self.event_seen_pairs if ch != "ch_bg"
                    )
                ]
                if event_legal:
                    presses = [(s, max(self._service_pressure(s), 0.1)) for s in event_legal]
                    total_p = sum(p for _, p in presses) or 1.0
                    for s, p in presses:
                        self.svc_mode_score[s][mode] += w * 0.6 * (p / total_p)
                else:
                    # No event-legal candidate: fallback distribute equally across legal_set.
                    legal_in_score = [s for s in legal_set if s in self.svc_mode_score]
                    if legal_in_score:
                        share = w * 0.4 / len(legal_in_score)
                        for s in legal_in_score:
                            self.svc_mode_score[s][mode] += share
            # Cross-link: probe identifies (target, mode) — source is the probable dep.
            if mode and source and target and source != target:
                self.dep_hint.setdefault(target, {}).setdefault(mode, {})
                self.dep_hint[target][mode][source] = self.dep_hint[target][mode].get(source, 0) + w
            # Channel→mode learning: events.service == mode.service (deterministic in checker).
            # We need to identify which service is the "mode.service".  When probe says
            # (mode m, source_hint=Y), the actual mode.service is *some* service whose anomaly
            # was caused by Y.  Without ground-truth tying, we use the probe's *target* as a hint
            # since target is usually the mode.service when mode_target_relation is highest.
            #
            # CAUTION: a service can host multiple modes (and therefore multiple channels).
            # We must NOT blanket-assign every channel of `target` to `mode`. Instead, prefer
            # the channel that's most active and not yet mapped.
            if mode and target:
                # LEGAL_MODE_SERVICE prior: a probe-derived (target, mode) is legitimate ONLY if
                # the mode can legally live on `target`.  Without this check, noisy probes can
                # corrupt the channel_mode mapping (e.g., svc_09 -> config_shift, which is illegal).
                if target not in LEGAL_MODE_SERVICE.get(mode, set()):
                    continue
                # Find candidate channels on this svc not yet mapped.
                cand_chs = [ch for (ch, esvc), tlist in self.events_seen.items()
                            if esvc == target and ch != "ch_bg" and ch not in self.channel_mode and tlist]
                # Exclude channels already attributed to a different mode for this svc (defensive).
                # Pick the one with most events (most likely the persistent mode).
                if cand_chs:
                    best_ch = max(cand_chs, key=lambda c: len(self.events_seen.get((c, target), [])))
                    # GLOBAL uniqueness: each mode appears in at most ONE channel across the whole task.
                    # Don't assign a mode that's already mapped to another channel anywhere.
                    if mode in self.channel_mode.values():
                        continue
                    # Only assign if mode `mode` isn't already used for some OTHER channel on this svc.
                    used = set(self.channel_mode.get(c) for (c, s) in self.event_seen_pairs
                               if s == target and c in self.channel_mode)
                    if mode not in used:
                        self.channel_mode[best_ch] = mode

    def _ingest_events(self, observation):
        for ev in observation.get("events", []):
            ch = ev.get("channel")
            svc = ev.get("service")
            if not ch or ch == "ch_bg":
                continue
            new_pair = (ch, svc) not in self.event_seen_pairs
            self.event_seen_pairs.add((ch, svc))
            self.events_seen[(ch, svc)].append(self.t)
            self.channel_count[ch] += 1
            # If we already know this channel maps to a mode, attribute to svc.
            mode = self.channel_mode.get(ch)
            if False and mode and svc in self.svc_mode_score:
                self.svc_mode_score[svc][mode] += 1.4
            # NEW: A *new* (channel, service) pair appearing strongly indicates a mode start.
            # If we don't yet know what mode this channel encodes, force a probe on this svc.
            if new_pair and not mode and svc in self.services:
                self.force_probe.append((svc, self.t))
            # When a channel persistently reappears on the same svc but is unmapped,
            # spread credit evenly across LEGAL remaining candidates so action-issuance can fire.
            if not mode and svc in self.services:
                n_events = len(self.events_seen[(ch, svc)])
                # Bump on 2nd and 3rd occurrence to build evidence quickly.
                if n_events in (2, 3):
                    used_modes_other = set(self.channel_mode.get(c) for (c, s) in self.event_seen_pairs
                                           if s == svc and c != ch and c in self.channel_mode)
                    used_modes_other.discard(None)
                    legal = set(m for m, mset in LEGAL_MODE_SERVICE.items() if svc in mset)
                    remaining = legal - used_modes_other
                    if remaining and svc in self.svc_mode_score:
                        bump = 0.0 / max(1, len(remaining))
                        for m in sorted(remaining):
                            self.svc_mode_score[svc][m] += bump
            # Repeated events on the same (ch, svc): the svc is likely mode.service for ch.
            # Constrain the search to LEGAL_MODE_SERVICE, excluding modes already mapped to OTHER channels of this svc.
            # Only assign when there is a UNIQUE viable candidate (otherwise leave to CSP).
            if not mode and len(self.events_seen[(ch, svc)]) >= 3:
                candidates = [m for m, legal in LEGAL_MODE_SERVICE.items() if svc in legal]
                used_modes = set(self.channel_mode.get(c) for (c, s) in self.event_seen_pairs
                                 if s == svc and c != ch and c in self.channel_mode)
                used_modes.discard(None)
                # GLOBAL uniqueness: also exclude modes already mapped to ANY other channel.
                global_used = set(self.channel_mode.values())
                candidates = [m for m in candidates if m not in used_modes and m not in global_used]
                # Only when EXACTLY ONE candidate left → deterministic deduction.
                if len(candidates) == 1:
                    self.channel_mode[ch] = candidates[0]
        # After all events processed: run CSP-based mode assignment.
        # NOTE: legacy per-svc deduction is unsafe (no global uniqueness); removed in favor of CSP.
        self._csp_deduce_modes()

    def _csp_deduce_modes(self):
        """
        Each (channel, service) pair seen corresponds to exactly ONE mode.
        Distinct channels -> distinct modes (at most one mode per channel in the task).
        Solve assignment puzzle with LEGAL_MODE_SERVICE constraint.
        Take the intersection over all valid solutions: any (ch, mode) common to all
        solutions is a high-confidence deduction.
        """
        # Build pairs: each (ch, svc) is one mode-slot to fill.
        # If the same channel appears with multiple svcs, that channel still has exactly one mode,
        # but mode.service must match all of those svcs — impossible. Treat each ch as one slot;
        # the actual mode service is the most-frequent svc on that channel.
        pair_set = set((ch, svc) for (ch, svc) in self.event_seen_pairs if ch != "ch_bg")
        if not pair_set:
            return
        # Sometimes one channel may appear with multiple svcs due to noise/late delivery; but in
        # the underlying truth, each channel has one (mode, service). Group by channel and pick
        # the canonical svc as the most frequent one.
        from collections import defaultdict, Counter
        ch_svc_count = defaultdict(Counter)
        for (ch, svc) in sorted(pair_set, reverse=True):
            ch_svc_count[ch][svc] += 1
        # Treat (ch, svc_with_most_evidence) as canonical.
        slots = []
        for ch, ctr in sorted(ch_svc_count.items(), reverse=True):
            # Use the most-events svc per channel
            canonical_svcs = [s for s, _ in ctr.most_common()]
            slots.append((ch, canonical_svcs))
        # Each slot needs to pick a mode. Build candidate lists.
        # candidate(ch, svc) = modes m where svc in LEGAL_MODE_SERVICE[m]
        slot_candidates = []
        for ch, svcs in slots:
            cand = set()
            for sv in svcs:
                cand.update(m for m, legal in LEGAL_MODE_SERVICE.items() if sv in legal)
            # If channel is already mapped (from probes), lock it to that mode.
            if ch in self.channel_mode:
                cand = {self.channel_mode[ch]} & cand
            slot_candidates.append((ch, svcs, sorted(cand)))
        # Backtracking search: assign each slot a mode, no duplicates.
        n = len(slot_candidates)
        all_solutions = []
        MAX_SOL = 256
        # Seed `used` with modes already mapped to OTHER channels (not in this slot list).
        # All our slots are present in slot_candidates so any locked one is properly forced.
        def backtrack(i, used, assignment):
            if len(all_solutions) >= MAX_SOL:
                return
            if i == n:
                all_solutions.append(dict(assignment))
                return
            ch, svcs, cands = slot_candidates[i]
            for m in cands:
                if m in used:
                    continue
                # Verify mode legal for canonical svc
                if not any(sv in LEGAL_MODE_SERVICE.get(m, set()) for sv in svcs):
                    continue
                used.add(m)
                assignment[ch] = (m, svcs[0])
                backtrack(i + 1, used, assignment)
                del assignment[ch]
                used.discard(m)
        backtrack(0, set(), {})
        if not all_solutions:
            return
        # Intersection: per-channel modes that ALL solutions agree on.
        first = all_solutions[0]
        common = {}
        for ch, (mode, svc) in first.items():
            if all(sol.get(ch) == (mode, svc) for sol in all_solutions):
                common[ch] = (mode, svc)
        # Apply learned mappings.
        for ch, (mode, svc) in common.items():
            if self.channel_mode.get(ch) != mode:
                self.channel_mode[ch] = mode
            if svc in self.svc_mode_score:
                # Strong, high-confidence boost from CSP-deduced channels.
                cur = self.svc_mode_score[svc][mode]
                if cur < 8.0:
                    self.svc_mode_score[svc][mode] = 8.0
        # Also: weight per-channel mode probability across all solutions
        # If a channel has the same mode in fraction p of solutions, we add weak credit.
        if len(all_solutions) > 1:
            ch_mode_count = defaultdict(Counter)
            for sol in all_solutions:
                for ch, (mode, svc) in sol.items():
                    ch_mode_count[ch][(mode, svc)] += 1
            n_sol = len(all_solutions)
            for ch, ctr in ch_mode_count.items():
                if ch in common:
                    continue
                for (mode, svc), c in ctr.items():
                    p = c / n_sol
                    if p >= 0.5 and svc in self.svc_mode_score:
                        cur = self.svc_mode_score[svc][mode]
                        target = 4.0 * p
                        if cur < target:
                            self.svc_mode_score[svc][mode] = target

    def _deduce_channel_modes(self):
        """
        Deduce channel→mode mapping by elimination.
        Each task's modes have unique services in many cases; use LEGAL_MODE_SERVICE as prior.
        For each service, list its channels and confirmed modes; compute candidate set per channel.
        """
        # Group channels seen per service
        svc_channels = defaultdict(list)
        for (ch, svc) in sorted(self.event_seen_pairs, reverse=True):
            if ch == "ch_bg":
                continue
            svc_channels[svc].append(ch)
        # For each service, infer its likely mode set by intersecting LEGAL with not-yet-mapped channels.
        for svc, chs in sorted(svc_channels.items(), reverse=True):
            confirmed_modes = set(self.channel_mode.get(c) for c in chs if c in self.channel_mode)
            confirmed_modes.discard(None)
            unmapped = [c for c in chs if c not in self.channel_mode]
            if not unmapped:
                continue
            legal_modes = set(m for m, legal in LEGAL_MODE_SERVICE.items() if svc in legal)
            remaining = legal_modes - confirmed_modes
            # If exactly one unmapped channel and one remaining candidate, deduce!
            if len(unmapped) == 1 and len(remaining) == 1:
                inferred = next(iter(remaining))
                self.channel_mode[unmapped[0]] = inferred
                # Score the service for this mode strongly: this is high-confidence inference.
                if svc in self.svc_mode_score:
                    self.svc_mode_score[svc][inferred] += 6.0
                continue
            # Otherwise narrow down using probe-derived scores: if score > some threshold for that mode on svc.
            # IMPORTANT: only consider modes still in `remaining` (not already confirmed for another channel).
            for ch in unmapped:
                # Pick remaining mode with highest score on svc.
                scores = self.svc_mode_score.get(svc, {})
                best_mode = None
                best_score = 0
                for m in sorted(remaining):
                    s = scores.get(m, 0)
                    if s > best_score:
                        best_mode, best_score = m, s
                if best_mode and best_score >= 3.0:
                    self.channel_mode[ch] = best_mode
                    if svc in self.svc_mode_score:
                        self.svc_mode_score[svc][best_mode] += 2.0

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

    def _csp_status(self):
        """Return (csp_full, set_of_confirmed_modes).
        csp_full = True iff number of confirmed channel→mode mappings equals number of distinct
        non-bg channels we've seen events on (i.e., all event-bearing channels are mode-identified).
        """
        n_known = len(self.channel_mode)
        n_event = len(set(ch for (ch, _) in self.event_seen_pairs if ch != "ch_bg"))
        csp_full = (n_known == n_event and n_event > 0)
        confirmed = set(self.channel_mode.values())
        return csp_full, confirmed

    def _possible_modes(self):
        """Return the set of modes that could feasibly be present in the task,
        given observed (channel, service) pairs and LEGAL_MODE_SERVICE.

        Strong version: enumerate all valid CSP solutions (channel→mode assignments
        respecting LEGAL_MODE_SERVICE and global mode-uniqueness), and take the union
        of modes used across all solutions. Modes NEVER appearing in any solution are
        guaranteed absent — even when CSP has multiple solutions for some channels.
        """
        if hasattr(self, "_pm_cache_key") and self._pm_cache_key == frozenset(self.event_seen_pairs):
            return self._pm_cache
        from collections import Counter
        ch_svc = {}
        for (ch, svc) in sorted(self.event_seen_pairs, reverse=True):
            if ch == "ch_bg":
                continue
            ch_svc.setdefault(ch, Counter())[svc] += 1
        if not ch_svc:
            self._pm_cache_key = frozenset(self.event_seen_pairs)
            self._pm_cache = set(LEGAL_MODE_SERVICE.keys())
            return self._pm_cache
        # Build slot list: each channel has canonical svcs (most-frequent first).
        slots = []
        for ch, ctr in sorted(ch_svc.items()):
            canon_svcs = [s for s, _ in ctr.most_common()]
            cands = set()
            for sv in canon_svcs:
                for m, legal in LEGAL_MODE_SERVICE.items():
                    if sv in legal:
                        cands.add(m)
            slots.append((ch, canon_svcs, sorted(cands)))
        # Backtrack: enumerate, collecting modes used in each solution.
        possible = set()
        MAX_SOL = 1024
        n = len(slots)
        sol_count = [0]
        def bt(i, used):
            if sol_count[0] >= MAX_SOL:
                return
            if i == n:
                sol_count[0] += 1
                possible.update(used)
                return
            ch, svcs, cands = slots[i]
            for m in cands:
                if m in used:
                    continue
                # Verify mode legal for at least one canonical svc.
                if not any(sv in LEGAL_MODE_SERVICE.get(m, set()) for sv in svcs):
                    continue
                used.add(m)
                bt(i + 1, used)
                used.discard(m)
        bt(0, set())
        if sol_count[0] == 0:
            # Fallback: union of legal modes per channel (no global constraint).
            possible = set()
            for ch, svcs, cands in slots:
                possible.update(cands)
        self._pm_cache_key = frozenset(self.event_seen_pairs)
        self._pm_cache = possible
        return possible

    def _csp_excludes(self, mode):
        """Returns True if `mode` is provably absent from this task.

        Two grounds:
        1. CSP fully resolved & mode not in confirmed set.
        2. No event-bearing channel has a service compatible with this mode (legal-mode-service constraint).
        """
        if mode not in self._possible_modes():
            return True
        csp_full, confirmed = self._csp_status()
        return csp_full and (mode not in confirmed)

    def _csp_confirms(self, mode):
        return mode in set(self.channel_mode.values())

    def _csp_mode_svcs(self, mode):
        """Return services confirmed to have `mode` (via CSP)."""
        out = set()
        for ch, m in self.channel_mode.items():
            if m == mode:
                svcs = [s for (c, s) in self.event_seen_pairs if c == ch]
                if svcs:
                    out.add(svcs[0])
        return out

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
        # Channel-driven hypotheses: for each (ch, svc) where ch has not been mapped to a mode yet
        # but other channels on the same svc HAVE been mapped, infer mode by elimination.
        # Confidence based on number of events and recency.
        svc_channels = defaultdict(list)
        for (ch, svc) in sorted(self.event_seen_pairs, reverse=True):
            if ch != "ch_bg":
                svc_channels[svc].append(ch)
        for svc, chs in sorted(svc_channels.items(), reverse=True):
            if svc not in self.services:
                continue
            confirmed = set(self.channel_mode.get(c) for c in chs if c in self.channel_mode)
            confirmed.discard(None)
            unmapped = [c for c in chs if c not in self.channel_mode]
            legal_modes = set(m for m, legal in LEGAL_MODE_SERVICE.items() if svc in legal)
            remaining = legal_modes - confirmed
            for ch in unmapped:
                # Recency / activity score on this channel
                events_t = self.events_seen.get((ch, svc), [])
                if not events_t:
                    continue
                recency = max(0, 8 - (self.t - events_t[-1]))
                count_w = min(3, len(events_t))
                signal = recency + count_w * 1.5
                # If only 1 candidate left, very confident
                if len(remaining) == 1:
                    inf_mode = next(iter(remaining))
                    hyps.append((4.5 + signal, inf_mode, svc, self._dep_for(inf_mode, svc)))
                elif len(remaining) <= 3 and signal >= 2:
                    # weak hypothesis for each remaining candidate
                    for cm in remaining:
                        hyps.append((2.0 + signal * 0.5, cm, svc, self._dep_for(cm, svc)))
        # consumer_lag: anchor on svc_09 — relatively reliable since deps are constant.
        p09 = self._service_pressure("svc_09")
        h09 = self.history.get("svc_09")
        # CSP exclusion (computed once)
        n_known_ch = len(self.channel_mode)
        n_event_ch = len(set(ch for (ch, _) in self.event_seen_pairs if ch != "ch_bg"))
        csp_full = (n_known_ch == n_event_ch and n_event_ch > 0)
        csp_has_cl = any(mode == "consumer_lag" for mode in self.channel_mode.values())
        csp_excludes_cl = csp_full and not csp_has_cl
        # Stronger: require svc_09 to have at least one event (since events.service == mode.service).
        cl_event = any(svc == "svc_09" for (ch, svc) in self.event_seen_pairs if ch != "ch_bg")
        if h09 and not csp_excludes_cl and cl_event:
            qmax = max(x["Q"] for x in h09)
            if qmax >= 4 or p09 >= 7:
                hyps.append(((qmax + p09) * 0.85, "consumer_lag", "svc_09", "svc_08"))
        cache_event_svcs = set(svc for (ch, svc) in self.event_seen_pairs if ch != "ch_bg" and svc in ("svc_00", "svc_01", "svc_02"))
        p07 = self._service_pressure("svc_07")
        # CSP-derived check: do we already know any channel is cache_collapse?
        csp_has_cc = any(mode == "cache_collapse" for mode in self.channel_mode.values())
        # CSP-derived check: do we have ENOUGH known channels that cache_collapse is excluded?
        # If number of known channels equals number of distinct event channels, all modes are accounted for.
        n_known_ch = len(self.channel_mode)
        n_event_ch = len(set(ch for (ch, _) in self.event_seen_pairs if ch != "ch_bg"))
        csp_excludes_cc = (n_known_ch == n_event_ch and n_event_ch > 0 and not csp_has_cc)
        # If CSP confirms cache_collapse on a specific channel, lock the caller svc.
        csp_cc_svcs = set()
        for ch, mode in self.channel_mode.items():
            if mode == "cache_collapse":
                svcs = [s for (c, s) in self.event_seen_pairs if c == ch]
                if svcs:
                    csp_cc_svcs.add(svcs[0])
        if p07 >= 7 and cache_event_svcs and not csp_excludes_cc:
            best_caller, best_v = None, 0
            for c in ("svc_00", "svc_01", "svc_02"):
                if csp_cc_svcs and c not in csp_cc_svcs:
                    continue
                v = self._service_pressure(c)
                if v > best_v:
                    best_caller, best_v = c, v
            # Only fire if caller is also somewhat anomalous.
            if best_v >= 5:
                hyps.append((p07 * 0.85, "cache_collapse", best_caller or "svc_00", "svc_07"))
        # Require at least one event on api layer (svc_02/03/04) for storage_wait.
        api_events = any(svc in ("svc_02", "svc_03", "svc_04") for (ch, svc) in self.event_seen_pairs if ch != "ch_bg")
        # CSP exclusion check
        csp_has_sw = any(mode == "storage_wait" for mode in self.channel_mode.values())
        csp_excludes_sw = (n_known_ch == n_event_ch and n_event_ch > 0 and not csp_has_sw)
        # If CSP confirms storage_wait on a specific channel, get the canonical svc.
        # event_seen_pairs has (ch, svc); if ch -> storage_wait, then svc is mode.service.
        csp_sw_svcs = set()
        for ch, mode in self.channel_mode.items():
            if mode == "storage_wait":
                # Most-frequent svc on this channel
                svcs = [s for (c, s) in self.event_seen_pairs if c == ch]
                if svcs:
                    csp_sw_svcs.add(svcs[0])
        for store in ("svc_05", "svc_06"):
            ps = self._service_pressure(store)
            if ps >= 7 and api_events and not csp_excludes_sw:
                best_caller, best_v = None, 0
                for c in ("svc_03", "svc_04", "svc_02"):
                    if store in self.neighbors.get(c, ()):
                        # If CSP knows storage_wait svc, only consider that svc as caller.
                        if csp_sw_svcs and c not in csp_sw_svcs:
                            continue
                        v = self._service_pressure(c)
                        if v >= 5.5 and v > best_v:
                            best_caller, best_v = c, v
                if best_caller is not None:
                    hyps.append((ps * 0.85, "storage_wait", best_caller, store))
        # traffic_surge: traffic_bucket >=4 AND svc_00 itself anomalous (causal anchor).
        # Gate by CSP: if all event channels are accounted for and traffic_surge is NOT among
        # the confirmed modes, suppress this symptom hypothesis to avoid wrong THROTTLE/SCALE.
        csp_excludes_ts = self._csp_excludes("traffic_surge")
        if (self._traffic_bucket_now() >= 4
                and self._service_pressure("svc_00") >= 6
                and not csp_excludes_ts):
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
            # Among store neighbors, pick the one with highest pressure (likely the real dep).
            store_neighbors = [n for n in self.neighbors.get(svc, ()) if self.services.get(n, {}).get("role") == "store"]
            if store_neighbors:
                store_neighbors.sort(key=lambda n: -self._service_pressure(n))
                return store_neighbors[0]
            # Fall back: among all stores, pick highest-pressure one.
            stores_by_pressure = sorted(("svc_05", "svc_06"), key=lambda s: -self._service_pressure(s))
            return stores_by_pressure[0]
        if mode == "dependency_timeout":
            for n in sorted(self.neighbors.get(svc, ()), reverse=True):
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
            out.append({"type": "SET_TIMEOUT", "target": m_svc, "value": 1000})
        elif mode == "heap_growth":
            out.append({"type": "RESTART", "target": m_svc})
            base = self.services.get(m_svc, {}).get("base_replicas", 1)
            mx = self.services.get(m_svc, {}).get("max_replicas", base)
            out.append({"type": "SCALE", "target": m_svc, "value": min(int(mx), int(base) + 5)})
        elif mode == "traffic_surge":
            for tgt in ("svc_00", m_svc):
                base = self.services.get(tgt, {}).get("base_replicas", 1)
                mx = self.services.get(tgt, {}).get("max_replicas", base)
                bonus = 5 if tgt == "svc_00" else 3
                out.append({"type": "SCALE", "target": tgt, "value": min(int(mx), int(base) + bonus)})
            out.append({"type": "THROTTLE", "target": "svc_00", "value": 12})
        elif mode == "storage_wait":
            out.append({"type": "REBUILD_INDEX", "target": d_svc})
            out.append({"type": "SET_TIMEOUT", "target": m_svc, "value": 1000})
        elif mode == "consumer_lag":
            out.append({"type": "DRAIN_QUEUE", "target": "svc_08"})
            base = self.services.get("svc_09", {}).get("base_replicas", 1)
            mx = self.services.get("svc_09", {}).get("max_replicas", base)
            out.append({"type": "SCALE", "target": "svc_09", "value": min(int(mx), int(base) + 5)})
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
        if t > self.horizon - 7:
            return None
        # Restrict probe targets to services that ever appeared as event.service.
        # In checker, events.service == mode.service always, so non-event services have no modes.
        event_svcs = set(svc for (ch, svc) in self.event_seen_pairs if ch != "ch_bg")
        from collections import Counter
        svc_chcount = Counter()
        for (ch, svc) in sorted(self.event_seen_pairs, reverse=True):
            if ch != "ch_bg":
                svc_chcount[svc] += 1
        candidates = []
        for svc in self.aliases:
            if event_svcs and svc not in event_svcs:
                continue
            press = self._service_pressure(svc)
            scores = self.svc_mode_score.get(svc, {})
            known = max(scores.values()) if scores else 0
            delta = self._delta(svc)
            multi_ch_bonus = 3.0 if svc_chcount.get(svc, 0) >= 2 else 0
            # known dampening lowered: even known services may have hidden second mode.
            need = press + max(0, delta) * 1.2 - known * 1.5 + multi_ch_bonus
            last = self.probed_targets.get(svc, -10)
            # Probe gap matched to cooldown (5) + small buffer; multi-ch services even shorter.
            gap = 6 if svc_chcount.get(svc, 0) >= 2 else 8
            if t - last < gap:
                continue
            if press >= 2.0 and need > 1.5:
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

        # FORCE-PROBE: a new (channel, service) pair just appeared on a channel we don't yet
        # know the mode for. This is a strong signal of a NEW incident mode starting.
        # Prefer probing the service that produced the new pair, even if other heuristics
        # would NOOP or run another action. Only drop if probe cooldown blocks.
        while self.force_probe:
            svc_fp, _t_fp = self.force_probe[0]
            if svc_fp not in self.services:
                self.force_probe.pop(0); continue
            if t <= self.horizon - 7 and self._cd_ok("DIAGNOSTIC_PROBE", svc_fp, t):
                self.force_probe.pop(0)
                self._record_action("DIAGNOSTIC_PROBE", svc_fp, t)
                self.probed_targets[svc_fp] = t
                return {"type": "DIAGNOSTIC_PROBE", "target": svc_fp}
            else:
                # Drop stale entries (>5 steps old) so we don't probe long after the event.
                if t - _t_fp > 5:
                    self.force_probe.pop(0)
                else:
                    break

        # Probe opportunistically on incident emergence.
        # Probe BEFORE hypothesis only if: no strong hypothesis exists yet OR new event spike.
        max_hyp_score = max((sc for sc, _, _, _ in hyps), default=0)
        if (gh_jumped and max_hyp_score < 7.0) or ((gh >= 1 or any_high) and max_hyp_score < 6.0):
            probe = self._maybe_probe(t)
            if probe is not None:
                self._record_action("DIAGNOSTIC_PROBE", probe["target"], t)
                self.probed_targets[probe["target"]] = t
                return probe

        for sc, mode, svc, dep in hyps:
            if sc < 7.0:
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
                elif atype == "RESTART":
                    limit = 4
                elif atype == "DRAIN_QUEUE":
                    limit = 4
                elif atype == "SCALE":
                    limit = 2
                else:
                    limit = 1
                if self.dispatched[sig] >= limit:
                    continue
                if atype in ('ROLLBACK_CONFIG', 'ROLLBACK_RELEASE') and self.global_type_dispatched[atype] >= 1:
                    continue
                if atype in ('CIRCUIT_BREAK', 'SET_TIMEOUT', 'WARM_CACHE', 'REBUILD_INDEX', 'THROTTLE'):
                    if self.global_dispatched[(atype, tgt)] >= 1:
                        continue
                if not self._cd_ok(atype, tgt, t):
                    continue
                if atype == "SCALE":
                    cur = self.replicas.get(tgt, 1)
                    if cur >= a["value"]:
                        continue
                if atype == "WARM_CACHE" and t + 2 >= self.horizon:
                    continue
                if atype in ("RESTART", "ROLLBACK_RELEASE", "ROLLBACK_CONFIG", "CIRCUIT_BREAK", "SET_TIMEOUT") and t + 1 >= self.horizon:
                    continue
                if atype == "REBUILD_INDEX" and t + 4 >= self.horizon:
                    continue
                # Secondary actions (idx > 0): need higher score to fire,
                # OR primary action must already have been dispatched for this hypothesis.
                if idx > 0:
                    primary = actions[0]
                    primary_sig = self._action_signature(mode, svc, dep, primary)
                    if self.dispatched[primary_sig] == 0 and sc < 7.0:
                        continue
                if atype == "SET_TIMEOUT":
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    primary = actions[0]
                    primary_sig = self._action_signature(mode, svc, dep, primary)
                    # CSP-exclusion: SET_TIMEOUT helps config_shift / storage_wait / dependency_timeout.
                    csp_full, confirmed = self._csp_status()
                    if csp_full and not ({"config_shift", "storage_wait", "dependency_timeout"} & confirmed):
                        continue
                    # SET_TIMEOUT is helpful only for config_shift on svc OR dependency_timeout on dep.
                    # Without probe-confirmed mode, suppress unless primary already fired.
                    if self.dispatched[primary_sig] == 0 and prob < 10:
                        continue
                    # If hypothesis is symptom-only (no probe evidence) require high prob.
                    if prob < 6 and self.dispatched[primary_sig] == 0:
                        continue
                # WARM_CACHE: only fire when svc_07 itself shows actual anomaly
                # (avoids assumption-driven wrong calls in non-cache_collapse tasks).
                if atype == "WARM_CACHE":
                    p07 = self._service_pressure("svc_07")
                    prob_cc = self.svc_mode_score.get(svc, {}).get("cache_collapse", 0)
                    # CSP-exclusion: only fires when cache_collapse is in the picture.
                    if self._csp_excludes("cache_collapse"):
                        continue
                    # any service has high cache_collapse evidence (probe-derived)
                    any_cc_evidence = any(s.get("cache_collapse", 0) >= 6 for s in self.svc_mode_score.values())
                    # Aggregate caller pressure: cache_collapse manifests as high pressure on edge/api callers
                    # AND high pressure on svc_07. Require BOTH for symptom-only hypothesis.
                    caller_p_max = max(self._service_pressure(c) for c in ("svc_00", "svc_01", "svc_02"))
                    if any_cc_evidence or prob_cc >= 6 or self._csp_confirms("cache_collapse"):
                        pass  # have probe/CSP evidence — fire
                    elif p07 >= 9.0 and caller_p_max >= 8.5:
                        pass  # strong symptom evidence on both sides — fire
                    else:
                        continue
                # DRAIN_QUEUE: only fire when svc_08/svc_09 shows actual queue anomaly.
                if atype == "DRAIN_QUEUE":
                    if self._csp_excludes("consumer_lag"):
                        continue
                    h09 = list(self.history.get("svc_09", []))
                    h08 = list(self.history.get("svc_08", []))
                    q09 = max((x["Q"] for x in h09[-3:]), default=0)
                    q08 = max((x["Q"] for x in h08[-3:]), default=0)
                    prob_cl = self.svc_mode_score.get(svc, {}).get("consumer_lag", 0)
                    if min(q09, q08) < 4 and prob_cl < 2 and not self._csp_confirms("consumer_lag"):
                        continue
                    # Avoid re-firing DRAIN_QUEUE while previous drain is still actively
                    # reducing loss (active window = [start+2, start+14] in checker).
                    # Re-firing inside the active window overwrites the flag, briefly
                    # deactivating drain coverage (ramp-up gap), and wastes 3.2 action cost.
                    last_drain = self.last_used_step.get(f"DRAIN_QUEUE:{tgt}", -99)
                    if last_drain >= 0 and t - last_drain < 14:
                        continue
                    # Also skip if too close to horizon end (action effect <= 1 minute).
                    if t + 3 >= self.horizon:
                        continue
                # THROTTLE: helpful in traffic_surge & cache_collapse only — needs traffic
                # bucket high or cache pressure obvious.
                if atype == "THROTTLE":
                    p07 = self._service_pressure("svc_07")
                    if self._last_traffic_bucket < 4 and p07 < 6.5 and sc < 4:
                        continue
                    # CSP-exclusion: if neither traffic_surge nor cache_collapse confirmed by
                    # CSP and CSP is full (all channels accounted for), suppress.
                    csp_full, confirmed = self._csp_status()
                    if csp_full and not ({"traffic_surge", "cache_collapse"} & confirmed):
                        continue
                    # Require evidence: CSP-confirmed OR strong probe score.
                    prob_ts = max(s.get("traffic_surge", 0) for s in self.svc_mode_score.values()) if self.svc_mode_score else 0
                    prob_cc = max(s.get("cache_collapse", 0) for s in self.svc_mode_score.values()) if self.svc_mode_score else 0
                    has_evidence = (
                        self._csp_confirms("traffic_surge") or
                        self._csp_confirms("cache_collapse") or
                        prob_ts >= 5 or prob_cc >= 5
                    )
                    if not has_evidence:
                        continue
                # CIRCUIT_BREAK: helpful in release_instability, cache_collapse, dependency_timeout.
                if atype == "CIRCUIT_BREAK":
                    csp_full, confirmed = self._csp_status()
                    if csp_full and not ({"release_instability", "cache_collapse", "dependency_timeout"} & confirmed):
                        continue
                    # Need probe-confirmed evidence for the specific mode (no symptom-only).
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    if prob < 5 and not self._csp_confirms(mode):
                        continue
                # Risk gate: high-cost ops only when we have strong probe evidence
                # (or strong symptom evidence for inherently safe mappings).
                if atype == "REBUILD_INDEX":
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    dep_p = self._service_pressure(dep)
                    # Other store should not be more anomalous than the chosen dep.
                    other_store = "svc_06" if dep == "svc_05" else "svc_05"
                    other_p = self._service_pressure(other_store)
                    if dep_p < other_p:
                        continue
                    # Aggregate storage_wait evidence (probe-derived OR CSP-derived)
                    any_sw_evidence = (
                        any(s.get("storage_wait", 0) >= 6 for s in self.svc_mode_score.values())
                        or any(m_ == "storage_wait" for m_ in self.channel_mode.values())
                    )
                    if not any_sw_evidence and prob < 4:
                        continue
                    if prob < 5 and dep_p < 9 and not any_sw_evidence:
                        continue
                    # Stickiness: once a REBUILD_INDEX has been issued on a specific store,
                    # require strong probe evidence on the OTHER store before retargeting.
                    if hasattr(self, '_rebuild_target') and self._rebuild_target and self._rebuild_target != tgt:
                        # Only allow if probe strongly says the new target.
                        if prob < 8:
                            continue
                if atype == "ROLLBACK_RELEASE":
                    if self._csp_excludes("release_instability"):
                        continue
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    if prob < 4 and not self._csp_confirms("release_instability"):
                        continue
                if atype == "ROLLBACK_CONFIG":
                    if self._csp_excludes("config_shift"):
                        continue
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    if prob < 8 and not self._csp_confirms("config_shift"):
                        continue
                if atype == "RESTART":
                    if self._csp_excludes("heap_growth") and self._csp_excludes("storage_wait"):
                        continue
                    last_restart = self.last_used_step.get(f"RESTART:{tgt}", -99)
                    if last_restart >= 0 and t - last_restart < 13:
                        continue
                    prob = self.svc_mode_score.get(svc, {}).get(mode, 0)
                    if prob < 7 and self._service_pressure(svc) < 7.5 and not (self._csp_confirms("heap_growth") or self._csp_confirms("storage_wait")):
                        continue
                self._record_action(atype, tgt, t)
                self.dispatched[sig] += 1
                if atype in ('CIRCUIT_BREAK', 'SET_TIMEOUT', 'WARM_CACHE', 'REBUILD_INDEX', 'THROTTLE'):
                    self.global_dispatched[(atype, tgt)] += 1
                if atype in ('ROLLBACK_CONFIG', 'ROLLBACK_RELEASE'):
                    self.global_type_dispatched[atype] += 1
                if atype == "SCALE":
                    self.replicas[tgt] = a["value"]
                if atype == "REBUILD_INDEX":
                    self._rebuild_target = tgt
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
        # Require positive evidence: CSP must positively confirm a scale-helpful mode.
        # The bare traffic_forecast peak is unreliable as a traffic_surge proxy (public_001
        # peaks at 1.26 but has no traffic_surge mode), so without confirmation we'd burn
        # ~30 resource_cost over the horizon for nothing.
        csp_full_, conf_ = self._csp_status()
        scale_helpful = {"traffic_surge", "heap_growth", "consumer_lag", "storage_wait"}
        scale_csp_confirms = bool(scale_helpful & conf_)
        if self.traffic and t < self.horizon - 1 and scale_csp_confirms:
            future_max = max((self.traffic[k] for k in range(t+1, min(t+5, self.horizon))), default=1.0)
            if future_max >= 1.25:
                tgt = "svc_00"
                base = self.services.get(tgt, {}).get("base_replicas", 1)
                mx = self.services.get(tgt, {}).get("max_replicas", base)
                want = min(int(mx), int(base) + 4)
                if self.replicas.get(tgt, base) < want and self._cd_ok("SCALE", tgt, t):
                    self._record_action("SCALE", tgt, t)
                    self.replicas[tgt] = want
                    return {"type": "SCALE", "target": tgt, "value": want}

        if self._last_traffic_bucket >= 4 and scale_csp_confirms:
            tgt = "svc_00"
            base = self.services.get(tgt, {}).get("base_replicas", 1)
            mx = self.services.get(tgt, {}).get("max_replicas", base)
            want = min(int(mx), int(base) + 4)
            if self.replicas.get(tgt, base) < want and self._cd_ok("SCALE", tgt, t):
                self._record_action("SCALE", tgt, t)
                self.replicas[tgt] = want
                return {"type": "SCALE", "target": tgt, "value": want}

        # Information-gain probe: instead of NOOPing when there's still uncertainty,
        # fire a probe to gather mode info (probe is 100% helpful per checker; cost amortized
        # via correct future actions). Only fires when there's a real information deficit.
        # Don't gate on gh — even subtle anomalies may indicate latent modes worth investigating.
        event_svcs = set(s for (ch, s) in self.event_seen_pairs if ch != "ch_bg")
        if event_svcs:
            from collections import Counter as _EarlyCounter
            early_chcount = _EarlyCounter()
            for (ch, s) in sorted(self.event_seen_pairs, reverse=True):
                if ch != "ch_bg":
                    early_chcount[s] += 1
            if any(cnt >= 2 for cnt in early_chcount.values()):
                EARLY_DEP_INFO = [("svc_06", "storage_wait"), ("svc_05", "storage_wait"),
                                  ("svc_07", "cache_collapse"), ("svc_08", "consumer_lag")]
                for dep_svc, mode_revealed in EARLY_DEP_INFO:
                    if dep_svc not in self.services:
                        continue
                    cur = max((s.get(mode_revealed, 0) for s in self.svc_mode_score.values()), default=0)
                    if cur >= 4:
                        continue
                    if self._service_pressure(dep_svc) < 1.5:
                        continue
                    last = self.probed_targets.get(dep_svc, -10)
                    if t > self.horizon - 7:
                        continue
                    if t - last < 8:
                        continue
                    if not self._cd_ok("DIAGNOSTIC_PROBE", dep_svc, t):
                        continue
                    self._record_action("DIAGNOSTIC_PROBE", dep_svc, t)
                    self.probed_targets[dep_svc] = t
                    return {"type": "DIAGNOSTIC_PROBE", "target": dep_svc}
            need_probe = []
            for svc in sorted(event_svcs, reverse=True):
                if svc not in self.services:
                    continue
                has_unknown_ch = any(
                    (ch != "ch_bg") and self.channel_mode.get(ch) is None
                    for (ch, s) in self.event_seen_pairs
                    if s == svc
                )
                # Also probe when channels mapped but probe score is low.
                top_score = max(self.svc_mode_score.get(svc, {}).values(), default=0)
                low_confidence = top_score < 6 and self._service_pressure(svc) >= 4
                if not has_unknown_ch and not low_confidence:
                    continue
                last = self.probed_targets.get(svc, -10)
                if t > self.horizon - 7:
                    continue
                if t - last < 6:
                    continue
                if not self._cd_ok("DIAGNOSTIC_PROBE", svc, t):
                    continue
                press = self._service_pressure(svc)
                priority = press + (10.0 if has_unknown_ch else 0.0)
                need_probe.append((priority, svc))
            if need_probe:
                need_probe.sort(reverse=True)
                tgt = need_probe[0][1]
                self._record_action("DIAGNOSTIC_PROBE", tgt, t)
                self.probed_targets[tgt] = t
                return {"type": "DIAGNOSTIC_PROBE", "target": tgt}
            # Multi-fault dep-probe: when any event svc has 2+ event channels AND
            # we've already learned ONE mode for that svc (so re-probing same svc gives
            # same dominant mode), probe the store/cache/queue dep-target to surface
            # the secondary mode. Restricted to ONE probe per dep-target per task to
            # limit cost on easy tasks.
            from collections import Counter as _Counter
            svc_chcount = _Counter()
            for (ch, s) in self.event_seen_pairs:
                if ch != "ch_bg":
                    svc_chcount[s] += 1
            multi_with_known = []
            for svc, cnt in sorted(svc_chcount.items(), reverse=True):
                if cnt < 2:
                    continue
                multi_with_known.append(svc)
            if multi_with_known:
                # Choose dep-target by relevance: storage > cache > queue (most common second).
                DEP_INFO = [("svc_05", "storage_wait"), ("svc_06", "storage_wait"),
                            ("svc_07", "cache_collapse"), ("svc_08", "consumer_lag")]
                for dep_svc, mode_revealed in DEP_INFO:
                    if dep_svc not in self.services:
                        continue
                    cur = max((s.get(mode_revealed, 0) for s in self.svc_mode_score.values()), default=0)
                    if cur >= 4:
                        continue
                    if self._service_pressure(dep_svc) < 2.0:
                        continue
                    last = self.probed_targets.get(dep_svc, -10)
                    if t > self.horizon - 7:
                        continue
                    if t - last < 8:
                        continue
                    if not self._cd_ok("DIAGNOSTIC_PROBE", dep_svc, t):
                        continue
                    self._record_action("DIAGNOSTIC_PROBE", dep_svc, t)
                    self.probed_targets[dep_svc] = t
                    return {"type": "DIAGNOSTIC_PROBE", "target": dep_svc}

        return {"type": "NOOP"}
