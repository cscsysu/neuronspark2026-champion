class Agent:
    def reset(self, task_info):
        self.task_info = task_info
        self.services = {svc["alias"]: svc for svc in task_info.get("services", [])}
        self.used = set()

    def _once(self, key, action):
        if key in self.used:
            return None
        self.used.add(key)
        return action

    def act(self, observation):
        health = observation.get("health", [])
        worst = max(
            health,
            key=lambda h: h.get("latency_bucket", 0) + h.get("error_bucket", 0) + h.get("saturation_bucket", 0) + h.get("queue_bucket", 0),
            default={"service": "svc_00"},
        )
        target = worst.get("service", "svc_00")
        if target not in self.services:
            target = "svc_00"
        if observation.get("global_health_bucket", 0) >= 4:
            svc = self.services["svc_00"]
            action = self._once("scale_svc_00", {"type": "SCALE", "target": "svc_00", "value": min(svc["max_replicas"], svc["base_replicas"] + 3)})
            if action:
                return action
        if any(item.get("queue_bucket", 0) >= 4 for item in health):
            action = self._once("drain_queue", {"type": "DRAIN_QUEUE", "target": "svc_08"})
            if action:
                return action
            svc = self.services["svc_09"]
            action = self._once("scale_worker", {"type": "SCALE", "target": "svc_09", "value": min(svc["max_replicas"], svc["base_replicas"] + 3)})
            if action:
                return action
        if observation.get("traffic_bucket", 0) >= 4:
            svc = self.services.get(target, self.services["svc_00"])
            action = self._once(f"scale_{target}", {"type": "SCALE", "target": target, "value": min(svc["max_replicas"], svc["base_replicas"] + 2)})
            if action:
                return action
        if observation.get("time", 0) > 30 and observation.get("global_health_bucket", 0) >= 3:
            action = self._once("throttle", {"type": "THROTTLE", "target": "svc_00", "value": 10})
            if action:
                return action
        return {"type": "NOOP"}
