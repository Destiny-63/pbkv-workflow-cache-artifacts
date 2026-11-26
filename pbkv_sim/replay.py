"""Offline closed-loop replay with recorded durations, policy-driven eviction, and shadow policies."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field, replace
import hashlib

import numpy as np

from .eta import DurationModel, candidate_discounts, exact_candidate_discounts, oracle_tables, oracle_time_tables, rank_urgency, urgency_tables
from .predictors import N_ROLES, MarkovPredictor, oracle_rows, uniform_rows
from .prefix_tree import PrefixTree
from .scores import candidate_score, node_score, priority, workflow_vector
from .belady import build_next_use


@dataclass
class ReplayConfig:
    policy: str
    predictor: str
    concurrency: int
    capacity: int | None
    gamma: float = 0.7
    tau: float | None = None
    oracle_time: bool = False
    k_max: int = 3
    paths: int = 256
    seed: int = 0
    shadow: tuple = ()
    step_len: float = 0.0
    drop_retired: bool = False
    tool_visible: bool = True
    candidate_estimator: str = "exact"


@dataclass
class Live:
    wf: object
    steps: list
    idx: int = 0
    step_start: float = 0.0
    running: int = 0
    p: np.ndarray | None = None
    timeline: list = field(default_factory=list)
    wait_start: float | None = None
    waiting_tools: tuple = ()
    step_finished: bool = False


class Replay:
    def __init__(self, workflows, cfg: ReplayConfig, markov: MarkovPredictor, durations: DurationModel,
                 prediction_bank=None, next_use=None):
        self.cfg = cfg
        self.markov = markov
        self.roles = markov.roles
        self.next_use = next_use
        self.prediction_bank = prediction_bank
        if prediction_bank is not None and prediction_bank.roles != self.roles:
            raise ValueError("Prediction bank and replay role tables differ")
        if cfg.predictor == "learned" and prediction_bank is None:
            raise ValueError("Learned replay requires a prediction bank")
        self.learned_updates = 0
        self.durations = durations
        self.rng = np.random.default_rng(cfg.seed)
        self.shadow_rng = {name: np.random.default_rng(np.random.SeedSequence([
            cfg.seed, int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")]))
            for name in cfg.shadow}
        self.shadow_path_cache = {name: {} for name in cfg.shadow}
        self.tree = PrefixTree(cfg.capacity)
        self.pending = list(workflows)
        self.live: dict[str, Live] = {}
        self.terminated: set = set()
        self.events: list = []
        self.calls: list = []
        self.access_log: list = []
        self.heap: list = []
        self.seq = itertools.count()
        self.event_ids = itertools.count()
        self.divergent: dict[str, set] = {name: set() for name in cfg.shadow}
        self.refetch_by_event: dict[int, int] = {}
        self.counts = dict(hit=0, compulsory=0, refetch=0)
        self.path_cache = {}
        self.now = 0.0

    def push(self, t, kind, payload):
        heapq.heappush(self.heap, (t, next(self.seq), kind, payload))

    def run(self) -> dict:
        med = float(np.median([w.end - w.calls[0].start for w in self.pending])) if self.pending else 0.0
        for _ in range(min(self.cfg.concurrency, len(self.pending))):
            self.push(float(self.rng.uniform(0, med)), "admit", None)
        while self.heap:
            self.now, _, kind, payload = heapq.heappop(self.heap)
            getattr(self, f"on_{kind}")(payload)
        return self.result()

    def on_admit(self, _):
        if not self.pending:
            return
        wf = self.pending.pop(0)
        live = Live(wf, wf.steps())
        t, starts = self.now, []
        for step in live.steps:
            starts.append((step.role, t))
            t += step.end - step.start + wf.delay_after(step)
        live.timeline = starts + [("END", t)]
        self.live[wf.workflow_id] = live
        self.start_step(live)

    def start_step(self, live: Live):
        live.wait_start = None
        live.waiting_tools = ()
        live.step_finished = False
        live.step_start = self.now
        step = live.steps[live.idx]
        live.p = self.predict(live)
        live.running = len(step.calls)
        if self.cfg.predictor == "learned":
            # The first Reader by call_index represents a grouped Reader step.
            # Its hidden vector is unavailable until prefill finishes. Use the
            # train-only Markov prior before then, identically for all policies.
            representative = step.calls[0]
            ready = representative.first_token - step.start
            if not 0 <= ready <= step.end - step.start:
                raise ValueError("Prefill timestamp lies outside the current step")
            self.push(self.now + ready, "prediction_ready", (live.wf.workflow_id, live.idx))
        for call in step.calls:
            offset = call.submit - step.start
            self.push(self.now + offset, "call_start", (live.wf.workflow_id, call))

    def predict(self, live: Live) -> np.ndarray:
        k = self.cfg.k_max
        if self.cfg.policy == "he_count":
            return uniform_rows(k, self.roles)
        if self.cfg.predictor == "oracle":
            return oracle_rows([s.role for s in live.steps[live.idx + 1:]], k, self.roles)
        if self.cfg.predictor == "markov":
            return self.markov.predict(live.steps[live.idx].role, k)
        if self.cfg.predictor == "learned":
            return self.markov.predict(live.steps[live.idx].role, k)
        return uniform_rows(k, self.roles)

    def on_prediction_ready(self, payload):
        wid, idx = payload
        live = self.live[wid]
        if live.idx != idx:
            raise ValueError("Prediction arrived after its step ended")
        live.p = self.prediction_bank.predict(live.wf, idx, self.cfg.k_max)
        self.learned_updates += 1

    def on_call_start(self, payload):
        wid, call = payload
        acc = self.tree.access(wid, call.role, call.segments, self.now, self.on_need)
        self.access_log.append((self.now, wid, call.call_index, tuple(n.prefix_key for n in acc.path)))
        self.counts["hit"] += acc.hit_tokens
        self.counts["compulsory"] += acc.compulsory_tokens
        self.counts["refetch"] += acc.refetch_tokens
        for ev, tok in acc.refetch_by_event.items():
            self.refetch_by_event[ev] = self.refetch_by_event.get(ev, 0) + tok
        self.calls.append({"workflow_id": wid, "call_index": call.call_index, "sim_hit": acc.hit_tokens,
                           "cached_tokens": call.cached_tokens, "prompt_tokens": call.prompt_tokens})
        self.push(self.now + (call.end - call.submit), "call_end", (wid, call, acc.path))

    def on_call_end(self, payload):
        wid, call, path = payload
        self.tree.release(path, self.now)
        live = self.live[wid]
        live.running -= 1
        step = live.steps[live.idx]
        delay = live.wf.delay_after(step)
        if live.running == 0:
            # Output admission can trigger eviction. At that point the LLM
            # completion is already observable, so remaining service is zero.
            live.step_finished = True
            if delay > 0:
                live.wait_start = self.now
                live.waiting_tools = live.wf.tools_after(step)
        if path:
            out = self.tree.add_output(path[-1], call.output_tokens, wid, call.role, self.now, self.on_need)
            out.lock = 0
        if live.running:
            return
        if delay > 0:
            self.push(self.now + delay, "step_ready", wid)
            return
        self.on_step_ready(wid)

    def on_step_ready(self, wid):
        live = self.live[wid]
        live.idx += 1
        if live.idx < len(live.steps):
            self.start_step(live)
            return
        del self.live[wid]
        self.terminated.add(wid)
        if self.cfg.drop_retired:
            self.tree.drop_retired(self.terminated)
        self.push(self.now, "admit", None)

    def vectors(self, policy: str) -> dict:
        if policy.endswith("_oracleT"):
            base = policy[: -len("_oracleT")]
            saved = self.cfg.oracle_time
            self.cfg.oracle_time = True
            try:
                return self.vectors(base)
            finally:
                self.cfg.oracle_time = saved
        if policy in ("lru", "lae"):
            return {}
        gamma = 1.0 if policy == "he_time" else self.cfg.gamma
        if policy in ("he_count", "he_step"):
            return {w: workflow_vector(self.p_for(policy, l), gamma, end_id=self.roles.end_id)
                    for w, l in self.live.items()}
        tables = {w: self.urgency(l) for w, l in self.live.items()}
        if policy == "he_urg_rank":
            ets = {w: t[1] for w, t in tables.items()}
            masks = {w: self.p_for(policy, self.live[w]) > 0 for w in tables}
            us = rank_urgency(ets, masks)
            return {w: workflow_vector(self.p_for(policy, l), gamma, us[w], self.roles.end_id)
                    for w, l in self.live.items()}
        return {w: workflow_vector(self.p_for(policy, l), gamma, tables[w][0], self.roles.end_id)
                for w, l in self.live.items()}

    def p_for(self, policy: str, live: Live) -> np.ndarray:
        return uniform_rows(self.cfg.k_max, self.roles) if policy == "he_count" else live.p

    def urgency(self, live: Live):
        step = live.steps[live.idx]
        if self.cfg.oracle_time:
            future = live.timeline[live.idx + 1:]
            if self.cfg.predictor == "oracle":
                return oracle_tables(future, self.now, self.cfg.k_max, self.cfg.tau, self.roles)
            U = oracle_time_tables(future, self.now, self.cfg.k_max, self.cfg.tau, self.cfg.step_len, self.roles)
            return U, np.zeros_like(U)
        if self.cfg.predictor == "oracle":
            paths = np.array([[np.argmax(row) for row in live.p]])
        else:
            key = step.role
            if key not in self.path_cache:
                self.path_cache[key] = self.markov.sample_paths(key, self.cfg.k_max, self.cfg.paths, self.rng)
            paths = self.path_cache[key]
        return urgency_tables(self.durations, paths, step.role, live.wf.thinking, self.now - live.step_start,
                              len(step.calls), self.cfg.tau, self.rng)

    def key_fn(self, policy: str):
        if policy == "ref_offline":
            if self.next_use is None:
                raise ValueError("Offline reference requires an access-log prepass")
            return lambda n: (-self.next_use.after(n.prefix_key, self.now), n.last_access)
        if policy == "nu_min":
            if self.cfg.predictor != "oracle" or not self.cfg.oracle_time:
                raise ValueError("nu_min is defined only for oracle roles and true time")
            def next_role_use(n):
                times = [t for w, l in self.live.items() for role, t in l.timeline[l.idx + 1:]
                         if role in n.roles_by_wf.get(w, ())]
                return (0 if n.roles_by_wf and set(n.roles_by_wf) <= self.terminated else 1,
                        -min(times, default=float("inf")), n.last_access)
            return next_role_use
        if policy == "cand":
            predictions = {w: l.p for w, l in self.live.items()}
            discounts = {}
            scores = {}
            for w, live in self.live.items():
                if self.cfg.oracle_time:
                    discounts[w] = oracle_time_tables(live.timeline[live.idx + 1:], self.now,
                        self.cfg.k_max, self.cfg.tau, self.cfg.step_len, self.roles)[:, 0]
                else:
                    step = live.steps[live.idx]
                    wait_elapsed = self.now - live.wait_start if live.wait_start is not None else 0.0
                    if self.cfg.candidate_estimator == "exact":
                        discounts[w] = exact_candidate_discounts(self.durations, live.p, step.role,
                            live.wf.thinking, self.now - live.step_start, len(step.calls), self.cfg.tau,
                            live.wf.repo, live.waiting_tools, wait_elapsed, self.cfg.tool_visible, live.step_finished)
                    elif self.cfg.candidate_estimator == "mc":
                        discounts[w] = candidate_discounts(self.durations, live.p, step.role, live.wf.thinking,
                            self.now - live.step_start, len(step.calls), self.cfg.tau, self.rng,
                            self.cfg.paths, live.wf.repo, live.waiting_tools, wait_elapsed, self.cfg.tool_visible,
                            live.step_finished)
                    else:
                        raise ValueError("Unknown candidate estimator")
            def candidate_key(n):
                retired = bool(n.roles_by_wf) and set(n.roles_by_wf) <= self.terminated
                not_used = 1.0
                for wid, roles in sorted(n.roles_by_wf.items()):
                    if wid not in predictions:
                        continue
                    cache_key = (wid, tuple(sorted(roles)))
                    if cache_key not in scores:
                        scores[cache_key] = candidate_score({wid: roles}, predictions, discounts,
                                                           self.roles.index, self.roles.end_id)
                    not_used *= 1 - scores[cache_key]
                score = 1 - not_used
                return priority(policy, retired, len(n.roles_by_wf), score, n.last_access)
            return candidate_key
        vecs = self.vectors(policy)

        def key(n):
            members = n.roles_by_wf
            retired = bool(members) and set(members) <= self.terminated
            active = {w: members[w] for w in self.live if w in members}
            return priority(policy, retired, len(members), node_score(active, vecs, self.roles.index), n.last_access)

        return key

    def on_need(self, need: int):
        eid = next(self.event_ids)
        victims = self.tree.select(need, self.key_fn(self.cfg.policy))
        record = {"id": eid, "t": self.now, "need": need,
                  "victims": [v.id for v in victims], "victim_tokens": sum(v.tokens for v in victims), "shadow": {}}
        for name in self.cfg.shadow:
            rng, cache = self.rng, self.path_cache
            self.rng, self.path_cache = self.shadow_rng[name], self.shadow_path_cache[name]
            try:
                alt = self.tree.select(need, self.key_fn(name))
            finally:
                self.rng, self.path_cache = rng, cache
            diverge = {v.id for v in alt} != {v.id for v in victims}
            if diverge:
                self.divergent[name].add(eid)
            record["shadow"][name] = {"victims": [v.id for v in alt], "diverge": diverge}
        record["shadow"]["diverge"] = any(v["diverge"] for v in record["shadow"].values())
        self.tree.apply(victims, eid)
        self.events.append(record)

    def result(self) -> dict:
        summary = {
            "workflows": len(self.terminated),
            "hit_tokens": self.counts["hit"],
            "compulsory_tokens": self.counts["compulsory"],
            "refetch_tokens": self.counts["refetch"],
            "evictions": len(self.events),
            "peak_tokens": self.tree.peak,
            "overflow_tokens": self.tree.overflow_tokens,
            "peak_locked_tokens": self.tree.peak_locked,
            "reuse_tokens": self.tree.reuse_tokens,
            "untagged_reuse_tokens": self.tree.untagged_reuse_tokens,
        }
        if self.cfg.predictor == "learned":
            summary["learned_updates"] = self.learned_updates
        for name, ids in self.divergent.items():
            summary[f"diverge_rate_{name}"] = len(ids) / max(len(self.events), 1)
            summary[f"refetch_from_divergent_{name}"] = sum(self.refetch_by_event.get(e, 0) for e in ids)
        return {"summary": summary, "events": self.events, "calls": self.calls, "access_log": self.access_log}


def replay(workflows, cfg: ReplayConfig, markov: MarkovPredictor, durations: DurationModel,
           prediction_bank=None, next_use=None) -> dict:
    if cfg.policy == "ref_offline" and next_use is None:
        prepass = Replay(workflows, replace(cfg, policy="lru", predictor="none", capacity=None, shadow=()),
                         markov, durations).run()
        next_use = build_next_use(prepass["access_log"])
    return Replay(workflows, cfg, markov, durations, prediction_bank, next_use).run()
