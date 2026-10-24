"""Baseline E: Isolated from-scratch small neural reference model (NumPy only).

IMPORTANT ISOLATION RULE:
- This module is an isolated baseline reference in ``nexus/baselines/`` and is NOT part
  of NEXUS. Importing this module from anywhere under ``nexus/learner/`` is forbidden
  and raises a ``RuntimeError`` at import time.

Mechanism:
- Encodes the scene (object attributes, per-object conjunctions, and directed relation
  endpoints) into a feature vector ``x_scene`` and the sentence (positional token one-hots
  + bag-of-words) into ``x_sent``.
- Trains a 2-layer Multimodal Multiplicative MLP (``h_s = ReLU(W_s x_scene + b_s)``,
  ``h_u = ReLU(W_u x_sent + b_u)``, ``z = ReLU(W_z [h_s * h_u, h_s, h_u] + b_z)``,
  ``p = sigmoid(w_o^T z + b_o)``) from scratch with Adam in pure NumPy.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np

from nexus.eval.learner_api import Prediction
from nexus.world.example import Correction, Observation
from nexus.world.scene import Scene


def _enforce_neural_isolation() -> None:
    for frame_info in inspect.stack():
        filename = frame_info.filename.replace("\\", "/")
        if "/nexus/learner/" in filename:
            raise RuntimeError(
                f"Forbidden import of neural baseline from learner module: {filename}"
            )


_enforce_neural_isolation()


@dataclass
class _NeuralObs:
    example_id: str
    scene: Scene
    tokens: tuple[str, ...]
    label: bool


@dataclass
class FromScratchNeuralBaseline:
    """Baseline E: Small from-scratch NumPy neural network trained on (scene, tokens, label)."""

    seed: int = 0
    hidden_dim: int = 64
    epochs: int = 70
    lr: float = 0.015
    l2: float = 1e-4
    name: str = "neural_from_scratch"

    _observations: list[_NeuralObs] = field(default_factory=list)
    _obs_by_id: dict[str, int] = field(default_factory=dict)
    _dirty: bool = True

    _vocab: dict[str, int] = field(default_factory=dict)
    _kinds: dict[str, int] = field(default_factory=dict)
    _colors: dict[str, int] = field(default_factory=dict)
    _sizes: dict[str, int] = field(default_factory=dict)
    _rels: dict[str, int] = field(default_factory=dict)

    _params: dict[str, np.ndarray] = field(default_factory=dict)

    def observe(self, observation: Observation) -> None:
        idx = len(self._observations)
        self._observations.append(
            _NeuralObs(
                example_id=observation.example_id,
                scene=observation.scene,
                tokens=observation.tokens,
                label=observation.label,
            )
        )
        self._obs_by_id[observation.example_id] = idx
        self._dirty = True

    def correct(self, correction: Correction) -> None:
        if correction.target_example_id in self._obs_by_id:
            idx = self._obs_by_id[correction.target_example_id]
            self._observations[idx].label = correction.corrected_label
            self._dirty = True

    def _build_inventories(self) -> None:
        vocab_set: set[str] = set()
        kinds_set: set[str] = set()
        colors_set: set[str] = set()
        sizes_set: set[str] = set()
        rels_set: set[str] = set()

        for obs in self._observations:
            vocab_set.update(obs.tokens)
            for o in obs.scene.objects:
                kinds_set.add(o.kind)
                colors_set.add(o.color)
                sizes_set.add(o.size)
            for r in obs.scene.relations:
                rels_set.add(r.name)

        self._vocab = {w: i for i, w in enumerate(sorted(vocab_set))}
        self._kinds = {k: i for i, k in enumerate(sorted(kinds_set))}
        self._colors = {c: i for i, c in enumerate(sorted(colors_set))}
        self._sizes = {s: i for i, s in enumerate(sorted(sizes_set))}
        self._rels = {r: i for i, r in enumerate(sorted(rels_set))}

    def _encode_scene(self, scene: Scene) -> np.ndarray:
        nk = max(1, len(self._kinds))
        nc = max(1, len(self._colors))
        ns = max(1, len(self._sizes))
        nr = max(1, len(self._rels))

        # Features:
        # 1) kind counts (nk), color counts (nc), size counts (ns)
        # 2) per-object (kind x color) conjunctions (nk * nc)
        # 3) per-object (kind x size) conjunctions (nk * ns)
        # 4) relation x subject_kind x object_kind tensor (nr * nk * nk)
        dim = nk + nc + ns + (nk * nc) + (nk * ns) + (nr * nk * nk)
        vec = np.zeros(dim, dtype=np.float64)

        off_c = nk
        off_s = off_c + nc
        off_kc = off_s + ns
        off_ks = off_kc + nk * nc
        off_r = off_ks + nk * ns

        for o in scene.objects:
            ki = self._kinds.get(o.kind)
            ci = self._colors.get(o.color)
            si = self._sizes.get(o.size)
            if ki is not None:
                vec[ki] += 1.0
            if ci is not None:
                vec[off_c + ci] += 1.0
            if si is not None:
                vec[off_s + si] += 1.0
            if ki is not None and ci is not None:
                vec[off_kc + ki * nc + ci] = 1.0
            if ki is not None and si is not None:
                vec[off_ks + ki * ns + si] = 1.0

        objs = scene.objects
        for r in scene.relations:
            ri = self._rels.get(r.name)
            if ri is None or r.subject >= len(objs) or r.object >= len(objs):
                continue
            k_sub = self._kinds.get(objs[r.subject].kind)
            k_obj = self._kinds.get(objs[r.object].kind)
            if k_sub is not None and k_obj is not None:
                vec[off_r + ri * (nk * nk) + k_sub * nk + k_obj] = 1.0

        return vec

    def _encode_tokens(self, tokens: tuple[str, ...]) -> np.ndarray:
        nv = max(1, len(self._vocab))
        max_pos = 6
        # Features: positional slots (max_pos * nv) + bag-of-words (nv) + final-token (nv) + length
        dim = (max_pos + 2) * nv + 1
        vec = np.zeros(dim, dtype=np.float64)
        for pos, tok in enumerate(tokens[:max_pos]):
            wi = self._vocab.get(tok)
            if wi is not None:
                vec[pos * nv + wi] = 1.0
        bow_off = max_pos * nv
        for tok in tokens:
            wi = self._vocab.get(tok)
            if wi is not None:
                vec[bow_off + wi] += 1.0
        if tokens:
            last_wi = self._vocab.get(tokens[-1])
            if last_wi is not None:
                vec[(max_pos + 1) * nv + last_wi] = 1.0
        vec[-1] = float(len(tokens)) / 6.0
        return vec

    def _fit(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self._params.clear()

        if not self._observations:
            return

        self._build_inventories()
        if not self._vocab or not self._kinds:
            return

        X_s = np.stack([self._encode_scene(o.scene) for o in self._observations], axis=0)
        X_u = np.stack([self._encode_tokens(o.tokens) for o in self._observations], axis=0)
        y = np.array([1.0 if o.label else 0.0 for o in self._observations], dtype=np.float64)

        n_samples = X_s.shape[0]
        ds = X_s.shape[1]
        du = X_u.shape[1]
        h = self.hidden_dim

        rng = np.random.default_rng(self.seed)
        W_s = rng.normal(0.0, np.sqrt(2.0 / ds), size=(ds, h))
        b_s = np.zeros(h, dtype=np.float64)
        W_u = rng.normal(0.0, np.sqrt(2.0 / du), size=(du, h))
        b_u = np.zeros(h, dtype=np.float64)
        W_z = rng.normal(0.0, np.sqrt(2.0 / (3 * h)), size=(3 * h, h))
        b_z = np.zeros(h, dtype=np.float64)
        w_o = rng.normal(0.0, np.sqrt(1.0 / h), size=(h,))
        b_o = np.zeros(1, dtype=np.float64)

        params = {
            "W_s": W_s,
            "b_s": b_s,
            "W_u": W_u,
            "b_u": b_u,
            "W_z": W_z,
            "b_z": b_z,
            "w_o": w_o,
            "b_o": b_o,
        }
        m_adam = {k: np.zeros_like(v) for k, v in params.items()}
        v_adam = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        batch_size = min(64, n_samples)

        for _ in range(self.epochs):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                bs = float(len(idx))
                xs = X_s[idx]
                xu = X_u[idx]
                yb = y[idx]

                # Forward pass
                pre_s = xs @ params["W_s"] + params["b_s"]
                hs = np.maximum(0.0, pre_s)
                pre_u = xu @ params["W_u"] + params["b_u"]
                hu = np.maximum(0.0, pre_u)

                prod = hs * hu
                cat = np.concatenate([prod, hs, hu], axis=1)
                pre_z = cat @ params["W_z"] + params["b_z"]
                z = np.maximum(0.0, pre_z)

                logits = np.clip(z @ params["w_o"] + params["b_o"][0], -30.0, 30.0)
                probs = 1.0 / (1.0 + np.exp(-logits))

                # Backward pass
                d_logits = (probs - yb) / bs
                dw_o = z.T @ d_logits + self.l2 * params["w_o"]
                db_o = np.array([np.sum(d_logits)], dtype=np.float64)

                dz = np.outer(d_logits, params["w_o"]) * (pre_z > 0.0)
                dW_z = cat.T @ dz + self.l2 * params["W_z"]
                db_z = np.sum(dz, axis=0)

                dcat = dz @ params["W_z"].T
                dprod = dcat[:, :h]
                dhs_direct = dcat[:, h : 2 * h]
                dhu_direct = dcat[:, 2 * h :]

                dhs = (dprod * hu + dhs_direct) * (pre_s > 0.0)
                dhu = (dprod * hs + dhu_direct) * (pre_u > 0.0)

                dW_s = xs.T @ dhs + self.l2 * params["W_s"]
                db_s = np.sum(dhs, axis=0)
                dW_u = xu.T @ dhu + self.l2 * params["W_u"]
                db_u = np.sum(dhu, axis=0)

                grads = {
                    "W_s": dW_s,
                    "b_s": db_s,
                    "W_u": dW_u,
                    "b_u": db_u,
                    "W_z": dW_z,
                    "b_z": db_z,
                    "w_o": dw_o,
                    "b_o": db_o,
                }
                step += 1
                for k, g in grads.items():
                    m_adam[k] = beta1 * m_adam[k] + (1.0 - beta1) * g
                    v_adam[k] = beta2 * v_adam[k] + (1.0 - beta2) * (g * g)
                    m_hat = m_adam[k] / (1.0 - beta1**step)
                    v_hat = v_adam[k] / (1.0 - beta2**step)
                    params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        self._params = params

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        self._fit()
        if not self._params:
            return Prediction(0.5, None)
        xs = self._encode_scene(scene)
        xu = self._encode_tokens(tokens)

        hs = np.maximum(0.0, xs @ self._params["W_s"] + self._params["b_s"])
        hu = np.maximum(0.0, xu @ self._params["W_u"] + self._params["b_u"])
        cat = np.concatenate([hs * hu, hs, hu])
        z = np.maximum(0.0, cat @ self._params["W_z"] + self._params["b_z"])
        logit = float(np.clip(z @ self._params["w_o"] + self._params["b_o"][0], -30.0, 30.0))
        p_true = float(1.0 / (1.0 + np.exp(-logit)))
        return Prediction(p_true, None)

    def model_size(self) -> int:
        self._fit()
        if not self._params:
            return 0
        return int(sum(arr.size for arr in self._params.values()))
