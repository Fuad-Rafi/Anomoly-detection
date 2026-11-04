from __future__ import annotations

import json

from src.aivad_core import select_persons
from src.eval_aivad import candidates_json


def candidate(**overrides):
    base = {
        "x": 10,
        "y": 20,
        "w": 30,
        "h": 40,
        "area": 1200,
        "score": 1.2,
        "velocity": 0.1,
        "pose": 0.2,
        "deep": 0.5,
    }
    base.update(overrides)
    return base


def test_select_persons_accepts_internal_candidate_shape():
    detections, focus = select_persons(
        [candidate()],
        threshold=1.0,
        relative_threshold=0.5,
        motion_threshold=0.0,
        cluster_radius=0.0,
        top_k=3,
    )
    assert len(detections) == 1
    assert detections[0].box == (10, 20, 30, 40)
    assert focus == (25.0, 40.0)


def test_pass_through_serialisation_keeps_internal_keys():
    payload = json.loads(candidates_json([candidate()]))
    assert payload[0]["box"] == [10, 20, 30, 40]
    assert payload[0]["score"] == 1.2
    # The temporal state machine must still receive x/y/w/h, not the serialised form.
    detections, _ = select_persons(
        [candidate()],
        threshold=1.0,
        relative_threshold=0.5,
        motion_threshold=0.0,
        cluster_radius=0.0,
        top_k=3,
    )
    assert detections[0].x == 10
