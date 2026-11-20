"""The setup page: the step logic (run with node, skipped when node is missing) and a few markup guards."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = (Path(__file__).resolve().parents[1] / "predictor" / "setup.html").read_text(encoding="utf-8")
node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def steps_code():
    return PAGE.split("// <computeSteps>")[1].split("// </computeSteps>")[0]


def status(**changes):
    base = {"findings": [{"level": "ok", "check": "python", "message": "ok", "hint": ""}], "has_errors": False,
            "demo_data": [{"id": "a", "present": False}, {"id": "b", "present": False}], "demo_answer": None,
            "trainable": ["uc"], "trained": []}
    return {**base, **changes}


def compute(st, ctx=None):
    ctx = ctx or {"skippedTrain": False, "failed": None}
    script = steps_code() + f"\nconst s = computeSteps({json.dumps(st)}, {json.dumps(ctx)});" \
                            "\nconsole.log(JSON.stringify({steps: s, open: firstOpen(s)}));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=True).stdout
    return json.loads(out)


def states(result):
    return [s["state"] for s in result["steps"]]


@node
def test_a_fresh_project_opens_at_the_demo_data_question():
    result = compute(status())
    assert states(result) == ["done", "todo", "todo", "todo"] and result["open"] == 1
    assert not result["steps"][1]["canContinue"]


@node
def test_answering_no_with_nothing_to_train_lets_the_user_straight_through():
    result = compute(status(demo_answer=False, trainable=[], trained=["uc"]))
    assert states(result) == ["done", "done", "done", "todo"] and result["open"] == 3
    assert result["steps"][1]["summary"] == "Skipped: serve-only"
    assert result["steps"][2]["summary"] == "1 use case(s) have a model"


@node
def test_installed_demo_data_counts_as_answered_and_training_can_be_skipped():
    st = status(demo_data=[{"id": "a", "present": True}, {"id": "b", "present": True}])
    result = compute(st)
    assert states(result) == ["done", "done", "todo", "todo"] and result["open"] == 2
    assert result["steps"][1]["summary"] == "Demo data installed (2 sets)"
    skipped = compute(st, {"skippedTrain": True, "failed": None})
    assert states(skipped)[2] == "done" and skipped["steps"][2]["canContinue"]


@node
def test_training_is_never_done_before_the_demo_data_question_is_answered():
    result = compute(status(trainable=[], trained=[]))  # nothing to train only because there is no data yet
    assert states(result) == ["done", "todo", "todo", "todo"] and result["open"] == 1
    assert result["steps"][2]["summary"] == "Comes after the demo data step"


@node
def test_a_failed_check_blocks_everything_after_it():
    result = compute(status(has_errors=True, findings=[{"level": "error", "check": "x", "message": "m", "hint": ""}]))
    assert states(result)[0] == "problem" and result["open"] == 0 and not result["steps"][0]["canContinue"]
    assert result["steps"][0]["summary"] == "1 problem(s) to fix"


@node
def test_a_failed_job_marks_its_step_as_a_problem():
    result = compute(status(demo_answer=True), {"skippedTrain": False, "failed": 1})
    assert states(result)[1] == "problem" and result["open"] == 1


def test_the_status_line_has_no_spinner_only_the_button_does():
    template = next(line for line in PAGE.splitlines() if "el.innerHTML = '<span class=\"msg\">" in line)
    assert "spinner" not in template and 'class="bar"' in template
    assert 'class="spinner small"' in PAGE  # the one in the busy button


def test_every_section_has_a_stepper_target_and_navigation():
    for n in (1, 2, 3):
        assert f'id="next-{n}"' in PAGE
    for n in (2, 3, 4):
        assert f'id="back-{n}"' in PAGE
    assert len(re.findall(r'<section id="s\d">', PAGE)) == 4
    assert "predictor run --open" in PAGE and "start.command" in PAGE and "Ctrl+C" in PAGE
