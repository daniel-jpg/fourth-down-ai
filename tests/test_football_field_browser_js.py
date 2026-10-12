"""Run the field-position components.v2 JavaScript with a fake DOM.

Headless Streamlit AppTest does not execute this JavaScript; this test does.
"""

import ast
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_field_position_edit_commit_cancel_and_keyboard():
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for browser component JavaScript QA")

    parsed = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    js = next(
        assignment.value.value
        for assignment in parsed.body
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FIELD_JS"
            for target in assignment.targets
        )
        and isinstance(assignment.value, ast.Constant)
    )
    assert "export default function(component)" in js
    js = js.replace(
        "export default function(component)",
        "function createField(component)", 1
    )
    harness = r"""
const assert = require("node:assert/strict");
const saved = [];
const element = () => ({
  textContent: "", value: "", style: {}, attrs: {},
  classList: { add() {}, remove() {} },
  setAttribute(key, value) { this.attrs[key] = value; },
  setPointerCapture() {},
  getBoundingClientRect() { return { left: 0, width: 100 }; },
  blur() {},
});
const elements = new Map();
for (const id of [
  "#field", "#ball", "#first-down", "#direction-info",
  "#field-status", "#los-input"
]) elements.set(id, element());
createField({
  parentElement: {
    querySelector(id) {
      assert.ok(elements.has(id), "Unexpected selector: " + id);
      return elements.get(id);
    }
  },
  data: { yardline_100: 30, ydstogo: 4 },
  setStateValue(name, value) { saved.push([name, value]); }
});
const input = elements.get("#los-input");
const ball = elements.get("#ball");
assert.equal(String(input.value), "30");
assert.equal(ball.style.left, "30%");

// Clearing must not rewrite the field to 1 during typing.
input.value = "";
input.oninput();
assert.equal(input.value, "");
assert.equal(ball.style.left, "30%");
assert.deepEqual(saved, []);

// A valid draft previews without being committed.
input.value = "75";
input.oninput();
assert.equal(input.value, "75");
assert.equal(ball.style.left, "75%");
assert.deepEqual(saved, []);
input.onchange();
assert.equal(String(input.value), "75");
assert.deepEqual(saved, [["yardline_100", 75]]);

// Leaving the box empty restores the last persisted value.
input.value = "";
input.oninput();
input.onchange();
assert.equal(String(input.value), "75");
assert.equal(ball.style.left, "75%");
assert.equal(saved.length, 1);

// Out-of-bounds drafts can be edited; clamp only when committed.
input.value = "100";
input.oninput();
assert.equal(input.value, "100");
assert.equal(ball.style.left, "75%");
input.onchange();
assert.equal(String(input.value), "99");
assert.equal(ball.style.left, "99%");
assert.deepEqual(saved.at(-1), ["yardline_100", 99]);

// A canceled pointer drag must not leave an unsaved field preview.
const count = saved.length;
ball.onpointerdown({ pointerId: 1 });
ball.onpointermove({ clientX: 10 });
assert.equal(ball.style.left, "10%");
ball.onpointercancel();
assert.equal(ball.style.left, "99%");
assert.equal(String(input.value), "99");
assert.equal(saved.length, count);

// Keyboard movements still update saved game state.
ball.onkeydown({ key: "ArrowLeft", preventDefault() {} });
assert.equal(ball.style.left, "98%");
assert.deepEqual(saved.at(-1), ["yardline_100", 98]);
console.log("Field-position input and drag tests passed");
"""
    process = subprocess.run(
        ["node", "-"],
        input=js + "\n" + harness,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert process.returncode == 0, process.stdout + "\n" + process.stderr
    assert "tests passed" in process.stdout
