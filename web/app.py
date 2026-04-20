from __future__ import annotations

import dataclasses
import sys
import tempfile
import threading
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_agent.input_file_generator import (
    _default_output_name,
    _normalize_software,
    generate_input_file,
)
from mat_agent.models import TaskRoute, UserRequest
from mat_agent.orchestrator import MaterialsAgentApp

app = Flask(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "mat_agent_web_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

RUNS: dict[str, dict] = {}


def _serialize(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _serialize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def start_run():
    form = request.form

    paper_paths: list[str] = []
    for f in request.files.getlist("papers"):
        if f and f.filename:
            dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(f.filename).name}"
            f.save(dest)
            paper_paths.append(str(dest))

    structure_paths: list[str] = []
    for f in request.files.getlist("structures"):
        if f and f.filename:
            dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(f.filename).name}"
            f.save(dest)
            structure_paths.append(str(dest))

    route_map = {"qa": TaskRoute.QA, "simulation": TaskRoute.SIMULATION}
    route_hint = route_map.get(form.get("route", "").lower())

    constraints: dict[str, Any] = {}
    for key in ("software", "accuracy", "budget"):
        val = form.get(key, "").strip()
        if val:
            constraints[key] = val

    user_request = UserRequest(
        task=form["task"],
        material_id=form.get("material_id", "").strip() or None,
        material_name=form.get("material_name", "").strip() or None,
        paper_paths=paper_paths,
        structure_paths=structure_paths,
        constraints=constraints,
        route_hint=route_hint,
    )

    enable_ai = form.get("no_ai") != "true"
    state_root = str(PROJECT_ROOT / "run-artifacts")
    config_path = str(PROJECT_ROOT / "config.json")

    web_run_id = uuid.uuid4().hex[:12]
    RUNS[web_run_id] = {"status": "running", "result": None, "error": None}

    def _worker():
        try:
            instance = MaterialsAgentApp(
                state_root=state_root,
                enable_ai=enable_ai,
                config_path=config_path,
            )
            result = instance.run(user_request, approve_execution=False)
            RUNS[web_run_id]["status"] = "done"
            RUNS[web_run_id]["result"] = result
        except Exception as exc:
            RUNS[web_run_id]["status"] = "error"
            RUNS[web_run_id]["error"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"run_id": web_run_id})


@app.route("/status/<run_id>")
def run_status(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    if run["status"] == "running":
        return jsonify({"status": "running"})
    if run["status"] == "error":
        return jsonify({"status": "error", "error": run["error"]})

    result = run["result"]
    return jsonify({
        "status": "done",
        "orch_run_id": result.run_id,
        "result": _serialize(result),
    })


@app.route("/download/<run_id>/<path:filename>")
def download_file(run_id: str, filename: str):
    file_path = PROJECT_ROOT / "run-artifacts" / run_id / "generated" / filename
    if not file_path.exists():
        return "File not found", 404
    return send_file(str(file_path), as_attachment=True, download_name=filename)


@app.route("/preview/<run_id>/<path:filename>")
def preview_file(run_id: str, filename: str):
    file_path = PROJECT_ROOT / "run-artifacts" / run_id / "generated" / filename
    if not file_path.exists():
        return "File not found", 404
    return file_path.read_text(encoding="utf-8", errors="replace"), 200, {
        "Content-Type": "text/plain; charset=utf-8"
    }


@app.route("/generate-input", methods=["POST"])
def start_generate_input():
    form = request.form
    software_raw = form.get("software", "").strip()
    prompt = form.get("prompt", "").strip()
    model = form.get("model", "").strip() or None

    if not software_raw or not prompt:
        return jsonify({"error": "software and prompt are required"}), 400

    try:
        software = _normalize_software(software_raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    output_name = _default_output_name(software)
    output_dir = PROJECT_ROOT / "test_generated_files"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_name
    config_path = PROJECT_ROOT / "config.json"

    web_run_id = uuid.uuid4().hex[:12]
    RUNS[web_run_id] = {"status": "running", "result": None, "error": None, "type": "generate"}

    def _worker():
        try:
            kwargs: dict[str, Any] = {
                "requirement": prompt,
                "software": software,
                "output_path": output_path,
                "config_path": config_path,
            }
            if model:
                kwargs["model"] = model
            path = generate_input_file(**kwargs)
            content = Path(path).read_text(encoding="utf-8", errors="replace")
            RUNS[web_run_id]["status"] = "done"
            RUNS[web_run_id]["result"] = {
                "path": str(path),
                "filename": Path(path).name,
                "content": content,
            }
        except Exception as exc:
            RUNS[web_run_id]["status"] = "error"
            RUNS[web_run_id]["error"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"run_id": web_run_id})


@app.route("/generate-status/<run_id>")
def generate_status(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    if run["status"] == "running":
        return jsonify({"status": "running"})
    if run["status"] == "error":
        return jsonify({"status": "error", "error": run["error"]})
    return jsonify({"status": "done", "result": run["result"]})


@app.route("/download-generated/<filename>")
def download_generated(filename: str):
    file_path = PROJECT_ROOT / "test_generated_files" / filename
    if not file_path.exists():
        return "File not found", 404
    return send_file(str(file_path), as_attachment=True, download_name=filename)


if __name__ == "__main__":
    print("\n  mat-agent-lab Web UI")
    print("  → http://localhost:5000\n")
    app.run(debug=True, port=5000, use_reloader=False)
