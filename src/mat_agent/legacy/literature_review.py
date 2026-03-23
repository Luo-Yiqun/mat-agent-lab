from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


class LegacyLiteratureReviewAdapter:
    def __init__(self, root: str | Path = "LiteratureReview") -> None:
        self.root = Path(root)

    def describe_assets(self) -> dict[str, Any]:
        expected_files = [
            "citer.py",
            "main.py",
            "reviewer.py",
            "utilities.py",
            "html_parser.py",
            "evaluate.py",
        ]
        return {
            "root": str(self.root.resolve()),
            "exists": self.root.exists(),
            "files": {name: (self.root / name).exists() for name in expected_files},
            "diagnostics": self.diagnose_environment(),
        }

    def diagnose_environment(self) -> dict[str, Any]:
        optional_modules = [
            "selenium",
            "trafilatura",
            "sentence_transformers",
            "curl_cffi",
            "openai",
            "tqdm",
            "pypdf",
        ]
        return {
            "has_citer_module": (self.root / "citer.py").exists(),
            "root_gateway_config_present": Path("config.json").exists(),
            "optional_dependencies": {
                module_name: importlib.util.find_spec(module_name) is not None
                for module_name in optional_modules
            },
        }

    def build_retrieval_handoff(self, material_id: str) -> dict[str, Any]:
        return {
            "material_id": material_id,
            "legacy_root": str(self.root.resolve()),
            "status": "prepared",
            "note": (
                "The preserved LiteratureReview workflow has been detected and handed off as the retrieval backend. "
                "Existing functions were left untouched; this adapter only reports environment readiness."
            ),
            "environment": self.diagnose_environment(),
        }
