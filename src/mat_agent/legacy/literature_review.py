from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


class LegacyLiteratureReviewAdapter:
    def __init__(self, root: str | Path = "LiteratureReview") -> None:
        self.root = Path(root)
        self.citing_json = self.root / "citing.json"

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
            "ccdc",
        ]
        ccdc_runtime = self._check_ccdc_runtime()
        return {
            "has_citer_module": (self.root / "citer.py").exists(),
            "root_gateway_config_present": Path("config.json").exists(),
            "chromedriver_path": self._detect_chromedriver_path(),
            "ccdc_runtime": ccdc_runtime,
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

    def resolve_citing_papers(self, material_id: str, allow_live: bool = False) -> dict[str, Any]:
        cached = self._load_cached_citing_papers(material_id)
        # Only use cache when citing_papers are present, or when live lookup is disabled.
        # An empty citing_papers with allow_live=True means we should try the live backend.
        if cached is not None and (cached.get("citing_papers") or not allow_live):
            return {
                "material_id": material_id,
                "legacy_root": str(self.root.resolve()),
                "backend": "LiteratureReview.CCDCCitingPaper",
                "status": "cached",
                "source_json": str(self.citing_json.resolve()),
                "original_papers": cached.get("original_papers", []),
                "citing_papers": cached.get("citing_papers", []),
                "environment": self.diagnose_environment(),
            }

        if not allow_live:
            return {
                "material_id": material_id,
                "legacy_root": str(self.root.resolve()),
                "backend": "LiteratureReview.CCDCCitingPaper",
                "status": "prepared",
                "source_json": str(self.citing_json.resolve()),
                "original_papers": cached.get("original_papers", []) if cached else [],
                "citing_papers": [],
                "note": "Legacy cited-paper backend is available for this task, but no cached results were found.",
                "environment": self.diagnose_environment(),
            }

        return self._run_live_citing_paper_lookup(material_id)

    def _load_cached_citing_papers(self, material_id: str) -> dict[str, Any] | None:
        if not self.citing_json.exists():
            return None

        data = json.loads(self.citing_json.read_text(encoding="utf-8"))
        entry = data.get(material_id)
        if not isinstance(entry, dict):
            return None
        return entry

    def _run_live_citing_paper_lookup(self, material_id: str) -> dict[str, Any]:
        environment = self.diagnose_environment()
        try:
            citer_module = self._load_legacy_module("citer.py", "legacy_literature_review_citer")
            webdriver = getattr(citer_module, "webdriver")
            service_cls = getattr(citer_module, "Service")
            driver = None
            driver_error = None
            chromedriver_path = self._detect_chromedriver_path()
            try:
                options = webdriver.ChromeOptions()
                options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--window-size=1920,1080")
                if chromedriver_path:
                    driver = webdriver.Chrome(service=service_cls(chromedriver_path), options=options)
                else:
                    driver = webdriver.Chrome(options=options)
            except Exception as exc:
                driver_error = f"{exc.__class__.__name__}: {exc}"

            try:
                citer = citer_module.CCDCCitingPaper(
                    refcode=material_id,
                    driver=driver,
                    json_file=str(self.citing_json),
                )
                # The constructor only sets self.driver when driver is not None.
                # Guard so citer() doesn't crash with AttributeError when driver=None.
                if not hasattr(citer, "driver"):
                    citer.driver = driver
                citer.get_original_papers()
                citing_papers = citer.citer(False)
                result = {
                    "material_id": material_id,
                    "legacy_root": str(self.root.resolve()),
                    "backend": "LiteratureReview.CCDCCitingPaper",
                    "status": "fetched",
                    "source_json": str(self.citing_json.resolve()),
                    "original_papers": list(citer.original_papers),
                    "citing_papers": list(citing_papers or []),
                    "environment": environment,
                }
                if driver_error:
                    result["note"] = "Legacy retrieval ran without Selenium driver; requests-based fallback was used."
                    result["webdriver_error"] = driver_error
                return result
            finally:
                if driver is not None:
                    try:
                        driver.quit()
                    except Exception:
                        pass
        except Exception as exc:
            return {
                "material_id": material_id,
                "legacy_root": str(self.root.resolve()),
                "backend": "LiteratureReview.CCDCCitingPaper",
                "status": self._categorize_legacy_error(exc),
                "source_json": str(self.citing_json.resolve()),
                "original_papers": [],
                "citing_papers": [],
                "error": f"{exc.__class__.__name__}: {exc}",
                "environment": environment,
            }

    def _load_legacy_module(self, filename: str, module_name: str):
        module_path = self.root / filename
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load legacy module {module_path}")

        module = importlib.util.module_from_spec(spec)
        root_path = str(self.root.resolve())
        inserted = False
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
            inserted = True
        try:
            spec.loader.exec_module(module)
        finally:
            if inserted:
                sys.path.remove(root_path)
        return module

    def _detect_chromedriver_path(self) -> str | None:
        # Prefer webdriver_manager so the driver version always matches the installed Chrome.
        try:
            from webdriver_manager.chrome import ChromeDriverManager  # type: ignore

            return ChromeDriverManager().install()
        except Exception:
            pass

        env_candidates = [
            os.environ.get("MAT_AGENT_CHROMEDRIVER"),
            os.environ.get("CHROMEDRIVER"),
            os.environ.get("WEBDRIVER_CHROME_DRIVER"),
        ]
        config_candidates = self._config_driver_candidates()
        candidates = [
            *(Path(candidate) for candidate in env_candidates if candidate),
            *(Path(candidate) for candidate in config_candidates if candidate),
            Path("chromedriver.exe"),
            Path("chromedriver"),
            self.root / "chromedriver.exe",
            self.root / "chromedriver",
            Path("C:/Users/18000/OneDrive/Desktop/VSCode/chromedriver-win64/chromedriver.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return None

    def _config_driver_candidates(self) -> list[str]:
        config_path = Path("config.json")
        if not config_path.exists():
            return []

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        candidates: list[str] = []
        for key in ("chromedriver_path", "chrome_driver_path", "webdriver_chrome_driver"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        return candidates

    def _check_ccdc_runtime(self) -> dict[str, Any]:
        if importlib.util.find_spec("ccdc") is None:
            return {"available": False, "error": "ccdc package not installed"}
        try:
            from ccdc.io import EntryReader  # type: ignore

            EntryReader("CSD")
        except Exception as exc:
            return {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}
        return {"available": True}

    def _categorize_legacy_error(self, exc: Exception) -> str:
        text = str(exc).lower()
        if (
            "support@ccdc.cam.ac.uk" in text
            or "la code" in text
            or "licence" in text
            or "license" in text
            or "csd data is not available" in text
            or "cannot load csd data" in text
        ):
            return "blocked-ccdc"
        if "unable to obtain driver for chrome" in text or "nosuchdriverexception" in text:
            return "blocked-webdriver"
        return "error"
