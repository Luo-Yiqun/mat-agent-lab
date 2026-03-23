"""Top-level package for the materials agent lab application."""

from .ai import AgentManager
from .orchestrator import MaterialsAgentApp

__all__ = ["AgentManager", "MaterialsAgentApp", "__version__"]

__version__ = "0.1.0"

