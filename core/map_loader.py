import sys
import importlib.util
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_MAPS_DIR = _PROJECT_ROOT / "maps"

# Ensure the project root is on sys.path so map files can do `from maps import BaseMapScenario`.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_map_scenario(map_name: str):
    """
    Load and instantiate the MapScenario for the given map name.

    Looks for maps/<normalised_name>.py where the name is lowercased with
    spaces replaced by underscores (e.g. "Tacbench 01" -> tacbench_01.py).

    Returns a MapScenario instance, or None if no file is found (with a
    warning printed to stdout so the absence is visible in the run log).

    When settings_overrides support is added, apply overrides here after
    instantiation using the returned scenario's settings_overrides dict.
    """
    normalised = map_name.lower().replace(" ", "_")
    module_path = _MAPS_DIR / f"{normalised}.py"

    if not module_path.exists():
        print(
            f"[TacBench] WARNING: no scenario file found for map '{map_name}' "
            f"(expected {module_path.relative_to(_PROJECT_ROOT)}) — "
            f"running without win/loss conditions or briefing."
        )
        return None

    spec = importlib.util.spec_from_file_location(normalised, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scenario = module.MapScenario()
    print(f"[TacBench] Loaded scenario: {module_path.name}")
    return scenario
