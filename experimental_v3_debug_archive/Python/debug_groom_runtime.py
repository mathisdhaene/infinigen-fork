import json
from pathlib import Path

import unreal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPROJECT_PATH = PROJECT_ROOT / "BEDLAM360.uproject"
OUTPUT_PATH = PROJECT_ROOT / "exports" / "bedlam360_groom_runtime" / "groom_runtime_report.json"


def _ensure_dir(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_uproject_plugins():
    if not UPROJECT_PATH.is_file():
        return {}
    data = json.loads(UPROJECT_PATH.read_text(encoding="utf-8"))
    plugins = {}
    for item in data.get("Plugins", []):
        name = item.get("Name")
        if name:
            plugins[name] = bool(item.get("Enabled"))
    return plugins


def _class_available(name):
    obj = getattr(unreal, name, None)
    return obj is not None, None if obj is None else str(obj)


def _plugin_manager_runtime_status():
    report = {
        "Plugins_class_available": hasattr(unreal, "Plugins"),
        "queries": {},
    }
    plugins_obj = getattr(unreal, "Plugins", None)
    if plugins_obj is None:
        return report
    for plugin_name in ("HairStrands", "AlembicHairImporter"):
        value = {"query_supported": False, "enabled": None, "error": None}
        try:
            if hasattr(plugins_obj, "is_plugin_enabled"):
                value["query_supported"] = True
                value["enabled"] = bool(plugins_obj.is_plugin_enabled(plugin_name))
        except Exception as exc:
            value["error"] = str(exc)
        report["queries"][plugin_name] = value
    return report


def main():
    plugins = _read_uproject_plugins()
    class_report = {}
    for class_name in ("GroomAsset", "GroomBindingAsset", "GroomComponent"):
        available, repr_value = _class_available(class_name)
        class_report[class_name] = {
            "available": available,
            "repr": repr_value,
        }
    groom_dir_matches = sorted(name for name in dir(unreal) if "Groom" in name or "Hair" in name)

    report = {
        "uproject_path": str(UPROJECT_PATH),
        "plugin_enabled_state": {
            "HairStrands": plugins.get("HairStrands"),
            "HairStrandsCore": "module_inside_HairStrands_plugin",
            "Groom": "friendly_name_of_HairStrands_plugin",
            "AlembicHairImporter": plugins.get("AlembicHairImporter"),
        },
        "plugin_runtime_status": _plugin_manager_runtime_status(),
        "python_class_availability": class_report,
        "dir_unreal_filtered": groom_dir_matches,
    }

    _ensure_dir(OUTPUT_PATH).write_text(json.dumps(report, indent=2), encoding="utf-8")
    unreal.log(f"[BEDLAM360][GROOM_RUNTIME] Wrote runtime report: {OUTPUT_PATH}")
    unreal.log(f"[BEDLAM360][GROOM_RUNTIME] plugin_enabled_state={json.dumps(report['plugin_enabled_state'])}")
    unreal.log(f"[BEDLAM360][GROOM_RUNTIME] python_class_availability={json.dumps(report['python_class_availability'])}")


if __name__ == "__main__":
    main()
