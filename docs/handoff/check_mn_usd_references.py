"""Read-only USD dependency audit, not Isaac/renderer readiness or evaluation.

Core USD may not resolve Omniverse builtin materials or network assets. Report
those separately; never silently rewrite USD references or activate/deactivate
prims. Inspect five DROID roots and the source-defined ARX robot root only.
"""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
MODELS = WORKSPACE.parent.parent / "Model"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from pxr import Sdf, Usd, UsdUtils
    roots = [("droid_sim", MODELS / "DROID-sim-environments" / name) for name in
        ("franka_robotiq_2f_85_flattened.usd", "scene1.usd", "scene2.usd", "scene3.usd", "table.usd")]
    roots.append(("robodojo", WORKSPACE / "upstream/RoboDojo/Assets/Robots/x5/ARX.usd"))
    rows = []
    for component, path in roots:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
        issues = []
        for ref in sorted(set(unresolved)):
            parsed = urlsplit(ref)
            # Only public authored asset IDs. Never serialize query/fragment/userinfo.
            if parsed.scheme:
                # Core USD normalizes http://host/path to http:/host/path.
                # Restore only the public display form, NEVER modify a layer.
                public = parsed.scheme + "://" + ((parsed.hostname or "") + parsed.path if parsed.netloc else parsed.path.lstrip("/"))
                kind = "remote_asset_not_resolved_by_core_usd"
            elif ref.endswith(".mdl") and ("/" not in ref or ref.startswith("gltf/")):
                public, kind = ref, "possible_runtime_material_builtin_not_verified"
            else:
                public, kind = ref, "unresolved_local_reference"
            issues.append({"reference": public, "kind": kind})
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
        prims = []
        if stage:
            for prim in stage.TraverseAll():
                if prim.HasAuthoredPayloads() or prim.HasAuthoredReferences():
                    prims.append({"path": str(prim.GetPath()), "active": prim.IsActive(),
                        "loaded": prim.IsLoaded(), "has_payload": prim.HasAuthoredPayloads(),
                        "has_references": prim.HasAuthoredReferences()})
                    if not prim.IsActive() and prim.HasAuthoredPayloads():
                        payloads = prim.GetMetadata("payload")
                        prims[-1]["inactive_authored_payloads"] = [p.assetPath for p in payloads.GetAddedOrExplicitItems()]
        rows.append({"component": component, "root": str(path), "opened": stage is not None,
            "resolved_layer_count": len(layers), "resolved_non_layer_count": len(assets),
            "unresolved_count": len(issues), "unresolved": issues, "reference_prims_load_none": prims,
            "complete_reference_closure_verified": stage is not None and not issues})
        print(component + ": " + path.name + ": unresolved=" + str(len(issues)), flush=True)
    receipt = {"kind": "read_only_core_usd_dependency_audit_not_simulator_ready", "at_utc": datetime.now(timezone.utc).isoformat(),
        "usd_core_version": metadata.version("usd-core"), "results": rows, "scope": "listed roots only, NOT all task objects/materials",
        "source_or_assets_modified": False, "remote_asset_downloaded": False,
        "simulator_started": False, "gpu_work": False, "official_score": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
