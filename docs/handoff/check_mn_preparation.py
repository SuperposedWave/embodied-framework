"""Create NEW config/preflight/coverage evidence, never rewrite old records."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
FILES=("stage3-openwam-libero-spatial-pilot.json","stage4-pi05-libero-spatial.json",
       "stage4-openwam-vlabench-select-fruit.json","stage4-pi05-vlabench-select-fruit.json",
       "stage4-openwam-robotwin-adjust-bottle.json","stage4-lingbot-robotwin-adjust-bottle.json")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify-assets",action="store_true")
    parser.add_argument("--clean-libero-repo",type=Path,help="Optional independent clean checkout for the original LIBERO revision")
    parser.add_argument("--port-offset",type=int,default=0,help="Diagnostic endpoints only; never changes weights or historical configs")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from embodied_eval.config import load_config
    from embodied_eval.codec import to_data
    from tests.legacy_support.frozen.mn.migration import translate
    from tests.legacy_support.frozen.mn.preflight import check
    results=[]
    for filename in FILES:
        old=load_config(ROOT/"examples"/filename)
        new=translate(old)
        if args.port_offset:
            from dataclasses import replace
            new=replace(new,model=replace(new.model,options=new.model.options|{"port":new.model.options["port"]+args.port_offset}))
        if args.clean_libero_repo and new.benchmark.adapter=="libero" and new.benchmark.options["protocol"]=="spatial-30settle":
            from dataclasses import replace
            new=replace(new,benchmark=replace(new.benchmark,options=new.benchmark.options|{"repo":str(args.clean_libero_repo.resolve())}))
        key=old.model.adapter
        (args.output/(key+".json")).write_text(json.dumps(to_data(new),ensure_ascii=False,indent=2)+"\n")
        result=check(new,verify_assets=args.verify_assets)
        (args.output/(key+"-preflight.json")).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        row={"legacy_adapter":key,"structurally_compatible":result["structurally_compatible"],"assets_verified":result["assets_verified"],
             "issues":result["issues"],"asset_errors":result["asset_errors"],"real_validation":"not_performed"}
        results.append(row)
        print(json.dumps(row),flush=True)
    receipt={"kind":"mn_independent_structure_and_local_assets_not_benchmark_score","at_utc":datetime.now(timezone.utc).isoformat(),
             "passed":all(r["structurally_compatible"] and not r["asset_errors"] for r in results),"results":results}
    (args.output/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    return 0 if receipt["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
