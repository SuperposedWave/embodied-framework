"""Diagnose benchmark factory/task enumeration only; no reset or actions/model."""
import argparse
import contextlib
import io
import json
import resource
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'docs/handoff'))
from platform_diagnostics import dependency_facts, import_failure_details
from probe_mn_components import public_exception


def main(args):
    phase, benchmark = 'config', None
    report = {'kind': 'benchmark_import_and_task_probe_no_reset_model_or_actions', 'passed': False}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from embodied_eval.config import load_config
            from embodied_eval.v03.registry import default_registry
            config = load_config(args.config)
            phase = 'benchmark_factory'
            benchmark = default_registry().benchmark(config.benchmark.adapter).factory(config.benchmark.options, args.output)
            phase = 'task_enumeration'
            tasks = benchmark.enumerate_tasks(config.episodes)
            report.update(passed=True, tasks=len(tasks))
    except Exception as error:
        report.update(phase=phase, exception_chain=public_exception(error),
                      dependency_facts=dependency_facts(traceback.format_exc()),
                      import_failure_details=import_failure_details(traceback.format_exc()))
    finally:
        if benchmark is not None:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                benchmark.close()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
