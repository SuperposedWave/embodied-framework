"""Allowlisted dependency facts only; never persist arbitrary native logs."""
import re
from pathlib import Path


def benchmark_imports(family):
    # VLABench's simulator client deliberately has no torch; inference runs in
    # its independent model process. LIBERO needs CPU torch for init states.
    return {'libero': ['torch', 'cv2', 'mujoco'],
            'vlabench': ['cv2', 'mujoco'],
            'robotwin': ['torch', 'cv2', 'sapien', 'mplib']}[family]


def dependency_facts(text):
    libraries = set()
    for pattern in (
        r"(?<![\w/])([A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)*): cannot open shared object file",
        r"error while loading shared libraries: ([A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)*)",
    ):
        libraries.update(re.findall(pattern, text))
    return {
        'missing_shared_libraries': sorted(libraries),
        'missing_modules': sorted(set(re.findall(r"No module named ['\"]([A-Za-z0-9_.]+)['\"]", text))),
        'exception_types': sorted(set(re.findall(r'\b((?:ModuleNotFound|Import|Runtime|Value|Type|FileNotFound|Attribute|Timeout)Error|OSError):', text))),
    }


def import_failure_details(text):
    """Names and safe traceback locations; no arbitrary exception message."""
    names = sorted(set(re.findall(r"cannot import name ['\"]([A-Za-z0-9_]+)['\"] from ['\"]([A-Za-z0-9_.]+)['\"]", text)))
    locations = []
    workspace = Path(__file__).resolve().parents[3]
    for filename, line, function in re.findall(r'File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)', text):
        path = Path(filename)
        try:
            safe = str(path.relative_to(workspace))
        except ValueError:
            safe = path.name
        locations.append({'file': safe, 'line': int(line), 'function': function})
    return {'failed_named_imports': [{'name': name, 'module': module} for name, module in names],
            'undefined_symbols': sorted(set(re.findall(r'undefined symbol: ([A-Za-z0-9_.$@]+)', text))),
            'traceback_locations': locations[-24:]}
