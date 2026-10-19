"""Allowlisted dependency facts only; never persist arbitrary native logs."""
import re


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
