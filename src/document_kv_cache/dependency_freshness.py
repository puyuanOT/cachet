"""Dependency freshness evidence for Cachet releases."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from document_kv_cache.serving_env import serving_environment_profiles
from document_kv_cache.storage import local_path


DEPENDENCY_FRESHNESS_RECORD_TYPE = "document_kv.dependency_freshness.v1"

__all__ = [
    "DEPENDENCY_FRESHNESS_RECORD_TYPE",
    "DependencyFreshnessEvidence",
    "DirectDependencyPin",
    "RuntimeDependencyPin",
    "TransitiveDependencyDrift",
    "dependency_freshness_to_record",
    "evaluate_dependency_freshness",
    "pyproject_direct_dependency_pins",
    "serving_profile_runtime_pins",
    "write_dependency_freshness_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class DirectDependencyPin:
    """Exact dependency pin declared by Cachet package metadata."""

    package: str
    pinned_version: str
    latest_version: str
    source: str
    current: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "package", _normalized_package(self.package))
        object.__setattr__(self, "pinned_version", _non_empty_string(self.pinned_version, "pinned_version"))
        object.__setattr__(self, "latest_version", _string_or_empty(self.latest_version, "latest_version"))
        object.__setattr__(self, "source", _non_empty_string(self.source, "source"))
        if type(self.current) is not bool:
            raise ValueError("current must be boolean")


@dataclass(frozen=True, slots=True)
class RuntimeDependencyPin:
    """Exact dependency pin used by isolated serving-engine profiles."""

    package: str
    pinned_version: str
    latest_version: str
    source: str
    current: bool
    allow_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "package", _normalized_package(self.package))
        object.__setattr__(self, "pinned_version", _non_empty_string(self.pinned_version, "pinned_version"))
        object.__setattr__(self, "latest_version", _string_or_empty(self.latest_version, "latest_version"))
        object.__setattr__(self, "source", _non_empty_string(self.source, "source"))
        object.__setattr__(self, "allow_reason", _string_or_empty(self.allow_reason, "allow_reason"))
        if type(self.current) is not bool:
            raise ValueError("current must be boolean")

    @property
    def allowed(self) -> bool:
        return self.current or bool(self.allow_reason)


@dataclass(frozen=True, slots=True)
class TransitiveDependencyDrift:
    """Resolved transitive package whose latest release is held by constraints."""

    package: str
    locked_version: str
    latest_version: str
    allowed: bool
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "package", _normalized_package(self.package))
        object.__setattr__(self, "locked_version", _non_empty_string(self.locked_version, "locked_version"))
        object.__setattr__(self, "latest_version", _non_empty_string(self.latest_version, "latest_version"))
        object.__setattr__(self, "reason", _string_or_empty(self.reason, "reason"))
        if type(self.allowed) is not bool:
            raise ValueError("allowed must be boolean")


@dataclass(frozen=True, slots=True)
class DependencyFreshnessEvidence:
    """Machine-checkable summary of Cachet dependency freshness policy."""

    pyproject_path: str
    direct_pins: tuple[DirectDependencyPin, ...]
    runtime_pins: tuple[RuntimeDependencyPin, ...] = ()
    transitive_outdated: tuple[TransitiveDependencyDrift, ...] = ()
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues

    def __post_init__(self) -> None:
        object.__setattr__(self, "pyproject_path", _non_empty_string(self.pyproject_path, "pyproject_path"))
        object.__setattr__(self, "direct_pins", tuple(self.direct_pins))
        object.__setattr__(self, "runtime_pins", tuple(self.runtime_pins))
        object.__setattr__(self, "transitive_outdated", tuple(self.transitive_outdated))
        explicit_issues = _string_tuple(self.issues, "issues")
        semantic_issues = _semantic_issues(
            direct_pins=self.direct_pins,
            runtime_pins=self.runtime_pins,
            transitive_outdated=self.transitive_outdated,
        )
        object.__setattr__(self, "issues", _dedupe_strings((*explicit_issues, *semantic_issues)))


def pyproject_direct_dependency_pins(pyproject_path: str | Path) -> tuple[tuple[str, str, str], tuple[str, ...]]:
    """Return ``(package, pinned_version, source)`` entries from ``pyproject.toml``."""

    path = local_path(str(pyproject_path))
    issues: list[str] = []
    try:
        pyproject = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return (), (f"{path} could not be read: {exc}",)
    except tomllib.TOMLDecodeError as exc:
        return (), (f"{path} must contain valid TOML: {exc}",)

    pins: list[tuple[str, str, str]] = []
    for source, requirement_text in _pyproject_requirement_texts(pyproject):
        parsed = _parse_exact_requirement(requirement_text, source, issues)
        if parsed is not None:
            package, pinned_version = parsed
            pins.append((package, pinned_version, source))
    return tuple(pins), tuple(issues)


def serving_profile_runtime_pins() -> tuple[tuple[str, str], ...]:
    """Return ``(source, requirement)`` entries from built-in serving profiles."""

    pins: list[tuple[str, str]] = []
    for profile in serving_environment_profiles():
        source = f"serving_env.{profile.backend.value}"
        pins.extend((source, constraint) for constraint in profile.dependency_constraints)
    return tuple(pins)


def evaluate_dependency_freshness(
    *,
    pyproject_path: str | Path = "pyproject.toml",
    latest_versions: Mapping[str, str],
    runtime_pins: Sequence[tuple[str, str]] | None = None,
    allowed_runtime_pins: Mapping[str, str] | None = None,
    transitive_outdated: Mapping[str, tuple[str, str]] | None = None,
    allowed_transitive_outdated: Mapping[str, str] | None = None,
) -> DependencyFreshnessEvidence:
    """Evaluate exact pins against latest package-index versions.

    Package metadata pins are strict: they must be exact and current. Runtime
    profile pins are also exact, but may be held behind an explicit
    compatibility or benchmark reason because those profiles are coupled to
    serving-engine runtime validation.
    """

    latest_by_package = _normalized_mapping(latest_versions)
    allowed_runtime_by_package = _normalized_mapping(allowed_runtime_pins or {})
    allowed_transitive_by_package = _normalized_mapping(allowed_transitive_outdated or {})
    direct_pin_inputs, parse_issues = pyproject_direct_dependency_pins(pyproject_path)
    direct_pins, direct_issues = _direct_dependency_pins(
        direct_pin_inputs,
        latest_by_package=latest_by_package,
    )
    parsed_runtime_pins, runtime_parse_issues = _runtime_dependency_pins(
        serving_profile_runtime_pins() if runtime_pins is None else tuple(runtime_pins),
        latest_by_package=latest_by_package,
        allowed_runtime_pins=allowed_runtime_by_package,
    )
    transitive_drifts, transitive_issues = _transitive_dependency_drifts(
        transitive_outdated or {},
        allowed_transitive_outdated=allowed_transitive_by_package,
    )
    return DependencyFreshnessEvidence(
        pyproject_path=str(local_path(str(pyproject_path))),
        direct_pins=direct_pins,
        runtime_pins=parsed_runtime_pins,
        transitive_outdated=transitive_drifts,
        issues=(
            *parse_issues,
            *direct_issues,
            *runtime_parse_issues,
            *transitive_issues,
        ),
    )


def dependency_freshness_to_record(evidence: DependencyFreshnessEvidence) -> dict[str, Any]:
    return {
        "record_type": DEPENDENCY_FRESHNESS_RECORD_TYPE,
        "ok": evidence.ok,
        "pyproject_path": evidence.pyproject_path,
        "direct_pins": [
            {
                "package": pin.package,
                "pinned_version": pin.pinned_version,
                "latest_version": pin.latest_version,
                "source": pin.source,
                "current": pin.current,
            }
            for pin in evidence.direct_pins
        ],
        "runtime_pins": [
            {
                "package": pin.package,
                "pinned_version": pin.pinned_version,
                "latest_version": pin.latest_version,
                "source": pin.source,
                "current": pin.current,
                "allowed": pin.allowed,
                "allow_reason": pin.allow_reason,
            }
            for pin in evidence.runtime_pins
        ],
        "transitive_outdated": [
            {
                "package": drift.package,
                "locked_version": drift.locked_version,
                "latest_version": drift.latest_version,
                "allowed": drift.allowed,
                "reason": drift.reason,
            }
            for drift in evidence.transitive_outdated
        ],
        "issues": list(evidence.issues),
    }


def write_dependency_freshness_json(evidence: DependencyFreshnessEvidence, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dependency_freshness_to_record(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _direct_dependency_pins(
    pins: Sequence[tuple[str, str, str]],
    *,
    latest_by_package: Mapping[str, str],
) -> tuple[tuple[DirectDependencyPin, ...], tuple[str, ...]]:
    direct_pins: list[DirectDependencyPin] = []
    issues: list[str] = []
    for package, pinned_version, source in pins:
        latest_version = latest_by_package.get(package, "")
        if not latest_version:
            issues.append(f"latest version missing for direct dependency {package}")
        direct_pins.append(
            DirectDependencyPin(
                package=package,
                pinned_version=pinned_version,
                latest_version=latest_version,
                source=source,
                current=bool(latest_version) and _versions_equal(pinned_version, latest_version),
            )
        )
    return tuple(direct_pins), tuple(issues)


def _runtime_dependency_pins(
    pins: Sequence[tuple[str, str]],
    *,
    latest_by_package: Mapping[str, str],
    allowed_runtime_pins: Mapping[str, str],
) -> tuple[tuple[RuntimeDependencyPin, ...], tuple[str, ...]]:
    runtime_pins: list[RuntimeDependencyPin] = []
    issues: list[str] = []
    for source, requirement_text in pins:
        parsed = _parse_exact_requirement(requirement_text, source, issues)
        if parsed is None:
            continue
        package, pinned_version = parsed
        latest_version = latest_by_package.get(package, "")
        if not latest_version:
            issues.append(f"latest version missing for runtime dependency {package}")
        runtime_pins.append(
            RuntimeDependencyPin(
                package=package,
                pinned_version=pinned_version,
                latest_version=latest_version,
                source=source,
                current=bool(latest_version) and _versions_equal(pinned_version, latest_version),
                allow_reason=allowed_runtime_pins.get(package, ""),
            )
        )
    return tuple(runtime_pins), tuple(issues)


def _transitive_dependency_drifts(
    transitive_outdated: Mapping[str, tuple[str, str]],
    *,
    allowed_transitive_outdated: Mapping[str, str],
) -> tuple[tuple[TransitiveDependencyDrift, ...], tuple[str, ...]]:
    drifts: list[TransitiveDependencyDrift] = []
    issues: list[str] = []
    for package_name, versions in sorted(transitive_outdated.items()):
        package = _normalized_package(package_name)
        if (
            not isinstance(versions, Sequence)
            or isinstance(versions, (str, bytes, bytearray))
            or len(versions) != 2
        ):
            issues.append(f"transitive_outdated[{package}] must be (locked_version, latest_version)")
            continue
        locked_version, latest_version = versions
        reason = allowed_transitive_outdated.get(package, "")
        drifts.append(
            TransitiveDependencyDrift(
                package=package,
                locked_version=str(locked_version),
                latest_version=str(latest_version),
                allowed=bool(reason),
                reason=reason,
            )
        )
    return tuple(drifts), tuple(issues)


def _pyproject_requirement_texts(pyproject: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    requirements: list[tuple[str, str]] = []
    build_system = pyproject.get("build-system")
    if isinstance(build_system, Mapping):
        requirements.extend(_requirement_entries(build_system.get("requires"), "build-system.requires"))
    project = pyproject.get("project")
    if isinstance(project, Mapping):
        requirements.extend(_requirement_entries(project.get("dependencies"), "project.dependencies"))
        optional_dependencies = project.get("optional-dependencies")
        if isinstance(optional_dependencies, Mapping):
            for extra_name, dependency_values in sorted(optional_dependencies.items()):
                requirements.extend(
                    _requirement_entries(
                        dependency_values,
                        f"project.optional-dependencies.{extra_name}",
                    )
                )
    return tuple(requirements)


def _requirement_entries(value: Any, source: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple((source, requirement) for requirement in value if isinstance(requirement, str))


def _parse_exact_requirement(
    requirement_text: str,
    source: str,
    issues: list[str],
) -> tuple[str, str] | None:
    try:
        requirement = Requirement(requirement_text)
    except InvalidRequirement as exc:
        issues.append(f"{source}: {requirement_text!r} is not a valid requirement: {exc}")
        return None
    specifiers = tuple(requirement.specifier)
    exact_specifiers = tuple(specifier for specifier in specifiers if specifier.operator == "==")
    if len(specifiers) != 1 or len(exact_specifiers) != 1:
        issues.append(f"{source}: {requirement_text!r} must be an exact == pin")
        return None
    return _normalized_package(requirement.name), exact_specifiers[0].version


def _semantic_issues(
    *,
    direct_pins: Sequence[DirectDependencyPin],
    runtime_pins: Sequence[RuntimeDependencyPin],
    transitive_outdated: Sequence[TransitiveDependencyDrift],
) -> tuple[str, ...]:
    issues: list[str] = []
    for pin in direct_pins:
        if not pin.current:
            issues.append(
                f"direct dependency {pin.package} pinned to {pin.pinned_version}, "
                f"latest is {pin.latest_version or 'unknown'}"
            )
    for pin in runtime_pins:
        if not pin.allowed:
            issues.append(
                f"runtime dependency {pin.package} pinned to {pin.pinned_version}, "
                f"latest is {pin.latest_version or 'unknown'}, and no allow reason was provided"
            )
    for drift in transitive_outdated:
        if not drift.allowed:
            issues.append(
                f"transitive dependency {drift.package} locked to {drift.locked_version}, "
                f"latest is {drift.latest_version}, and no allow reason was provided"
            )
    return tuple(issues)


def _versions_equal(left: str, right: str) -> bool:
    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return left == right


def _normalized_mapping(values: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        normalized[_normalized_package(str(key))] = str(value).strip()
    return normalized


def _normalized_package(package: str) -> str:
    normalized = canonicalize_name(str(package).strip())
    if not normalized:
        raise ValueError("package must be non-empty")
    return normalized


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_or_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value.strip()


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        normalized.append(value.strip())
    return tuple(normalized)


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _parse_key_value(values: Sequence[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} entries must use NAME=VALUE")
        key, parsed_value = value.split("=", maxsplit=1)
        parsed[_normalized_package(key)] = _non_empty_string(parsed_value, option)
    return parsed


def _parse_runtime_pin(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--runtime-pin entries must use SOURCE=REQUIREMENT")
    source, requirement = value.split("=", maxsplit=1)
    return _non_empty_string(source, "--runtime-pin source"), _non_empty_string(
        requirement,
        "--runtime-pin requirement",
    )


def _parse_outdated_package(values: Sequence[str]) -> dict[str, tuple[str, str]]:
    parsed: dict[str, tuple[str, str]] = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", maxsplit=1)[1]:
            raise ValueError("--outdated-package entries must use NAME=LOCKED:LATEST")
        package_name, versions = value.split("=", maxsplit=1)
        locked_version, latest_version = versions.split(":", maxsplit=1)
        parsed[_normalized_package(package_name)] = (
            _non_empty_string(locked_version, "--outdated-package locked version"),
            _non_empty_string(latest_version, "--outdated-package latest version"),
        )
    return parsed


def _write_or_print(record: Mapping[str, Any], output_json: str | None) -> None:
    if output_json:
        output_path = local_path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(record, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit Cachet dependency freshness evidence.")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument(
        "--latest-version",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help="Latest stable version observed for a package. Repeat for every checked package.",
    )
    parser.add_argument(
        "--runtime-pin",
        action="append",
        default=[],
        metavar="SOURCE=REQUIREMENT",
        help="Additional runtime profile exact pin to check.",
    )
    parser.add_argument(
        "--no-serving-profiles",
        action="store_true",
        help="Do not include built-in vLLM/SGLang serving profile pins.",
    )
    parser.add_argument(
        "--allow-runtime-pin",
        action="append",
        default=[],
        metavar="NAME=REASON",
        help="Allow a non-latest runtime profile pin with an explicit reason.",
    )
    parser.add_argument(
        "--outdated-package",
        action="append",
        default=[],
        metavar="NAME=LOCKED:LATEST",
        help="Resolved transitive dependency drift from poetry show --outdated.",
    )
    parser.add_argument(
        "--allow-transitive-outdated",
        action="append",
        default=[],
        metavar="NAME=REASON",
        help="Allow transitive drift with an explicit resolver-constraint reason.",
    )
    parser.add_argument("--output-json", help="Write evidence JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        runtime_pins = tuple(_parse_runtime_pin(value) for value in args.runtime_pin)
        if not args.no_serving_profiles:
            runtime_pins = (*serving_profile_runtime_pins(), *runtime_pins)
        evidence = evaluate_dependency_freshness(
            pyproject_path=args.pyproject,
            latest_versions=_parse_key_value(args.latest_version, option="--latest-version"),
            runtime_pins=runtime_pins,
            allowed_runtime_pins=_parse_key_value(args.allow_runtime_pin, option="--allow-runtime-pin"),
            transitive_outdated=_parse_outdated_package(args.outdated_package),
            allowed_transitive_outdated=_parse_key_value(
                args.allow_transitive_outdated,
                option="--allow-transitive-outdated",
            ),
        )
        record = dependency_freshness_to_record(evidence)
        _write_or_print(record, args.output_json)
    except Exception as exc:
        record = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        _write_or_print(record, args.output_json)
        return 1
    return 0 if record["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
