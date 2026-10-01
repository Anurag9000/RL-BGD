"""Generic JSON-kwargs callable invoker used by paper manifests."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from typing import Any


def resolve_target(target: str) -> Callable[..., Any]:
    if ":" not in target:
        raise ValueError("target must use module:function syntax")
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"experiment target is not callable: {target}")
    return function


def invoke_target(
    target: str,
    kwargs: dict[str, Any],
) -> Any:
    return resolve_target(target)(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--kwargs-json",
        default="{}",
    )
    args = parser.parse_args()
    kwargs = json.loads(args.kwargs_json)
    if not isinstance(kwargs, dict):
        raise TypeError("--kwargs-json must decode to an object")
    result = invoke_target(args.target, kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
