"""
flashkeras.notebooks

Discover and generate ready-to-use Jupyter notebooks for common ML tasks
(EDA, image classification baseline, text classification baseline, ...).

Example:
    >>> import flashkeras.notebooks as fkn
    >>> fkn.list_notebooks()
    >>> fkn.new_notebook("eda_dataframe", dest="my_project/", params={"csv_path": "data.csv"})
"""

import json
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

from ._catalog import NOTEBOOKS

__all__ = ["list_notebooks", "describe", "new_notebook"]


def list_notebooks(tag: str | None = None) -> dict[str, dict]:
    """
    Return the catalog of available notebooks, keyed by name.

    Args:
        tag: If given, only return notebooks that include this tag
             (e.g. "eda", "keras", "nlp", "images").
    """
    if tag is None:
        return dict(NOTEBOOKS)
    return {name: meta for name, meta in NOTEBOOKS.items() if tag in meta["tags"]}


def describe(name: str) -> dict:
    """Return the catalog entry for a single notebook by name."""
    if name not in NOTEBOOKS:
        available = ", ".join(sorted(NOTEBOOKS))
        raise ValueError(
            f"Unknown notebook '{name}'. Available notebooks: {available}. "
            "Use flashkeras.notebooks.list_notebooks() to browse them."
        )
    return NOTEBOOKS[name]


def new_notebook(
    name: str,
    dest: str | Path = ".",
    overwrite: bool = False,
    params: dict[str, Any] | None = None,
) -> Path:
    """
    Copy a notebook template into the user's project.

    Args:
        name: Notebook key, e.g. "eda_dataframe" (see list_notebooks()).
        dest: Destination directory or full file path. If a directory is
              given, the template's default filename is used.
        overwrite: If False (default), raises if the destination file
                   already exists.
        params: Optional dict of variable_name -> value. If given, each
                matching assignment in the notebook's "parameters" cell
                (the first cell, tagged "parameters") is updated with the
                new value before the file is written. Unmatched keys are
                ignored with a warning printed to stdout.

    Returns:
        The path the notebook was written to.
    """
    meta = describe(name)

    dest = Path(dest)
    dest_path = dest / meta["filename"] if dest.is_dir() or dest.suffix == "" else dest

    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{dest_path} already exists. Pass overwrite=True to replace it."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    src = resources.files("flashkeras.notebooks.templates").joinpath(meta["filename"])

    if not params:
        with resources.as_file(src) as src_path:
            shutil.copy(src_path, dest_path)
        return dest_path

    with resources.as_file(src) as src_path:
        with open(src_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

    _inject_params(nb, params, notebook_name=name)

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    return dest_path


def _inject_params(nb: dict, params: dict[str, Any], notebook_name: str) -> None:
    """
    Overwrite variable assignments in the notebook's parameters cell
    (first code cell tagged "parameters") in place, e.g. turning
        csv_path = "REPLACE_ME.csv"
    into
        csv_path = "data.csv"
    for every key present in `params`.
    """
    params_cell = next(
        (
            c
            for c in nb["cells"]
            if c.get("cell_type") == "code"
            and "parameters" in c.get("metadata", {}).get("tags", [])
        ),
        None,
    )

    if params_cell is None:
        print(
            f"Warning: notebook '{notebook_name}' has no parameters cell; "
            "params were not applied."
        )
        return

    lines = params_cell["source"]
    remaining = dict(params)

    for i, line in enumerate(lines):
        stripped = line.strip()
        for key, value in list(remaining.items()):
            if stripped.startswith(f"{key} ") or stripped.startswith(f"{key}="):
                if "=" not in stripped:
                    continue
                var_name = stripped.split("=", 1)[0].strip()
                if var_name != key:
                    continue
                newline = "\n" if line.endswith("\n") else ""
                lines[i] = f"{key} = {value!r}{newline}"
                del remaining[key]

    if remaining:
        print(
            f"Warning: these params were not found in '{notebook_name}' "
            f"and were ignored: {list(remaining)}"
        )
