"""dashboard/home.py runs its whole page-routing/data-loading logic at
module level (it's a Streamlit script, not just a library), so importing
it directly outside a real `streamlit run` would execute the live app —
not a safe or fast thing for a test to do. `py_compile` (parses + compiles
to bytecode without executing) is the right level of check here: it
catches syntax errors, typos, and indentation mistakes — exactly the class
of mistake a manual file split could introduce — without needing a live
data environment or Streamlit runtime context.
"""

import importlib
import py_compile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_home_py_compiles():
    py_compile.compile(str(PROJECT_ROOT / "dashboard" / "home.py"), doraise=True)


def test_dashboard_lib_modules_compile():
    lib_dir = PROJECT_ROOT / "dashboard" / "lib"
    if not lib_dir.exists():
        return  # split hasn't happened yet in this checkout
    for path in lib_dir.glob("*.py"):
        py_compile.compile(str(path), doraise=True)


def test_dashboard_lib_modules_import():
    """py_compile above only catches syntax errors. Unlike home.py (which
    executes page-routing/data-loading logic at module level), the
    dashboard/lib/*.py modules are plain function/constant libraries with
    no top-level Streamlit runtime calls, so actually importing them here
    is safe and additionally catches NameErrors, bad import lists, and
    circular imports that py_compile can't see.
    """
    lib_dir = PROJECT_ROOT / "dashboard" / "lib"
    if not lib_dir.exists():
        return  # split hasn't happened yet in this checkout
    for path in sorted(lib_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        importlib.import_module(f"dashboard.lib.{path.stem}")
