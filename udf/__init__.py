"""Reusable user-defined proximal function classes."""

from importlib import import_module
from pathlib import Path

__all__ = []


def _load_udf_classes():
    package_dir = Path(__file__).resolve().parent
    module_names = sorted(
        path.stem
        for path in package_dir.glob("*.py")
        if path.stem != "__init__"
    )

    import warnings
    for module_name in module_names:
        module = import_module(f"{__name__}.{module_name}")
        udf_class = getattr(module, module_name, None)
        if udf_class is None:
            warnings.warn(
                f"UDF module '{module_name}' has no class named '{module_name}', skipping",
                stacklevel=2,
            )
            continue
        globals()[module_name] = udf_class
        __all__.append(module_name)


_load_udf_classes()
