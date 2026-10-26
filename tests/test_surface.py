"""The package surface: one declaration per name, and no accidental exports."""

from __future__ import annotations

import importlib
import importlib.metadata
import unittest

import betapair

MODULE_NAMES = (
    "beta",
    "gate",
    "hierarchy",
    "irt",
    "mcnemar",
    "models",
    "report",
    "series",
    "snapshot",
    "tracker",
)
MODULES = tuple(importlib.import_module(f"betapair.{name}") for name in MODULE_NAMES)


class SurfaceTest(unittest.TestCase):
    def test_top_level_all_matches_the_submodule_declarations(self) -> None:
        declared = {name for module in MODULES for name in module.__all__}
        top_level = set(betapair.__all__) - {"__version__"}
        # Every top-level name must be declared by the module that owns it, and
        # every declared name that is re-exported must be listed here.
        self.assertLessEqual(top_level, declared)
        self.assertEqual(top_level, declared & set(dir(betapair)))

    def test_every_public_module_declares_all(self) -> None:
        for module in MODULES:
            with self.subTest(module=module.__name__):
                self.assertTrue(module.__all__, f"{module.__name__} has no __all__")
                self.assertEqual(len(set(module.__all__)), len(module.__all__))

    def test_every_exported_name_is_importable(self) -> None:
        for name in betapair.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(betapair, name))

    def test_version_comes_from_the_installed_distribution(self) -> None:
        self.assertEqual(betapair.__version__, importlib.metadata.version("betapair"))


class MarkerTest(unittest.TestCase):
    def test_the_three_sentinels_are_one_enum(self) -> None:
        snapshot = importlib.import_module("betapair.snapshot")
        self.assertEqual(type(snapshot.ERROR), snapshot.Marker)
        self.assertEqual(type(snapshot.RETIRED), snapshot.Marker)
        self.assertEqual(type(snapshot.FAILED), snapshot.Marker)
        self.assertEqual(len({snapshot.ERROR, snapshot.RETIRED, snapshot.FAILED}), 3)
        self.assertEqual(repr(snapshot.ERROR), "ERROR")
        self.assertEqual(snapshot.ERROR.value, "error")

    def test_the_fit_rename_keeps_both_names(self) -> None:
        hierarchy = importlib.import_module("betapair.hierarchy")
        self.assertIs(hierarchy.Fit, hierarchy.HierarchyFit)
        self.assertIn("HierarchyFit", betapair.__all__)
        self.assertNotIn("Fit", betapair.__all__)


if __name__ == "__main__":
    unittest.main()
