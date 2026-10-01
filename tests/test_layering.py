"""Architecture guards.

These tests fail if a layer boundary is violated, so the hexagonal rules are
enforced by the suite rather than by convention.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
ADAPTERS = ROOT / "adapters"

FORBIDDEN_IN_CORE = {
    "urllib",
    "http",
    "socket",
    "webbrowser",
    "ssl",
    "asyncio",
    "sqlite3",
    "requests",
    "httpx",
}

FORBIDDEN_IN_DOMAIN = FORBIDDEN_IN_CORE | {"abc", "typing_extensions"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                modules.add("<relative>")
            elif node.module:
                modules.add(node.module.split(".")[0])
    return modules


def python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


class TestCoreLayerIsolation(unittest.TestCase):
    def test_core_never_imports_transport_libraries(self):
        for path in python_files(CORE):
            with self.subTest(module=path.relative_to(ROOT).as_posix()):
                leaked = imported_modules(path) & FORBIDDEN_IN_CORE
                self.assertEqual(leaked, set(), f"{path} imports {leaked}")

    def test_core_never_imports_adapters(self):
        for path in python_files(CORE):
            with self.subTest(module=path.relative_to(ROOT).as_posix()):
                source = path.read_text(encoding="utf-8")
                for line in source.splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")) and "adapters" in stripped:
                        self.fail(f"{path} imports an adapter: {stripped}")

    def test_domain_is_free_of_framework_and_transport(self):
        domain = CORE / "domain"
        for path in python_files(domain):
            with self.subTest(module=path.relative_to(ROOT).as_posix()):
                leaked = imported_modules(path) & FORBIDDEN_IN_DOMAIN
                self.assertEqual(leaked, set(), f"{path} imports {leaked}")

    def test_domain_does_not_depend_on_ports_or_services(self):
        """The innermost layer must not point outward at all."""
        for path in python_files(CORE / "domain"):
            with self.subTest(module=path.relative_to(ROOT).as_posix()):
                source = path.read_text(encoding="utf-8")
                for line in source.splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")) and (
                        "ports" in stripped or "services" in stripped
                    ):
                        self.fail(f"{path} points outward: {stripped}")

    def test_no_relative_imports(self):
        """AGENTS.md requires absolute imports throughout."""
        for path in python_files(ROOT):
            if "__pycache__" in path.parts or path.name == "test_layering.py":
                continue
            with self.subTest(module=path.relative_to(ROOT).as_posix()):
                self.assertNotIn("<relative>", imported_modules(path), "relative import found")

    def test_services_depend_on_ports_not_adapters(self):
        for path in python_files(CORE / "services"):
            with self.subTest(module=path.relative_to(ROOT).as_posix()):
                self.assertNotIn("adapters", imported_modules(path))


class TestPortContracts(unittest.TestCase):
    def test_ports_declare_only_abstract_methods(self):
        from core.ports import inbound, outbound

        for module in (inbound, outbound):
            for name in dir(module):
                candidate = getattr(module, name)
                if not isinstance(candidate, type) or not name.endswith(("Port", "UseCase")):
                    continue
                if name.endswith("UseCase") and not hasattr(candidate, "__abstractmethods__"):
                    continue
                with self.subTest(port=name):
                    for attr, value in vars(candidate).items():
                        if attr.startswith("_") or not callable(value):
                            continue
                        self.assertTrue(
                            getattr(value, "__isabstractmethod__", False),
                            f"{name}.{attr} is not abstract",
                        )

    def test_translator_is_reachable_through_the_port(self):
        """The HTTP layer must only depend on the port, not the concrete class."""
        from adapters.inbound.http.openai_adapter import OpenAIProtocolTranslator
        from core.ports.inbound import ProtocolTranslatorPort

        self.assertTrue(issubclass(OpenAIProtocolTranslator, ProtocolTranslatorPort))
        for method in ProtocolTranslatorPort.__abstractmethods__:
            self.assertTrue(
                hasattr(OpenAIProtocolTranslator, method),
                f"translator is missing port method {method}",
            )

    def test_every_adapter_implements_its_port(self):
        from adapters.outbound.oauth.google_oauth import GoogleOAuthAdapter
        from adapters.outbound.storage.file_storage import FileTokenStorage
        from adapters.outbound.upstream.google_cloudcode import GoogleCloudCodeAdapter
        from core.ports.outbound import (
            OAuthProviderPort,
            TokenSourcePort,
            TokenStoragePort,
            UpstreamModelPort,
        )

        self.assertTrue(issubclass(GoogleCloudCodeAdapter, UpstreamModelPort))
        self.assertTrue(issubclass(GoogleOAuthAdapter, OAuthProviderPort))
        self.assertTrue(issubclass(FileTokenStorage, TokenStoragePort))
        # The keyring source must NOT claim to be writable.
        from adapters.outbound.storage.secret_service import SecretServiceCredentialSource

        self.assertTrue(issubclass(SecretServiceCredentialSource, TokenSourcePort))
        self.assertFalse(issubclass(SecretServiceCredentialSource, TokenStoragePort))


class TestCompositionRoot(unittest.TestCase):
    def test_only_the_container_knows_every_adapter(self):
        offenders = []
        for path in python_files(ROOT):
            if "__pycache__" in path.parts or path.name in ("container.py", "test_layering.py"):
                continue
            source = path.read_text(encoding="utf-8")
            hits = sum(
                1
                for line in source.splitlines()
                if line.strip().startswith(("import ", "from ")) and "adapters." in line
            )
            if hits >= 4:
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [], "more than one module wires concrete adapters")

    def test_wiring_constructor_contains_no_control_flow(self):
        """Container.__init__ should only wire, never branch.

        Presentation concerns (the startup banner) legitimately branch, so the
        check is scoped to the wiring method itself.
        """
        tree = ast.parse((ROOT / "container.py").read_text(encoding="utf-8"))
        container = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Container"
        )
        init = next(
            node
            for node in container.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        branching = [
            type(node).__name__
            for node in ast.walk(init)
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With))
        ]
        self.assertEqual(branching, [], "Container.__init__ branches instead of only wiring")


class TestSecretHygiene(unittest.TestCase):
    # Assembled from fragments so this file does not match itself.
    MARKERS = (
        "GOCS" + "PX-",
        "apps." + "googleusercontent.com",
        "ya2" + "9.",
        "AI" + "za",
    )

    def test_no_credential_literals_in_source(self):
        offenders = []
        for path in python_files(ROOT):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in self.MARKERS:
                if marker in text:
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}: {marker[:6]}...")
        self.assertEqual(offenders, [], "credential literals found in source")

    def test_env_is_gitignored(self):
        """A populated .env is expected locally; it must never be committed."""
        ignore = ROOT / ".gitignore"
        self.assertTrue(ignore.exists(), ".gitignore is missing")
        patterns = {
            line.strip()
            for line in ignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        self.assertIn(".env", patterns, ".env is not gitignored")
        self.assertIn("credentials.json", patterns)

    def test_env_example_carries_no_values(self):
        example = ROOT / ".env.example"
        if not example.exists():
            self.skipTest("no .env.example present")
        populated = []
        for line in example.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ANTIDAPTER_CLIENT"):
                key, _, value = line.partition("=")
                if value.strip():
                    populated.append(key)
        self.assertEqual(populated, [], ".env.example must not contain real values")


if __name__ == "__main__":
    unittest.main()
