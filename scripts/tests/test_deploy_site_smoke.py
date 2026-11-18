"""Keep the deployed site identity probe in sync with Next.js page metadata.

The Helm smoke test must not depend on landing-page marketing copy, which can
change independently of the application's identity.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def test_helm_site_probe_matches_stable_nextjs_title() -> None:
    layout = (ROOT / "apps/site/src/app/layout.tsx").read_text()
    helm_test = (ROOT / "deploy/helm/minerva/templates/tests/test-ready.yaml").read_text()

    metadata_title = re.search(r'^\s*title:\s*"([^"]+)"', layout, re.MULTILINE)
    assert metadata_title is not None, "Next.js site metadata needs a page title"

    title_probe = re.search(r'grep -q "(<title>[^"]+)"', helm_test)
    assert title_probe is not None, "Helm must check the site identity in <title>"
    assert title_probe.group(1) == "<title>Minerva", "Use the stable product identity, not marketing copy"
    assert f"<title>{metadata_title.group(1)}".startswith(title_probe.group(1))
