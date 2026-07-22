import json
from pathlib import Path

from cohortcoder.release import V010_REQUIRED_CAPABILITIES, build_release_manifest, validate_release_manifest


def test_v010_manifest_requires_all_application_capabilities():
    complete = {name: True for name in V010_REQUIRED_CAPABILITIES}
    manifest = build_release_manifest(release="0.1.0", capabilities=complete)
    assert manifest["software_release_complete"] is True
    assert validate_release_manifest(manifest)["valid"] is True

    incomplete = dict(complete)
    incomplete["topk_grounded_rationales"] = False
    manifest2 = build_release_manifest(release="0.1.0", capabilities=incomplete)
    assert manifest2["software_release_complete"] is False
    assert validate_release_manifest(manifest2)["valid"] is False


def test_committed_release_manifest_declares_boundary():
    manifest = json.loads(Path("release/v0.1.0.json").read_text())
    assert manifest["software_release_complete"] is True
    assert "does not imply clinical deployment" in manifest["important_boundary"]
