"""One config-resolution path for the CLI and the library entry (card 018, design_v2 5.1 / 7.1 test 3).

Expected values come from DEFAULT_CONFIG directly, never from resolve_config's own output.
"""
from __future__ import annotations


import pytest
import yaml


def test_hand_built_dict_equals_yaml(config, tmp_path):
    user = {"inputs": {"input_catalog": "cat.csv", "image_list_file": "imgs.txt",
                       "gaiaxp_synphot_csv": "xp.csv"},
            "outputs": {"work_dir": "wd"}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(user))
    a = config.resolve_config(user, tmp_path)
    b = config.load_config(p)
    assert b["config_path"] == p.resolve() and a["config_path"] is None
    a.pop("config_path"); b.pop("config_path")
    assert a == b
    # defaults reach the hand-built path (a literal fallback would not):
    d = config.DEFAULT_CONFIG
    assert a["crop"]["margin"] == d["crop"]["margin"]
    assert a["compute"]["gpu"]["solver"]["position_solver"] == d["compute"]["gpu"]["solver"]["position_solver"]
    assert a["inputs"]["input_catalog"] == (tmp_path / "cat.csv").resolve()
    assert a["outputs"]["final_catalog"] == (tmp_path / "wd" / d["outputs"]["final_catalog"]).resolve()


def test_unknown_key_rejected(config, tmp_path):
    with pytest.raises(KeyError):
        config.resolve_config({"outputs": {"work_dir": "wd"}, "nope": 1}, tmp_path)


def test_run_pipeline_resolves_only_unresolved(config, tmp_path, monkeypatch):
    import tract7dt.config as cmod
    import tract7dt.pipeline as pmod

    class Sentinel(Exception):
        pass

    def boom(*a, **k):
        raise Sentinel()

    resolved = config.resolve_config({"outputs": {"work_dir": str(tmp_path / "wd")}}, tmp_path)
    assert cmod.is_resolved_config(resolved)
    assert not cmod.is_resolved_config({"outputs": {"work_dir": "wd"}})
    monkeypatch.setattr(cmod, "resolve_config", boom)
    with pytest.raises(Sentinel):
        pmod.run_pipeline({"outputs": {"work_dir": str(tmp_path / "wd2")}})
