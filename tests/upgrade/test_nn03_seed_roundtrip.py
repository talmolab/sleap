"""Tier 0.5 regression tests: unset training seed must round-trip as ``None``.

Background
----------
sleap-nn 0.3.0 (PR #600) changed ``TrainerConfig.seed`` to default to ``42``
instead of being unset. The default ``42`` makes the train/val split RNG
deterministic, which is fine for *new* configs, but it is a behavior trap for
SLEAP's shipped training profiles: those intentionally leave the seed unset so
that each run gets a fresh random split.

In the shipped profiles the seed appears either as ``seed: null`` (most
profiles) or as a bare ``seed:`` key (``baseline.multi_class_topdown.yaml`` and
``baseline.multi_class_bottomup.yaml``), both of which YAML parses to ``None``.

The danger is *silently dropping* the unset seed somewhere in SLEAP's config
handling. If the key were dropped, a later merge against the sleap-nn schema
(whose ``seed`` default is ``42``) would resurrect ``42`` and silently make
every training run reuse the same train/val split. These tests lock in that an
empty/null seed survives as an explicit ``None`` everywhere SLEAP touches it:

  * when each shipped profile is loaded through SLEAP's config wrapper, and
  * when a ``seed=None`` config is round-tripped through save/load (YAML + JSON)
    and through the ``verify_training_cfg`` schema-merge that the GUI training
    runner performs before launching a job.
"""

import glob
import json
import os

import pytest
from omegaconf import OmegaConf

from sleap import util as sleap_utils
from sleap.gui.learning.configs import ConfigFileInfo


PROFILE_DIR = sleap_utils.get_package_file("training_profiles")
PROFILE_PATHS = sorted(glob.glob(os.path.join(PROFILE_DIR, "*.yaml")))
PROFILE_IDS = [os.path.basename(p) for p in PROFILE_PATHS]

# Profiles that ship a bare ``seed:`` (empty value) rather than ``seed: null``.
# Both forms parse to ``None`` but the empty form is the more fragile one, so we
# call it out explicitly here to make sure it stays covered.
EMPTY_SEED_PROFILES = {
    "baseline.multi_class_topdown.yaml",
    "baseline.multi_class_bottomup.yaml",
}


def test_profiles_discovered():
    """Sanity check: we actually found the shipped profiles to test against."""
    assert PROFILE_PATHS, f"No training profiles found in {PROFILE_DIR}"
    # Both the null-seed and empty-seed forms must be present in the shipped set.
    assert EMPTY_SEED_PROFILES.issubset(set(PROFILE_IDS))


def test_sleap_nn_seed_default_is_42():
    """Document the upstream default this test guards against.

    If sleap-nn ever changes ``TrainerConfig.seed`` back to an unset/``None``
    default, the silent-42 hazard disappears and this guard can be revisited.
    Until then, the default being 42 is precisely *why* an unset seed must be
    preserved as an explicit ``None`` rather than dropped.
    """
    from sleap_nn.config.trainer_config import TrainerConfig

    assert TrainerConfig().seed == 42


@pytest.mark.parametrize("profile_path", PROFILE_PATHS, ids=PROFILE_IDS)
def test_shipped_profile_seed_is_none(profile_path):
    """Every shipped profile must expose an explicit ``seed`` key equal to None.

    The key must be *present* (not missing) and equal to ``None`` -- a missing
    key would let a downstream schema merge default it to 42.
    """
    cfg = OmegaConf.load(profile_path)
    assert "seed" in cfg.trainer_config, (
        f"{os.path.basename(profile_path)} dropped the trainer_config.seed key; "
        "a missing seed would default to 42 on schema merge."
    )
    seed = OmegaConf.select(cfg, "trainer_config.seed", default="__MISSING__")
    assert seed is None, (
        f"{os.path.basename(profile_path)} has seed={seed!r}, expected None "
        "(an unset seed must stay None, not become 42)."
    )


@pytest.mark.parametrize("profile_path", PROFILE_PATHS, ids=PROFILE_IDS)
def test_configfileinfo_seed_is_none(profile_path):
    """SLEAP's own config wrapper must preserve the unset seed as None.

    ``ConfigFileInfo`` is the surface the GUI uses to load training profiles;
    for YAML it lazy-loads via ``OmegaConf.load``. This is the load half of the
    round-trip the guard cares about.
    """
    info = ConfigFileInfo(path=profile_path)
    assert info.config.trainer_config.seed is None


@pytest.mark.parametrize("profile_id", sorted(EMPTY_SEED_PROFILES))
def test_empty_seed_form_parses_to_none(profile_id):
    """A bare ``seed:`` (empty value) must parse to None, same as ``seed: null``.

    These two profiles use the empty form; verify the raw YAML really carries a
    bare ``seed:`` and that it parses to None (not the string "" and not 42).
    """
    path = os.path.join(PROFILE_DIR, profile_id)
    with open(path) as f:
        raw_lines = [ln.rstrip("\n") for ln in f if ln.strip().startswith("seed")]
    assert raw_lines, f"No seed line found in {profile_id}"
    # Exactly a bare 'seed:' with no value after it.
    assert any(ln.strip() == "seed:" for ln in raw_lines), (
        f"{profile_id} expected a bare 'seed:' line, found {raw_lines!r}"
    )
    cfg = OmegaConf.load(path)
    assert cfg.trainer_config.seed is None


def test_empty_and_null_seed_parse_equivalently():
    """The empty ``seed:`` form and the explicit ``seed: null`` form agree."""
    empty = OmegaConf.create("trainer_config:\n  seed:")
    null = OmegaConf.create("trainer_config:\n  seed: null")
    assert empty.trainer_config.seed is None
    assert null.trainer_config.seed is None
    assert empty.trainer_config.seed == null.trainer_config.seed


def test_seed_none_survives_yaml_roundtrip(tmp_path):
    """Saving a seed=None config to YAML and reloading keeps it None.

    This mirrors what the GUI runner does (``OmegaConf.save`` then later
    ``OmegaConf.load``). The on-disk representation must be an explicit
    ``seed: null`` so that the key is never dropped.
    """
    cfg = OmegaConf.load(PROFILE_PATHS[0])
    assert cfg.trainer_config.seed is None  # precondition

    out = tmp_path / "roundtrip.yaml"
    OmegaConf.save(cfg, out.as_posix())

    # The saved YAML must carry an explicit (non-empty-key) null seed.
    saved_text = out.read_text()
    assert "seed: null" in saved_text, (
        "seed must be serialized as an explicit 'seed: null', got:\n"
        + "\n".join(ln for ln in saved_text.splitlines() if "seed" in ln)
    )

    reloaded = OmegaConf.load(out.as_posix())
    assert reloaded.trainer_config.seed is None


def test_seed_none_survives_json_roundtrip():
    """A seed=None config round-trips through JSON (container -> json -> back)."""
    from sleap_nn.config.training_job_config import TrainingJobConfig

    cfg = OmegaConf.structured(TrainingJobConfig())
    cfg.trainer_config.seed = None

    container = OmegaConf.to_container(cfg, resolve=True)
    assert container["trainer_config"]["seed"] is None

    reloaded = json.loads(json.dumps(container))
    assert reloaded["trainer_config"]["seed"] is None

    # And rebuilt as an OmegaConf object the seed is still None.
    cfg_back = OmegaConf.create(reloaded)
    assert cfg_back.trainer_config.seed is None


@pytest.mark.parametrize("profile_path", PROFILE_PATHS, ids=PROFILE_IDS)
def test_verify_training_cfg_preserves_none_seed(profile_path):
    """The schema-merge the GUI runner performs must not resurrect seed=42.

    ``verify_training_cfg`` merges the loaded config onto a structured schema
    whose ``seed`` default is 42. Because the loaded config carries an explicit
    ``None``, the merge result must stay ``None`` -- this is the exact spot
    where a dropped key would silently flip back to 42.
    """
    from sleap_nn.config.training_job_config import verify_training_cfg

    cfg = OmegaConf.load(profile_path)
    verified = verify_training_cfg(cfg)
    assert verified.trainer_config.seed is None, (
        f"{os.path.basename(profile_path)}: verify_training_cfg produced "
        f"seed={verified.trainer_config.seed!r}, expected None (must not become 42)."
    )


def test_schema_merge_none_override_beats_default_42():
    """Minimal proof that an explicit None override wins over the 42 default.

    This isolates the OmegaConf merge semantics the whole guard relies on: a
    structured schema with ``seed=42`` merged with ``{seed: None}`` yields None.
    If OmegaConf ever treated None as "unset" here, every other test above would
    be moot, so we pin the behavior directly.
    """
    from sleap_nn.config.training_job_config import TrainingJobConfig

    schema = OmegaConf.structured(TrainingJobConfig())
    assert schema.trainer_config.seed == 42  # schema default is the hazard

    override = OmegaConf.create({"trainer_config": {"seed": None}})
    merged = OmegaConf.merge(schema, override)
    assert merged.trainer_config.seed is None
