"""Static checks of the Kaggle launcher: it cannot be executed without a CUDA session."""

import ast
import json
import re

from fno_darcy.config import CONFIG_DIR, REPO_ROOT, load_config

NOTEBOOK = REPO_ROOT / "kaggle" / "run_cuda.ipynb"


def _code():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_parse_and_have_no_stored_output():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []


def test_expensive_steps_are_off_by_default_and_guarded():
    code = _code()
    assert re.search(r"^RUN_FULL = False\b", code[0], re.M)
    assert re.search(r"^RUN_OFFICIAL = False\b", code[0], re.M)
    for command, flag in (("--config full --resume", "RUN_FULL"), ("--config official --resume", "RUN_OFFICIAL")):
        cells = [c for c in code if command in c]
        assert len(cells) == 1
        # the step starts only behind `is not True` and a budget comparison, never by reducing the config
        assert f"if {flag} is not True:" in cells[0]
        assert "> hours_left()" in cells[0] and "was not reduced" in cells[0]
        assert cells[0].index(f"if {flag} is not True:") < cells[0].index(command)


def test_generator_check_runs_before_any_training():
    text = "\n".join(_code())
    assert text.index("verify_generator.py --device cuda") < text.index("scripts/train.py")


def test_official_runs_only_after_full():
    cell = next(c for c in _code() if "--config official --resume" in c)
    assert "elif not full_done:" in cell


def test_archive_names():
    text = "\n".join(_code())
    for name in ("generator-check-cuda", "full", "full-eval-bundle", "official", "official-eval-bundle"):
        assert f"physicsnemo-fno-darcy-{name}.zip" in text


def test_ref_must_be_a_full_sha():
    assert 'fullmatch(r"[0-9a-f]{40}", REF)' in _code()[0]


def test_referenced_scripts_and_configs_exist():
    text = "\n".join(_code())
    for script in set(re.findall(r"scripts/\w+\.py", text)):
        assert (REPO_ROOT / script).exists(), script
    for name in set(re.findall(r"--config (\w+)", text)):
        assert (CONFIG_DIR / f"{name}.yaml").exists(), name


def test_projection_matches_the_full_config():
    cfg = load_config("full")
    text = "\n".join(_code())
    assert f"n_runs = {len(cfg.study.fno_modes) * len(cfg.study.seeds)}" in text
    assert f"* {cfg.training.max_pseudo_epochs} * n_runs" in text
    assert max(cfg.study.fno_modes) == 16 and "study.fno_modes=[16]" in text
    official = load_config("official")
    assert f"* {official.training.max_pseudo_epochs} / 3600" in text
    assert official.study.fno_modes == [12] and "official_probe/modes12_seed0" in text
