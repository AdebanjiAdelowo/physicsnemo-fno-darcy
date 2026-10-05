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


def test_full_study_is_off_by_default_and_guarded():
    code = _code()
    assert re.search(r"^RUN_FULL = False\b", code[0], re.M)
    full_cells = [c for c in code if "--config full --resume" in c]
    assert len(full_cells) == 1 and full_cells[0].lstrip().startswith("if RUN_FULL is True:")


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
