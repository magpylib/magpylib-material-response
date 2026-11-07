from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

import magpylib_material_response as m


def test_version() -> None:
    assert importlib.metadata.version("magpylib_material_response") == m.__version__


def test_fem_without_ngsolve_says_how_to_install() -> None:
    if importlib.util.find_spec("ngsolve") is not None:
        pytest.skip("NGSolve is installed")
    with pytest.raises(ImportError, match=r"magpylib-material-response\[fem\]"):
        importlib.import_module("magpylib_material_response.fem")


SKILL_DIR = (
    Path(m.__file__).parent / ".agents" / "skills" / "magpylib-material-response"
)


def _prose(path):
    """A skill file's text without its code blocks."""
    return re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)


def test_agent_skill_is_well_formed():
    """Frontmatter follows the Agent Skills spec, the skill names the shipped
    version line, and its links and anchors resolve."""
    skill = SKILL_DIR / "SKILL.md"
    head = skill.read_text(encoding="utf-8").split("---\n")[1]
    assert re.search(r"^name: magpylib-material-response$", head, re.MULTILINE)
    assert f"Magpylib-Material-Response {m.__version__.split('.')[0]}.x" in _prose(
        skill
    )
    description = re.search(r"^description: >-\n((?:  .*\n)+)", head, re.MULTILINE)
    assert description, "description must be a folded block (description: >-)"
    assert len(" ".join(description[1].split())) <= 1024
    prose = _prose(skill)
    anchors = {
        re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        for heading in re.findall(r"^#+ (.+)$", prose, re.MULTILINE)
    }
    assert set(re.findall(r"\]\(#([^)]+)\)", prose)) <= anchors


def test_agent_skill_examples_run():
    """The skill's examples run in order, each on what the ones before it made,
    and give the shapes their comments state."""
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    namespace = {}
    for code in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
        exec(code, namespace)  # noqa: S102  # pylint: disable=exec-used
        for name, shape in re.findall(
            r"^(\S+?)(?: = .*)?  # \((\d[\d, ]*)\)$", code, re.MULTILINE
        ):
            value = eval(name, namespace)  # pylint: disable=eval-used
            assert np.shape(value) == tuple(
                int(n) for n in shape.split(",") if n.strip()
            )


@pytest.mark.parametrize(
    "name",
    ["apply_demag", "apply_face_charge", "fem_solve", "fem_study", "mesh_study"],
)
def test_agent_skill_uses_old_names_only_in_its_table(name):
    """Past the frontmatter they appear once, as what was before 0.9."""
    assert _prose(SKILL_DIR / "SKILL.md").split("---\n", 2)[2].count(name) == 1


#: Public names the skill leaves out, each with its reason. Every other name a
#: public module exports must appear in the skill, so that a release cannot ship
#: a function the skill does not know.
NOT_IN_SKILL = {
    "configure_logging": "logging setup, not modelling",
    "disable_logging": "logging setup, not modelling",
    "get_dataset": "the docs' example data",
    "Comparison": "what compare returns",
    "MeshStudy": "what study_cells returns",
    "FemSolution": "what solve_fem returns",
    "FemStudy": "what study_fem returns",
    "FaceChargeStudy": "what study_face_charge returns",
}


@pytest.mark.parametrize(
    "module",
    [
        "magpylib_material_response",
        "magpylib_material_response.face_charge",
        "magpylib_material_response.convergence",
        "magpylib_material_response.fem",
    ],
)
def test_agent_skill_names_every_public_name(module):
    """Each name a public module exports is in the skill, or in NOT_IN_SKILL."""
    if module.endswith(".fem"):
        pytest.importorskip("ngsolve")
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    names = set(importlib.import_module(module).__all__) - set(NOT_IN_SKILL)
    missing = sorted(n for n in names if not re.search(rf"\b{re.escape(n)}\b", text))
    assert not missing, f"name them in the skill, or in NOT_IN_SKILL: {missing}"
