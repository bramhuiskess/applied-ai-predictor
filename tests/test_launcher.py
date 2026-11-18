"""The one-click launchers. start.sh is run for real in --dry-run mode against a controlled PATH;
start.bat cannot run here, so it only gets static checks (a human runs it on Windows)."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
posix_only = pytest.mark.skipif(os.name == "nt", reason="start.sh is for macOS and Linux")
TOOLS = ("dirname", "tr", "cksum", "cat", "mkdir", "rm", "mktemp", "sed", "cut")


@pytest.fixture
def project(tmp_path):
    """A copy of the launcher inputs, and a PATH that holds only the tools the script needs."""
    home = tmp_path / "project"
    home.mkdir()
    for name in ("start.sh", ".python-version", "pyproject.toml"):
        shutil.copy(ROOT / name, home / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in TOOLS:
        found = shutil.which(tool)
        if found:
            (bin_dir / tool).symlink_to(found)
    (tmp_path / "home").mkdir()
    return home, bin_dir, tmp_path / "home"


def run(project, *args, stdin="", inputs=None):
    home, bin_dir, user_home = project
    env = {"PATH": str(bin_dir), "HOME": str(user_home)}
    return subprocess.run([shutil.which("bash"), "start.sh", *args], cwd=home, env=env, input=stdin,
                          capture_output=True, text=True, timeout=60)


def fake_old_python(bin_dir):
    script = bin_dir / "python3"
    script.write_text('#!/bin/sh\ncase "$1" in --version) echo "Python 3.9.6";; *) exit 1;; esac\n')
    script.chmod(0o755)


def real_python(bin_dir):
    (bin_dir / "python3").symlink_to(sys.executable)


@posix_only
def test_help_lists_the_flags(project):
    result = run(project, "--help")
    assert result.returncode == 0 and "--dry-run" in result.stdout and "--no-start" in result.stdout


@posix_only
def test_old_python_would_install_uv_and_python_without_downloading(project):
    fake_old_python(project[1])
    result = run(project, "--dry-run", "--yes", "--no-start")
    assert result.returncode == 0, result.stderr
    assert "No Python 3.11 or newer found (found: Python 3.9.6)" in result.stdout
    assert "https://astral.sh/uv/install.sh" in result.stdout
    assert "would: create .venv with Python 3.12 (uv)" in result.stdout
    assert not (project[0] / ".venv").exists()


@posix_only
def test_a_good_python_needs_no_download(project):
    real_python(project[1])
    result = run(project, "--dry-run", "--yes", "--no-start")
    assert result.returncode == 0, result.stderr
    assert "Python: Python 3." in result.stdout and "astral.sh" not in result.stdout
    assert "would: create .venv with" in result.stdout and "would: pip install" in result.stdout
    assert not (project[0] / ".venv").exists()  # a dry run changes nothing


@posix_only
def test_minimal_leaves_out_the_training_extras(project):
    real_python(project[1])
    assert 'pip install -e "."' in run(project, "--dry-run", "--yes", "--no-start", "--minimal").stdout


@posix_only
def test_an_up_to_date_venv_is_left_alone(project):
    home, bin_dir, _ = project
    real_python(bin_dir)
    (home / ".venv" / "bin").mkdir(parents=True)
    (home / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (home / ".venv" / "bin" / "predictor").write_text("#!/bin/sh\n")
    (home / ".venv" / "bin" / "predictor").chmod(0o755)
    checksum = subprocess.run(["cksum"], stdin=open(home / "pyproject.toml"), capture_output=True, text=True).stdout.strip()
    (home / ".venv" / ".installed").write_text(f"{checksum} train,datasets")
    result = run(project, "--dry-run", "--yes", "--no-start")
    assert "already installed and up to date" in result.stdout and "would: pip" not in result.stdout


@posix_only
def test_answering_no_to_the_python_install_stops_with_instructions(project):
    fake_old_python(project[1])
    result = run(project, stdin="n\n")  # a real (not dry) run: it must stop before doing anything
    assert result.returncode == 1
    assert "Install Python 3.12 yourself" in result.stdout and "brew install python@3.12" in result.stdout
    assert not (project[0] / ".venv").exists()


@posix_only
def test_unknown_flag_is_an_error(project):
    assert run(project, "--nope").returncode == 2


@posix_only
def test_setup_sh_still_works_as_a_wrapper():
    text = (ROOT / "scripts" / "setup.sh").read_text()
    assert "start.sh" in text and "--no-start" in text


def test_the_scripts_are_executable_and_start_bat_is_crlf():
    if os.name != "nt":
        assert all(os.access(ROOT / n, os.X_OK) for n in ("start.sh", "start.command", "scripts/setup.sh"))
    raw = (ROOT / "start.bat").read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n"), "start.bat must have CRLF line endings"


def test_start_bat_labels_are_all_defined():
    text = "\n".join(line for line in (ROOT / "start.bat").read_text().splitlines()
                     if not line.lower().startswith("rem"))
    defined = set(re.findall(r"^:([a-z_]+)\s*$", text, re.M))
    used = set(re.findall(r"(?:goto|call)\s+:?([a-z_]+)", text, re.I)) - {"eof"}
    assert used <= defined, used - defined


def test_start_bat_questions_have_no_parentheses():
    """A ')' in a question text closes the surrounding block in cmd (the question is echoed inside one)."""
    for question in re.findall(r'call :ask "([^"]*)"', (ROOT / "start.bat").read_text()):
        assert "(" not in question and ")" not in question, question
