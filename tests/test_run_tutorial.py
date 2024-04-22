import subprocess


def test_run_tutorial():
    result = subprocess.run(["jupyter", "nbconvert", "--execute", "--inplace", "examples/devkit_tutorial.ipynb"])
    assert result.returncode == 0
