import subprocess


def test_run_tutorial():
    result = subprocess.run(["jupyter", "nbconvert", "--execute", "--inplace", "examples/devkit_tutorial.ipynb"])
    assert result.returncode == 0


if __name__ == "__main__":
    test_run_tutorial()
