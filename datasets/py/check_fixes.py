"""Run the reproducers and the resume sweeps with py/fix_patch.py applied."""
import runpy, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import fix_patch  # noqa: F401
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
