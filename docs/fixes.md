Problem:
scratch/local/velikanov/miniconda/envs/navsim/lib/python3.10/site-packages/nuplan/common/maps/nuplan_map/utils.py:413: RuntimeWarning: invalid value encountered in cast

The fix is to silence it in our code, without touching nuplan:

- In the worker initializer of fast_epdms.py, add warnings.filterwarnings("ignore", message="invalid value encountered in cast").
- Or prefix the run commands with PYTHONWARNINGS=ignore::RuntimeWarning. Child processes inherit that setting.
