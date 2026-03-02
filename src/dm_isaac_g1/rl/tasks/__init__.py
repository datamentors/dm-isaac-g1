"""RL training tasks."""

import importlib
import os

# Auto-import all task subdirectories to trigger gym.register() calls
_tasks_dir = os.path.dirname(__file__)
for _name in sorted(os.listdir(_tasks_dir)):
    _path = os.path.join(_tasks_dir, _name)
    if os.path.isdir(_path) and os.path.exists(os.path.join(_path, "__init__.py")):
        importlib.import_module(f".{_name}", __name__)
