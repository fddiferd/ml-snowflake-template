"""PLTV ML Project Package"""

# Add the workspace root to Python path for imports
import sys
from pathlib import Path

# Add the workspace root (two levels up) to Python path
workspace_root = Path(__file__).parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

__version__ = "0.1.0"