"""
Procedural 3D asset generators.

Importing this package triggers registration of all generators
with the core registry system.
"""

# Import all modules to trigger their register() calls.
from . import rocks
from . import trees
from . import buildings
from . import terrain
from . import props
from . import furniture
from . import instruments
from . import childhood_home
