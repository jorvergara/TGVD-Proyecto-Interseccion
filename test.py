import sys
import os

# Añadir la carpeta "build" al PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(current_dir, 'build')
sys.path.append(build_path)

import build.module_name
from build.module_name import *

# Llamar a la función
print(dir(build.module_name))

