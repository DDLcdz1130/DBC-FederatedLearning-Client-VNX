
import sys
sys.path.append("..")

import os
print(os.getcwd())
os.chdir(sys.path[0])
print(os.getcwd())

from modules import learning


