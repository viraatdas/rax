"""Allow RAX to be run as a module: python -m rax"""

import sys
from .cli import main

if __name__ == '__main__':
    sys.exit(main()) 