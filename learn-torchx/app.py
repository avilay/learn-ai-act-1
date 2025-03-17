"""
To run this using the local scheduler -

```
torchx run --scheduler local_cwd utils.python --script my_app.py "APTG"
```
"""

import sys

print(f"Hello {sys.argv[1]}!")
