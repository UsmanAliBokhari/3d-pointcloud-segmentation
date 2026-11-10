"""
Writes the 21 functions from notebooks/project.ipynb into code/implementation.py, a plain
Python copy for graders and the submission zip. run.sh calls this automatically.
Do not edit implementation.py: edit the notebook, then run  python3 code/export_code.py
"""
import os
import scaffold

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'implementation.py')

HEADER = '''"""
GENERATED from notebooks/project.ipynb by code/export_code.py. DO NOT EDIT.
The 21 team functions of the Hackel et al. (2016) reproduction, in pipeline order.
Edit the notebook instead and re-run:  python3 code/export_code.py
"""
from nb_env import *  # numpy, scipy, sklearn, utils constants, @task

'''


def export(path=scaffold.NOTEBOOK, out=OUT):
    cells = scaffold.export_cells(path)
    with open(out, 'w') as f:
        f.write(HEADER + '\n\n'.join(c.strip() + '\n' for c in cells))
    return out, len(cells)


if __name__ == '__main__':
    out, n = export()
    print(f'exported {n} notebook cells -> {os.path.relpath(out, os.getcwd())}')
