# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

# +
from shutil import copy, copytree

from pathlib import Path
import pipreqs
import os

from distutils.dir_util import copy_tree

# +
analysis_path = Path('../../analysis/lollapalooza_lineups/')

current_path = Path('./')

# +
file = 'streamlit_app.py'
copy(analysis_path/file, current_path/file)

folder = 'data/prep/'
Path(current_path/folder).mkdir(exist_ok=True, parents=True)
copy_tree(str(analysis_path/folder), str(current_path/folder))


folder = 'sources/'
Path(current_path/folder).mkdir(exist_ok=True, parents=True)
copy_tree(str(analysis_path/folder), str(current_path/folder))
# -

# ! pipreqs --force ./ > requirements.txt
