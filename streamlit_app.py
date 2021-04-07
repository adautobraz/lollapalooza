# Setup
import streamlit as st

# Layout definitions
st.set_page_config(layout="wide")

import pandas as pd
from pathlib import Path
from PIL import Image
import string
from files import SessionState

from sources.data_load_functions import *
from sources.visualization_functions import *
from sources.general_functions import *
from files.part1 import part1
from files.part2 import part2

# Page params
query_params = st.experimental_get_query_params()
app_state = st.experimental_get_query_params()

session = SessionState.get(
    view=1, first_query_params=st.experimental_get_query_params()
)
first_query_params = session.first_query_params
default_values = {
    "view": int(session.first_query_params.get("parte", [0])[0]),
}

session.view = default_values['view']


# Layout definitions
cols_baseline=10

# Layout
parts = {1: 'Passado e Futuro (Parte 1)', 2:'A Cena Nacional (Parte 2)', 3:'Mulheres na Música (Parte 3)'}

if session.view == 1:
    part1(cols_baseline)

if session.view == 2:
    part2(cols_baseline)

other_parts = {k: v for k,v in parts.items() if k != session.view}
# for k, v in other_parts.items():
    
#     # text = """
#     #     <a href="https://share.streamlit.io/adautobraz/lollapalooza/?parte={}">{}</a>
#     # """.format(k, v)

#     text = """
#         <a href="http://localhost:8501/?parte={}">{}</a>
#     """.format(k, v)     
#     st.markdown(text, unsafe_allow_html=True)