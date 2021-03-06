# Setup
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import string

from sources.data_load_functions import *
from sources.visualization_functions import *
from sources.general_functions import *


def part2(cols_baseline, data_path):

    # Data definitions
    image_counter = -1

    # Load data
    data_dict = load_data(data_path)
    lineups_df = data_dict['lineups_df']
    genre_per_act_df = data_dict['genre_per_act_df']
    umap_df = data_dict['umap_df']
    dist_df = data_dict['dist_df']

    order_hour_dict = get_order_hour_dict(lineups_df)
    lineups_2019_df = lineups_df.loc[lineups_df['year'] <= 2019]

    # Color definitions
    color_reference_dict = load_color_references(lineups_df)
    main_palette = color_reference_dict['main_palette']
    lolla_palette = color_reference_dict['lolla_palette']
    match_color_year = color_reference_dict['match_color_year']
    color_dict = color_reference_dict['color_dict']
    tag_dict_map = color_reference_dict['tag_dict_map']
    palette_name = color_reference_dict['palette_name']

    # Data definitons
    data_references = load_data_references(lineups_df)
    category_orders = data_references['category_orders']
    col_labels = data_references['col_labels']

    image = Image.open(data_path/'image.jpg')