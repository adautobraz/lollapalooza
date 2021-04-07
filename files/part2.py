# Setup
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import string

from sources.data_load_functions import *
from sources.visualization_functions import *
from sources.general_functions import *


def part2(cols_baseline):

    # Data definitions
    data_path = Path('./data/prep')
    image_counter = -1

    # Load data
    data_dict = load_data(data_path)
    lineups_df = data_dict['lineups_df']
    genre_per_act_df = data_dict['genre_per_act_df']
    umap_df = data_dict['umap_df']
    order_hour_dict = get_order_hour_dict(lineups_df)


    # Color definitions
    color_reference_dict = load_color_references(lineups_df)
    main_palette = color_reference_dict['main_palette']
    lolla_palette = color_reference_dict['lolla_palette']
    match_color_year = color_reference_dict['match_color_year']
    color_dict = color_reference_dict['color_dict']
    tag_dict_map = color_reference_dict['tag_dict_map']


    # Data definitons
    data_references = load_data_references(lineups_df)
    category_orders = data_references['category_orders']
    col_labels = data_references['col_labels']


    ###### Title headline, subtitle
    center = pad_cols([cols_baseline])[0]
    center.video('https://www.youtube.com/watch?v=yaRC6vTGkFs&t=6s&ab_channel=LollapaloozaBrasil')

    text = """
    # Lollapalooza: A Cena Nacional
    ## O que as 8 edições do Lolla podem nos contar sobre o estado atual da música brasileira?
    Um ensaio visual por <b>Adauto Braz</b>
    """
    center.markdown(text, unsafe_allow_html=True)
    space_out(2)

    # fig = genre_year_trend(genre_per_act_df, tag_dict_map, category_orders)
    # image_counter, image_name = plot(center, fig, image_counter)