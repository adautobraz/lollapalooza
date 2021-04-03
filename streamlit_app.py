# Setup
import streamlit as st

# Layout definitions
st.set_page_config(layout="wide")

import pandas as pd
from pathlib import Path
from PIL import Image
import string

from sources.data_load_functions import *
from sources.visualization_functions import *
from sources.general_functions import *

image_counter = -1

# Data definitions
data_path = Path('./data/prep')


# Load data
data_dict = load_data(data_path)
lineups_df = data_dict['lineups_df']
genre_per_act_df = data_dict['genre_per_act_df']
umap_df = data_dict['umap_df']


# Color definitions
main_palette = px.colors.qualitative.Safe
lolla_palette = ['#279D91', '#E8483E']

palette = px.colors.sequential.deep
palette_name= 'deep'
years = sorted(lineups_df['year'].unique().tolist())
match_color_year = {str(years[i]):palette[i] for i in range(0, len(years))}

tone = 200
grey = f"rgb({tone},{tone},{tone})"
red = px.colors.qualitative.Safe[1]
green = px.colors.qualitative.Bold[1]
color_dict = {'red':red, 'grey':grey, 'green':green}


# Data definitons

genre_words = list(lineups_df.iloc[0]['lastfm_genre_tags'].keys())

tag_dict_map = {genre_words[i]:main_palette[i] for i in range(0, len(genre_words))}
category_orders = {'genre':
                   ['rock', 'electro', 'indie', 'alt', 'hop', 'rap', 'house', 'pop'],
                   'lineup_moment':lineups_df.sort_values(by='order_in_lineup')['lineup_moment'].unique().tolist()
                  }

order_hour_dict = get_order_hour_dict(lineups_df)

col_labels = {
    'show_hour':'<i>Horário do show</i>',
    'year':'<i>Ano</i>',
    'palco':'<i>Palco</i>',
    'career_time':'<i>Anos de carreira</i>',
    'female_presence_str':'<i>Vocais femininos?</i>',
    'is_br_str':'<i>Artista nacional?</i>',
    'main_genres':'<i>Gênero musical</i>'
}

# Layout definitions
cols_baseline=12
image = Image.open(data_path/'image.jpg')


###### Title headline, subtitle
center = pad_cols([cols_baseline])[0]
center.image(image)

text = """
# Lollapalooza: Passado e Futuro
## O que podemos aprender com 9 anos de dados do maior festival de música alternativa do país?
Um ensaio visual por <b>Adauto Braz</b>
"""
center.markdown(text, unsafe_allow_html=True)
space_out(2)

fig = genre_year_trend(genre_per_act_df, tag_dict_map, category_orders)
image_counter, image_name = plot(center, fig, image_counter)

fig = lineups_visualizer(lineups_df, col_labels)
image_counter, image_name = plot(center, fig, image_counter)

# Genre view
fig = acts_similarity_scatter(lineups_df, umap_df, match_color_year, col_labels)
image_counter, image_name = plot(center, fig, image_counter)

fig = genre_year_position_trend(genre_per_act_df, 4, category_orders, order_hour_dict)
image_counter, image_name = plot(center, fig, image_counter)