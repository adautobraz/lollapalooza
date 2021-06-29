import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from .general_functions import vectorize_column, get_order_hour_dict, discretize_lineup_position

@st.cache
def get_distances_df(lineups_df):

    # df = lineups_df\
    #         .loc[:, ['female_presence', 'is_br', 'career_time', 'year']]

    df = lineups_df\
            .loc[:, ['year']]

    df_genres = lineups_df['lastfm_genre_tags'].apply(pd.Series)

    df_simil = pd.concat([df_genres, df], axis=1).dropna()

    all_dists = []

    scaler = StandardScaler()

    # for y in df['year'].unique().tolist():
    df_scal = pd.DataFrame(scaler.fit_transform(df_simil.drop(columns=['year'])))

    df_coords = df_scal.copy()
    df_coords.index = df_simil.index
    df_coords.loc[:, 'year'] = df_simil['year']

    year_mean = df_coords.groupby(['year']).mean()

    df_dist = pd.DataFrame(cdist(df_scal.values, year_mean.values))
    df_dist.columns = year_mean.index.tolist()
    df_dist.index = df_simil.index

    dist_df = df_dist.unstack().to_frame().reset_index()
    dist_df.columns=['year_ref', 'artist_id', 'distance']

    df_info = lineups_df.loc[:, ['artist', 'career_time', 'palco', 'year']].reset_index()
    dist_df = pd.merge(left=dist_df, right=df_info, on='artist_id')

    return dist_df

    
@st.cache
def load_data(data_path):

    data_dict = {}

    # Lineups
    lineups_raw_df = pd.read_csv(data_path/'lineup_info.csv').set_index('artist_id')

    disc_raw_df =  pd.read_csv(data_path/'discography_info.csv').set_index('artist_id')

    order_hour_dict = get_order_hour_dict(lineups_raw_df)

    lineups_df = lineups_raw_df.join(disc_raw_df, how='left')

    lineups_df = vectorize_column(lineups_df, 'lastfm_genre_tags')
    lineups_df = discretize_lineup_position(lineups_df, 5, order_hour_dict)

    lineups_df.loc[:, 'show_hour'] = lineups_df.apply(lambda x: "{} - {}".format(x['hour_start_adj'], x['hour_end_adj']), axis=1)

    lineups_df.loc[:, 'date'] = pd.to_datetime(lineups_df['date'], infer_datetime_format=True)
    lineups_df.loc[:, 'rank_day'] = lineups_df.groupby(['year'])['date'].rank(method='dense').astype(int)
    lineups_df.loc[:, 'day'] = lineups_df.apply(lambda x: "{}-{}".format(x['year'], x['rank_day']), axis=1)

    lineups_df.loc[:, 'is_br_str'] = lineups_df['is_br'].apply(lambda x: "Sim" if x == 1 else 'Não')
    lineups_df.loc[:, 'female_presence_str'] = lineups_df['female_presence'].apply(lambda x: "Sim" if x == 1 else 'Não')

    data_dict['lineups_df'] = lineups_df

    # Genre per act
    df = lineups_df['lastfm_genre_tags'].apply(pd.Series).fillna(0).stack().to_frame().reset_index()
    df.columns = ['artist_id', 'genre', 'value']

    df_info = lineups_df.loc[:, ['year', 'order_in_lineup', 'artist_name', 'female_presence', 'is_br']]
    genre_per_act_df = pd.merge(left=df, right=df_info, on='artist_id')
    genre_per_act_df.loc[:, 'genre_importance'] = 100*genre_per_act_df['value']

    data_dict['genre_per_act_df'] = genre_per_act_df

    # UMAP df
    umap_df = pd.read_csv(data_path/'umap_projection.csv').set_index('artist_id')
    data_dict['umap_df'] = umap_df

    # Distance df
    dist_df = get_distances_df(lineups_df)
    data_dict['dist_df'] = dist_df

    return data_dict


def load_color_references(lineups_df):

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
    match_color_year['2020'] = f"rgba({tone},{tone},{tone}, 1)"

    genre_words = sorted(list(lineups_df.iloc[0]['lastfm_genre_tags'].keys()))
    tag_dict_map = {genre_words[i]:main_palette[i] for i in range(0, len(genre_words))}
   
    color_references = {
        'main_palette':main_palette,
        'lolla_palette':lolla_palette,
        'match_color_year':match_color_year,
        'tag_dict_map':tag_dict_map,
        'color_dict': color_dict,
        'palette_name': palette_name
    }

    return color_references


def load_data_references(lineups_df):
    # Data definitons

    category_orders = {
        'genre':['rock', 'electro', 'indie', 'alt', 'hop', 'rap', 'house', 'pop'],
        'lineup_moment':lineups_df.sort_values(by='order_in_lineup')['lineup_moment'].unique().tolist()
    }

    col_labels = {
        'show_hour':'<i>Horário do show</i>',
        'year':'<i>Ano</i>',
        'palco':'<i>Palco</i>',
        'career_time':'<i>Anos de carreira</i>',
        'female_presence_str':'<i>Vocais femininos?</i>',
        'is_br_str':'<i>Artista nacional?</i>',
        'main_genres':'<i>Gênero musical</i>'
    }


    data_references = {
        'category_orders':category_orders,
        'col_labels':col_labels
    }

    return data_references