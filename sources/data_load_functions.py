import pandas as pd
import streamlit as st

from .general_functions import vectorize_column, get_order_hour_dict, discretize_lineup_position

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

    return data_dict
