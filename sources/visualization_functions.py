import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels
from .general_functions import format_fig, facet_prettify, hour_on_ticks, discretize_lineup_position, get_genres
import re


## Musical genre views
def genre_year_trend(genre_per_act_df, tag_dict_map, category_orders):

    df = genre_per_act_df\
            .groupby(['year', 'genre'], as_index=False)\
            .agg({'value':'mean'})

    df.loc[:, 'genre_importance'] = 100*df['value']

    fig = px.scatter(df, x='year', y='genre_importance', facet_col='genre', trendline='ols', color='genre',
                    category_orders=category_orders, color_discrete_map=tag_dict_map
                    )

    tickvals=[df['year'].min(), df['year'].max()]
    fig.update_xaxes(color='grey', title='', showticklabels=False, tickvals=tickvals)
    fig.update_xaxes(col=1, showticklabels=True,  color='grey', title='', tickangle=45)
    fig.update_yaxes(range=[0, 40], dtick=10, ticksuffix='%', color='grey')
    fig.update_yaxes(col=1, title='Importância do gênero musical', titlefont_color='black')

    fig = format_fig(fig)
    fig = facet_prettify(fig)

    fig.update_layout(
        title='Percentual de cada gênero<br>musical, tendência',
        # title_y=0,
        # title_pad=dict(b=100, t=100),
        # title_yref='paper',
        showlegend=True,
        legend_orientation = 'h',
        legend_y=-0.2,
        legend_title='Gênero<br>musical',
        margin_t=130
    )

    fig.for_each_annotation(lambda x: x.update(textangle=20))

    return fig


def genre_year_position_trend(genre_per_act_df, smooth, category_orders, order_hour_dict):

    df = genre_per_act_df.copy()
    # View, per lineup order
    df = discretize_lineup_position(df, smooth, order_hour_dict)

    df = df.groupby(['year', 'genre', 'lineup_moment'], as_index=False).agg({'value':'mean'})

    df.loc[:, 'genre_importance'] = 100*df['value']

    df.loc[:, 'color'] = df['lineup_moment'].astype(str)
    df = df.sort_values(by=['color'])
    df.loc[:, 'size'] = 1

    palette = px.colors.sequential.Agsunset_r
    colors = [palette[i] for i in range(0, len(palette)) if i% 2 == 0]

    fig = px.scatter(df, facet_col='genre', y='genre_importance', x='year',
                    color='color', trendline='ols', height=800, size='size', size_max=2,
            category_orders=category_orders,
            color_discrete_sequence=colors
        )

    tickvals = [df['year'].min(), df['year'].max()]

    fig.update_yaxes(title='', color='grey', range=[0, 55], ticksuffix='%')
    fig.update_xaxes(title='', showticklabels=False, color='grey', tickvals=tickvals, tickangle=45)
    fig.update_xaxes(col=1, showticklabels=True)

    fig.update_layout(
        title='Percentual de cada gênero musical,<br>tendência por horário',
        showlegend=True, 
        height=500, 
        legend_orientation='h', 
        legend_title='Horário no Lineup',
        legend_y=-0.2,
        margin_t=130
    )
    fig.update_traces(marker_opacity=0.8)

    fig = format_fig(fig)
    fig.for_each_annotation(lambda x: x.update(textangle=20))

    fig = facet_prettify(fig)

    return fig


def genre_year_position_heatmap(genre_per_act_df, smooth, category_orders):
    # Position on lineup x Musical Genre
    df = genre_per_act_df.copy()

    binsize = int(100/smooth)

    df.loc[:, 'importance_order'] = df['order_in_lineup'].apply(lambda x: min([x//binsize, smooth-1]))
    df.loc[:, 'importance_order'] = df['importance_order'].apply(lambda x: "{:.0f}%-{:.0f}%".format(x*binsize, (x+1)*binsize)\
                                                                if x < smooth - 1 else "{:.0f}%-100%".format(x*binsize))
    df = df.sort_values('importance_order')

    df.loc[:, 'value'] = 100*df['value']

    df.loc[:, 'rank_on_order'] = df.groupby(['importance_order', 'genre', 'year'])['value'].rank(ascending=False)

    df_artists = df\
                    .loc[df['rank_on_order'] <= 5]\
                    .groupby(['genre', 'importance_order', 'year'])['artist_name'].apply(lambda x: '<br>'.join(x))\
                    .to_frame().reset_index()

    df = df\
            .groupby(['genre', 'importance_order', 'year'], as_index=False)\
            .agg({'value':'mean'})

    df = pd.merge(left=df, right=df_artists, on=['year', 'genre', 'importance_order'], how='left')


    fig = px.density_heatmap(df, animation_frame='genre', z='value', y='importance_order',
                        hover_data=['artist_name'], category_orders=category_orders,
                        x='year',color_continuous_scale=px.colors.sequential.RdPu, 
                        nbinsx=9, range_color=[0,50])

    fig = format_fig(fig)

    fig.update_layout(
        title='Evolução da relevância de cada gênero musical por horário do Lineup',
        coloraxis_colorbar_title='Participação<br>', 
        coloraxis_colorbar_ticksuffix='%',
        coloraxis_colorbar_tickvals=[0, 15, 30, 45])

    return fig


def genre_per_artist(genre_per_act_df, tag_dict_map):
    # Exemplos 
    sample_size = 6
    df = genre_per_act_df\
            .loc[genre_per_act_df['order_in_lineup'] > 90].copy()

    artist_ids = df.groupby(['year', 'artist_id'], as_index=False)\
                .agg({'genre':'sum'}).sample(sample_size, random_state=42)['artist_id'].tolist()

    df = df.loc[df['artist_id'].isin(artist_ids)]

    df.loc[:, 'genre_rank'] = df.groupby(['artist_id'])['genre_importance'].rank(ascending=False)

    df.loc[:, 'x_end'] = df.sort_values(by='genre_rank').groupby(['artist_id'])['genre_importance'].cumsum()
    df.loc[:, 'x_start'] = df.sort_values(by='genre_rank').groupby(['artist_id'])['x_end'].shift(1).fillna(0)
    df.loc[:, 'delta'] = df['x_end'] - df['x_start']
    df.loc[:, 'text'] = df['delta'].apply(lambda x: "{:.0f}%".format(x) if x > 10 else  '')
    df.loc[:, 'y_axis'] = df.apply(lambda x: "{}<br>({:.0f})".format(x['artist_name'], x['year']), axis=1)

    df = df.sort_values(by=['year'])

    fig = px.timeline(df, y='y_axis', x_start='x_start', x_end='x_end', color='genre',
                    color_discrete_map=tag_dict_map, hover_data=['delta'], text = 'text'
                    )

    fig.update_xaxes(type = 'linear', color='grey', tickvals=[0,100], ticksuffix='%')
    fig.update_yaxes(title='', color='grey')
    fig.update_traces(insidetextanchor='middle')


    for i in range(0, len(fig.data)):
    #     rank_day=int(re.findall('rank_day=([0-9])', fig.data[i].hovertemplate)[0])
        genre= fig.data[i].legendgroup
        aux = df.loc[(df['genre'] == genre)]
        fig.data[i].x = aux.delta.tolist()

    fig = format_fig(fig)

    fig.update_layout(
        title='Gênero musical<br>de alguns headliners',
        margin_t=120,
        legend_orientation='h', 
        legend_title='Gênero'
    )

    return fig

## Gender views
def gender_year_bar(lineups_df, colors_dict):
    # Números absolutos
    df = lineups_df.copy()

    df.loc[:, 'female_pr'] = 'Não'
    df.loc[df['female_presence'] == 1, 'female_pr'] = 'Sim'

    df = df\
            .reset_index()\
            .groupby(['year', 'female_pr', 'female_presence'], as_index=False)\
            .agg({'artist_id':'nunique'})

    df.loc[:, 'text'] = df['artist_id']
    df.loc[df['female_presence'] == 1, 'text'] = df['artist_id'].apply(lambda x: "<b>{}</b>".format(x))


    fig = px.bar(df, x='year', y='artist_id', color='female_pr', barmode='group', text='text',
                    color_discrete_map = {'Sim':colors_dict['red'], 'Não':colors_dict['grey']}
                    )
                
    fig.update_yaxes(range=[0, 75], tickvals=[0,70])    
    fig.update_traces(textposition='outside')    

    fig = format_fig(fig)

    fig.update_layout(
        title='Vocais femininos no Lolla, por ano',
        xaxis_title='Ano', 
        yaxis_title='Quantidade de artistas',
        legend_title='Atração com vocais femininos?',
        legend_orientation='h',
        legend_y=-0.2,
        height=400
    )

    return fig


def gender_share_year_trend(lineups_df, colors_dict):
    # When is breakeaven?
    df = lineups_df.groupby(['year', 'female_presence']).agg({'artist':'count'})
    df.loc[:, 'percentage'] = df.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))['artist']
    df.reset_index(inplace=True)

    aux_dict = {0: '% Homens', 1:'% Mulheres'}

    df.loc[:, 'gender'] = df['female_presence'].astype(bool).apply(lambda x: aux_dict[x])

    fig = px.scatter(df, x='year', y='percentage', symbol='gender', 
                    trendline='ols', color='female_presence')

    last_year = df['year'].max()
    # Extend hovertemplate
    # Female
    x = df.loc[df['female_presence'] == 1, 'year'].values.reshape(-1, 1)
    y = df.loc[df['female_presence'] == 1, 'percentage'].values

    model = LinearRegression()
    model.fit(x, y)

    year_of_equality = int(np.ceil((50 - model.intercept_)/model.coef_))

    x_pred = np.array([last_year, year_of_equality])
    y_pred = model.predict(x_pred.reshape(-1,1))

    fem_df = pd.DataFrame({'percentage':y_pred, 'year':x_pred})
    fem_df.loc[:, 'female_presence'] = True


    # Male
    x = df.loc[df['female_presence'] == 0, 'year'].values.reshape(-1, 1)
    y = df.loc[df['female_presence'] == 0, 'percentage'].values

    model = LinearRegression()
    model.fit(x, y)

    year_of_equality = int(np.ceil((50 - model.intercept_)/model.coef_))

    x_pred = np.array([last_year, year_of_equality])
    y_pred = model.predict(x_pred.reshape(-1,1))

    mal_df = pd.DataFrame({'percentage':y_pred, 'year':x_pred})
    mal_df.loc[:, 'female_presence'] = False

    proj_df = pd.concat([fem_df, mal_df], axis=0)
    proj_df = pd.concat([df, proj_df], axis=0)
    proj_df.loc[:, 'projection'] = proj_df['artist'].isnull().apply(lambda x: 'Projetado' if x else 'Até hoje')

    proj_df.loc[:, 'gender'] = proj_df['female_presence'].astype(bool).apply(lambda x: aux_dict[x])


    fig = px.scatter(proj_df, x='year', y='percentage', color='gender', 
                    trendline='ols', symbol='projection', 
                    color_discrete_map={'% Homens':colors_dict['grey'], '% Mulheres':colors_dict['red']},
                    symbol_map={'Projetado':'line-ns', 'Até hoje':'circle'})

    tickvals = [df['year'].min(), last_year, year_of_equality]
    ticktext = [f"<b>{t:.0f}</b>" if t == year_of_equality else f"{t:.0f}"  for t in tickvals]
    fig.update_xaxes(tickvals=tickvals,
                    title='', 
                    color='black',
                    tickmode='array',
                    ticktext=ticktext
                    )

    fig.update_yaxes(range=[0, 101], dtick=50, ticksuffix='%', title='Porcentagem de artistas<br> do lineup')
    fig.update_layout(
        legend_title='Gênero, período',
        legend_orientation='h',
        legend_y=-0.15,
        height=400,
        margin_t=120,
        title=f'Igualdade de gênero no Lolla?<br>Só em <b>{year_of_equality}</b>'
    )

    fig.add_vline(x=year_of_equality, line_color='grey', line_dash='dash')

    tr_line=[]
    for k, trace  in enumerate(fig.data):
            if trace.mode is not None\
            and trace.mode == 'lines'\
            and trace\
            and 'Projetado' in trace.name:
                tr_line.append(k)

                
    for id in tr_line:
        fig.data[id].update(line_dash='dot', line_width=2)
        
    fig = format_fig(fig)
    return fig


def gender_position(lineups_df, order_hour_dict, colors_dict):
    # Que horas elas tocam?
    df = lineups_df.copy()
    grey = colors_dict['grey']
    red = colors_dict['red']
    
    df.loc[:, 'gender'] = df['female_presence'].apply(lambda x: 'Mulheres' if x == 1 else  'Homens')
    palette = px.colors.qualitative.Safe
    fig = px.violin(df, x='order_in_lineup', color='gender', orientation='h',
                    color_discrete_map={'Homens':grey, 'Mulheres':red}
                )

    fig.update_traces(spanmode='soft', meanline_visible=True, side='positive',
                    fillcolor='rgba(255,255,255,0)')

    fig.for_each_trace(lambda x: x.update(line_width=3) if x.legendgroup=='Mulheres' else x.update(line_width=2))
    tone = 160
    grey = f"rgb({tone},{tone},{tone})"

    tickvals = [0, 20, 50, 75, 100]
    ticktext = hour_on_ticks(tickvals, order_hour_dict)

    fig.update_xaxes(range=[0,100], tickvals=[0, 20, 50, 75, 100], 
                    ticktext=ticktext,
                    zeroline=False, title='Momento da atração', ticksuffix='%')
    fig.add_vline(x=100, line_color=grey, line_width=1)
    fig.add_vline(x=0, line_color=grey, line_width=1)
    fig = format_fig(fig)
    fig.update_layout(
        title='Que horas elas tocam?',
        legend_title='Gênero',
        violinmode='overlay')
    
    return fig


def gender_position_year_ridge(lineups_df, match_color):

    df = lineups_df.copy()

    only_girls = df.loc[df['female_presence'] == 1].sort_values(by='year')

    fig = px.violin(only_girls, y='year', x='order_in_lineup', orientation='h',
                    color='year', color_discrete_map=match_color)

    fig.update_traces(side='positive', width=1.5, spanmode='soft', meanline_visible=True, opacity=1)

    fig.update_xaxes(range=[0,100], dtick=25, title='Momento da atração', tickvals=[0, 50, 100], 
                    ticktext=['0%<br>(12h)', '50%<br>(17h)','100%<br>(22h)'])
    range_y = [df['year'].min() - 1, df['year'].max() + 1]
    fig.update_yaxes(categoryorder='category ascending', tickvals=df['year'].sort_values().unique().tolist(),
                    range=range_y)
    fig.update_yaxes(col=1, title='Ano')
    fig.update_layout(
        title=f"Que horas elas tocam?<br>{df['year'].min()} a {df['year'].max()}",
        showlegend=False,
        margin_t=120,
        height=600, width=500)

    for i in range(0, len(fig.data)):
        year = fig.data[i].name
        fig.data[i].fillcolor = match_color[year]
        fig.data[i].line.color = 'grey'
        fig.data[i].line.width = 2
        fig.data[i].opacity = 0.7
        if year < '2016':
            fig.data[i].meanline.color = 'grey'
        else:
            fig.data[i].meanline.color = 'white'
            

    fig = format_fig(fig)
    return fig


def gender_share_stage_year(lineups_df, color_dict):
    # Visão por palco e dia
    df = lineups_df.copy()
    grey = color_dict['grey']
    red = color_dict['red']

    df_day = df.groupby(['year', 'day', 'palco', 'rank_day'], as_index=False)\
                .agg({'female_presence':'sum', 'artist':'count'})

    df_day.loc[:, 'female_quota'] = 100*df_day['female_presence']/df_day['artist']
    df_day.loc[:, 'equal'] = df_day['female_quota'].apply(lambda x: 'Acima de 50%' if x >= 50 else 'Abaixo')

    df_day.loc[:, 'rank'] = df_day['female_quota'].rank(method='dense',  ascending=False)

    df_day.loc[:, 'text'] = df_day.apply(lambda x: x['palco'] if x['rank'] == 1 else '', axis=1)

    fig = px.scatter(df_day, x='rank_day', y='female_quota',
                    symbol='palco', facet_col='year',
                    color='equal', text='text', hover_name='palco',
                    color_discrete_map={'Acima de 50%':red, 'Abaixo':grey}
                    )

    fig = format_fig(fig)
    fig = facet_prettify(fig)

    fig.update_traces(textposition='top center')

    fig.update_xaxes(title='', dtick=1, ticksuffix='º', range=[0.5,2.5], color='grey')

    for i in [7, 8, 9]:
        fig.update_xaxes(col=i, range=[0.5, 3.5])

    fig.update_yaxes(range=[-1, 90], tickvals=[0,50,80])

    fig.update_layout(
        title='Participação feminina, por palco, por ano',
        yaxis_title='Participação feminina',
        yaxis_ticksuffix='%',
        legend_orientation='h',
        margin_t=120,
        showlegend=False
    )

    return fig


def gender_genre_year_trend(genre_per_act_df, color_dict, category_orders):
    
    grey = color_dict['grey']
    red = color_dict['red']
    # Gênero por gênero musical
    df = genre_per_act_df\
            .groupby(['year', 'genre', 'female_presence'], as_index=False)\
            .agg({'value':'mean', 'artist_name':'count'})

    df.loc[:, 'artist_gender'] = 'Homens'
    df.loc[df['female_presence']==1, 'artist_gender'] = 'Mulheres'
    df.loc[:, 'value'] = 100*df['value']

    fig = px.scatter(df, x='year', y='value', facet_col='genre', trendline='ols', color='artist_gender',
            symbol='artist_gender',
            color_discrete_map={'Mulheres':red, 'Homens':grey},
            category_orders=category_orders)

    fig = facet_prettify(fig)
    fig.update_xaxes(tickvals=[df['year'].min(),df['year'].max()], title='', color='grey', tickangle=45, showticklabels=False)
    fig.update_xaxes(col=1, showticklabels=True)
    fig.update_yaxes(range=[-1,45], dtick=20, ticksuffix='%', title='')
    fig.update_layout(
        legend_title='Gênero', 
        title='O que elas vieram tocar?',
        margin_t=120,
        legend_orientation='h'
    )

    fig = format_fig(fig)
    return fig


def gender_national_share(lineups_df, color_dict):
    # Gênero por nacionalidade
    df = lineups_df\
            .groupby(['year', 'is_br', 'female_presence'], as_index=False)\
            .agg({'artist_name':'count'})

    df.loc[:, 'country'] = 'Internacional'
    df.loc[df['is_br']==1, 'country'] = 'Nacional'
    df.loc[:, 'total'] = df.groupby(['year', 'female_presence'])['artist_name'].transform('sum')
    df.loc[:, 'value'] = 100*df['artist_name']/df['total']

    df = df.loc[df['is_br'] == 1]

    df.loc[:, 'artist_gender'] = 'Homens'
    df.loc[df['female_presence']==1, 'artist_gender'] = 'Mulheres'

    grey = color_dict['grey']
    red = color_dict['red']

    fig = px.bar(df, x='year', y='value', color='artist_gender', 
            hover_data=['total'], barmode='group',
            color_discrete_map={'Homens':grey, 'Mulheres':red},
    )

    # fig = facet_prettify(fig)
    fig.update_xaxes(dtick=1, title='Ano', color='grey')
    fig.update_yaxes(range=[-1,101], dtick=100, ticksuffix='%', color='grey',
                    title='Taxa de artistas nacionais')
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(
        legend_title='Gênero', 
        title='Quantos % de cada gênero são de artistas nacionais?',
        margin_t=120,
        legend_orientation='h')

    fig = format_fig(fig)
    return fig


# Overall views
def lineups_visualizer(lineups_df, labels):
    df = lineups_df.copy().fillna('')

    hover_data = {
        'show_hour':True, 
        'palco':True, 
        'career_time':True,
        'x_start':False,
        'x_end':False,
        'y_rel':False,
        'rank_day':False,
        'female_presence_str':True,
        'is_br_str':True
    }

    df.loc[:, 'date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

    df.loc[:, 'rank_day'] = df.groupby(['year'])['date'].rank(method='dense').astype(int)

    df.loc[:, 'stages'] = df.groupby(['year'])['palco'].transform('nunique')

    stages = df['palco'].unique().tolist()
    df.loc[:, 'stage_id'] = df['palco'].apply(lambda x: stages.index(x))

    df.loc[:, 'rank_stages'] = df.groupby(['year', 'date'])['stage_id'].rank(method='dense') - 1
    df.loc[:, 'max_stage'] = df.groupby(['year', 'date'])['rank_stages'].transform('max')
    df.loc[:, 'rel_pos'] = df

    df.loc[:, 'x_start'] = df['hour_start_adj'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1])/60) 
    df.loc[:, 'x_end'] = df['x_start'] + df['duration_min']/60
    df.loc[:, 'delta'] = df['duration_min']/60

    df.loc[:, 'y'] = df['year'] + df['rank_stages']/df['stages']/1.5
    df.loc[:, 'max_stage'] = df.groupby(['year', 'date'])['y'].transform('max')
    df.loc[:, 'min_stage'] = df.groupby(['year', 'date'])['y'].transform('min')

    df.loc[:, 'y_rel'] = df['y']- (df['max_stage']-df['min_stage'])/2


    fig = px.timeline(df, y='y_rel', x_start='x_start', x_end='x_end', custom_data=['rank_day'],
                    hover_name='artist_name', hover_data=hover_data, labels=labels,
                    facet_col='rank_day', color='palco', color_discrete_sequence=px.colors.qualitative.Pastel
                    )

    fig.update_xaxes(type = 'linear', tickvals=[12, 17, 22], color='grey')
    fig.update_traces(width=0.1)
    range_year = [df['year'].min() - 0.5, df['year'].max() + 0.5]
    fig.update_yaxes(range=range_year, title='', color='grey')

    for i in range(0, len(fig.data)):
        rank_day=int(fig.data[i].customdata[0][0])
        stage= fig.data[i].legendgroup
        aux = df.loc[(df['rank_day'] == rank_day) & (df['palco'] == stage)]
        fig.data[i].x = aux.delta.tolist()

    fig = format_fig(fig)

    fig = facet_prettify(fig)
    fig.for_each_annotation(lambda x: x.update(text = f'Dia {x.text}'))

    fig.update_layout(
        title='Todos os lineups',
        height=600, 
        legend_orientation='h', 
        legend_title='Palco',
        legend_y=-0.1
    )

    return fig


def acts_similarity_scatter(lineups_df, umap_df, match_color_year, labels):
    # UMAP
    df = pd.concat([umap_df, 
                        lineups_df.loc[:, ['artist', 'year', 'palco', 'is_br_str', 'show_hour',
                                            'female_presence_str', 'career_time', 'lastfm_genre_tags']]], axis=1)

    df.loc[:, 'main_genres'] = df['lastfm_genre_tags'].apply(lambda x: get_genres(x))
    df.loc[:, 'size'] = 1
    df.loc[:, 'year'] = df['year'].astype(str)

    hover_data = {
        'show_hour':True,
        'palco':True,
        'is_br_str':True,
        'female_presence_str':True,
        'career_time':True, 
        'main_genres':True,
        'dim0':False,
        'dim1':False,
        'size':False
    }

    fig = px.scatter(df.sort_values(by='year'), x='dim0', y='dim1', color='year', 
                    hover_name='artist', size='size', size_max=7, color_discrete_map=match_color_year,
                    hover_data=hover_data, labels=labels
                    
                    )
    fig = format_fig(fig)
    fig.update_traces(marker_line_width=0.5, marker_line_color='grey')
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, title='')
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, title='')
    fig.update_layout(
        title='Todas as atrações do Lolla,<br>agrupadas por similaridade',
        legend_title='Ano',
        legend_orientation='h'
    )
    
    return fig


def cumulative_festival_time(lineups_df, match_color_year):
    # Tempo total acumulado
    df = lineups_df\
            .groupby(['year'], as_index=False)\
            .agg({'duration_min': 'sum'})

    df.loc[:, 'duration_h'] = df['duration_min']/60
    df.loc[:, 'duration_days'] = df['duration_h']/24
    df.loc[:, 'text'] = df.apply(lambda x: "<b>{:.0f}</b>: {:.0f}h".format(x['year'], x['duration_h']), axis=1)

    df.loc[:, 'x_end'] = df['duration_min'].cumsum()/60/24

    df.loc[:, 'x_start'] = df['x_end'].shift(1).fillna(0)
    df.loc[:, 'delta'] = df['x_end'] - df['x_start']
    df.loc[:, 'year'] = df['year'].astype(str)
    df.loc[:, 'y'] = 1


    fig=px.bar(df, x='y', y='duration_days', color='year', text='text',
            color_discrete_map=match_color_year)

    fig.update_traces(insidetextanchor='middle')

    fig = format_fig(fig)

    fig.update_layout(
        width=300,
        title='Tempo acumulado<br>de festival',
        margin_t=120,
        yaxis_title='',
        yaxis_showticklabels=True,
    #     yaxis_autorange='reversed',
        yaxis_color='grey',
        yaxis_dtick=3,
        yaxis_ticksuffix=' dias',
        xaxis_showticklabels=False,
        xaxis_title='',
        showlegend=False,
        legend_orientation='v',
        legend_title='Ano',
    )

    return fig


def artist_participation_distribution(lineups_df, lolla_palette):
    # Total de artistas
    df = lineups_df.reset_index().copy()
    df.loc[:, 'ids_list'] = df['artists_ids'].fillna(df['artist_name']).apply(lambda x: x.split(','))
    df = pd.DataFrame.explode(df, column='ids_list').loc[:, ['artist_id', 'ids_list', 'artist', 'date']]

    df = df\
            .groupby(['ids_list'], as_index=False)\
            .agg({'artist':'max', 'date':'nunique', 'artist_id':'nunique'})\
            .sort_values(by=['artist_id'], ascending=False)

    df = df.groupby(['artist_id'], as_index=False).agg({'ids_list':'nunique'})
    df.loc[:, 'percent'] = 100*df['ids_list']/df['ids_list'].sum()
    df
    fig = px.bar(df, x='artist_id', y='percent', color_discrete_sequence=lolla_palette)

    fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
    fig.update_yaxes(range=[0,101], tickvals=[0, 100], ticksuffix='%', color='grey', title='Porcentagem dos artistas')
    fig.update_xaxes(dtick=1, color='grey', title='Total de participações')
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')

    fig = format_fig(fig)

    fig.update_layout(
        title='Quantas vezes os<br> artistas já participaram?',
        margin_t=120
    )
    return fig


## Career time
def career_time_distribution_bar(lineups_df, lolla_palette):
    # Tempo de carreira
    df = lineups_df.loc[lineups_df['has_discography']].copy()

    group = 2
    df.loc[:, 'career_disc'] = df['career_time'].apply(lambda x: "{:.0f} - {:.0f}".format((x//group)*group, ((x//group)*group) + group - 1) if x < 20 else '20+')

    df = df\
            .groupby(['career_disc'], as_index=False)\
            .agg({'artist':'count', 'career_time':'mean'})\
            .sort_values(by=['career_time'])

    df.loc[:, 'percent'] = 100*df['artist']/df['artist'].sum()

    df.loc[:, 'rank'] = df['percent'].rank(ascending=False)
    df.loc[:, 'text'] = df.apply(lambda x: "<b>{:.1f}%</b>".format(x['percent']) if x['rank'] == 1 else "{:.1f}%".format(x['percent']), axis=1)

    fig = px.bar(df, x='career_disc', y='percent', text='text',
                color_discrete_sequence=lolla_palette
            )
    fig.update_yaxes(ticksuffix='%', color='grey', dtick=25, range=[0,30])
    fig.update_xaxes(color='grey', tickangle=45)
    fig.update_traces(textposition='outside')

    fig = format_fig(fig)
    fig.update_layout(
        title='Quanto tempo de<br>carreira têm os artistas?',
        margin_t=120,
        xaxis_title='Anos de carreira',
        yaxis_title='Porcentagem dos artistas'
    )
    
    return fig


def career_time_distribution_year_violin(lineups_df, match_color_year):
   # Tempo de carreira ao longo do tempo
    df = lineups_df.loc[lineups_df['has_discography']]

    df = df.sort_values(by=['year'], ascending=False)

    fig = px.violin(df, x='career_time', hover_name='artist_name', y='year', orientation='h',
                    color='year', color_discrete_map=match_color_year,
                    hover_data=['first_release', 'show_hour'], points='outliers'
                )

    fig.update_traces(side = 'positive', spanmode='soft', width=3,
                    meanline_visible=True)

    years = df['year'].unique().tolist()
    fig.update_yaxes(dtick=1, tickvals=years, gridwidth=2, color='grey')
    fig.update_xaxes(range=[0, 40], color='grey', tickvals=[0, 6, 10, 20, 30, 40])

    fig.update_layout(
        width=400,
        title='Tempo de carreira dos artistas,<br>distribuição por ano',
        margin_t=120,
        xaxis_title='Anos de carreira',
        yaxis_title='',

        showlegend=False
    )

    for i in range(0, len(fig.data)):
        year = fig.data[i].name
        fig.data[i].fillcolor = match_color_year[year]
        fig.data[i].marker.color = match_color_year[year]
        fig.data[i].marker.line.width = 1
        fig.data[i].marker.line.color = 'black'
        fig.data[i].line.width = 1
        fig.data[i].meanline.width = 2
        fig.data[i].opacity = 0.7
        if year < '2014':
            fig.data[i].line.color = 'grey'
            fig.data[i].meanline.color = 'grey'
        else:
            fig.data[i].line.color = 'white'
            fig.data[i].meanline.color = 'white'
            
    fig = format_fig(fig)
    return fig


def career_time_per_year_line(lineups_df, lolla_palette, color_dict):
    df = lineups_df.copy()
    df = df.groupby(['year'])\
            .agg({'career_time':['mean', 'median']})\
            .droplevel(0, axis=1)\
            .reset_index()

    df.loc[:, 'min'] = df['mean'].min()
    df.loc[:, 'max'] = df['mean'].max()
    df.loc[:, 'is_highlight'] = df.apply(lambda x: x['mean'] in [x['min'], x['max']], axis=1)
    df.loc[:, 'text'] = df.apply(lambda x: "{:.1f}".format(x['mean']) if x['is_highlight'] else '', axis=1)
    df.loc[:, 'x_axis'] = df.apply(lambda x: "<b>{:.0f}</b>".format(x['year']) if x['is_highlight'] else "{:.0f}".format(x['year']), axis=1)

    fig = px.line(df, x='year', y='mean', text='text',
                color_discrete_sequence=lolla_palette)

    fig = format_fig(fig)
    fig.update_yaxes(rangemode='tozero', color='grey', range=[-0.2, 11],
                    dtick=2,
                    title='Tempo médio de carreira')

    years = df['year'].unique().tolist()
    years_text = df['x_axis'].tolist()
    fig.update_xaxes(tickvals=years, ticktext=years_text, tickangle=45,
                    range=[min(years)-1, max(years)+1],
                    showgrid=False,
                    title='Ano do festival')

    fig.update_traces(mode='lines+markers+text', textposition='bottom left')
    # fig.update_traces(texttemplate='%{y:.1f}', textposition='top center')

    highlights = df.loc[df['is_highlight'], 'year'].tolist()
    for h in highlights:
        fig.add_vline(x=h, line_color=color_dict['grey'], 
                    line_dash='dash', line_width=1)

    fig.update_layout(
        title='Tempo médio de carreira,<br>por ano',
        margin_t=120
    )
    return fig


def career_time_position_bar(lineups_df, lolla_palette):
    df = lineups_df.copy()
    df = df.groupby(['lineup_moment'])\
            .agg({'career_time':['mean', 'median']})\
            .droplevel(0, axis=1)\
            .reset_index()

    fig = px.bar(df, y='lineup_moment', x='mean', color_discrete_sequence=lolla_palette)
    fig = format_fig(fig)
    fig.update_xaxes(tickvals=[], title='Anos de carreira',
                    color='grey')
    fig.update_yaxes(title='Horário no lineup', color='grey')
    fig.update_traces(texttemplate='%{x:.1f}', textposition='inside')
    fig.update_layout(
        title='Quanto tempo de carreira<br>para tocar em cada horário?',
        margin_t=120
    )
    return fig


def career_time_per_position_heatmap(lineups_df, category_orders, palette_name):
    # Tempo desde o último álbum - Posição no Lineup e Anos
    df = lineups_df.copy()

    fig = px.density_heatmap(lineups_df, x='year', y='lineup_moment', z='career_time',
                            category_orders=category_orders,
                    histfunc='avg', nbinsx=9, nbinsy=10, color_continuous_scale=palette_name)
    fig = format_fig(fig)
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        title='Quanto tempo demora<br>para ser headliner?',
        margin_t=120,
        coloraxis_colorbar_title='Anos de<br>carreira',
        yaxis_title='Horário no Lineup',
        xaxis_title='Ano do festival',
        xaxis_color='grey',
        yaxis_color='grey'
    )
    return fig
    #lineups_dsc.head()


## Time since latest release
def last_release_time_distribution_bar(lineups_df, lolla_palette):
    # Tempo desde o último lançamento 
    df = lineups_df.loc[lineups_df['has_discography']].copy()

    df.loc[:, 'disc'] = df['time_since_latest_release'].apply(lambda x: "{:.0f}".format(x) if x < 5 else '5+')

    df = df.groupby(['disc'], as_index=False).agg({'artist':'count'})
    df.loc[:, 'percent'] = 100*df['artist']/df['artist'].sum()
    df.loc[:, 'rank'] = df['percent'].rank(ascending=False)
    df.loc[:, 'text'] = df.apply(lambda x: "<b>{:.1f}%</b>".format(x['percent']) if x['rank'] == 1 else "{:.1f}%".format(x['percent']), axis=1)

    df
    fig = px.bar(df, x='disc', y='percent', text='text',
                    color_discrete_sequence=lolla_palette[1:]
                )

    fig.update_traces(textposition='outside')
    fig.update_yaxes(range=[0,66], tickvals=[0, 65], ticksuffix='%', color='grey', title='Porcentagem do Lineup')
    fig.update_xaxes(type='category', title='Tempo desde o último lançamento', color='grey')
    fig = format_fig(fig)
    fig.update_layout(
        title='Quanto tempo demora para o artista<br>tocar depois do último lançamento?',
        margin_t=120,
        xaxis_title='Anos'
    )
    
    return fig

