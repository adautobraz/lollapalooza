import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import string

global image_counter 
image_counter = -1

## Streamlit Layout
def space_out(space):
    for i in range(0, space):
        st.text('')


def pad_cols(col_list):
    new_list = [1, *col_list, 1]
    all_cols = st.beta_columns(new_list)
    return all_cols[1:-1]


## Plotly Layout 
def plot(streamlit_el, fig):
    general_config ={'displayModeBar':False}
    global image_counter
    image_counter += 1
    image_name = string.ascii_uppercase[image_counter]
    image_ref = f"<b>{image_name}</b>"
    old_title = fig.layout.title.text
    new_title = f"{image_ref} - {old_title}"
    fig.update_layout(title=new_title)
    streamlit_el.plotly_chart(fig, use_container_width=True, config=general_config)
    
    return image_ref


def format_fig(fig):
    fig.update_layout(
        font_family='Helvetica', 
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin_r=0,
        margin_l=0
    )
    fig.update_xaxes(fixedrange = True)
    fig.update_yaxes(fixedrange = True)

    return fig


def facet_prettify(fig_raw, capitalize=True):
    if capitalize:
        fig = fig_raw.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1].replace('_', ' ').title()))
    else:
        fig = fig_raw.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1]))
    return fig


def leave_only_slider(fig):
    fig['layout']['updatemenus']=[]
    fig['layout']['sliders'][0]['x']=0
    fig['layout']['sliders'][0]['len']=1
    return fig

## Miscellaneous
def break_text(text, limit=15):
    new_text = ''
    words = text.split(' ')
    line = ''
    for w in words:
        if len(line + ' ' + w) <= limit:
            line = f"{line} {w}"
        else:
            new_text += f"{line}<br>"
            line = w
    
    new_text += " " + line
    return new_text.strip()


def vectorize_column(raw_df, col):
    df = raw_df.copy()
    df.loc[:, col] = df[col].fillna("{}").apply(lambda x: eval(x))
    return df


def palette_year_match(palette, lineups_df):
    palette = [f for f in palette.split('\n') if f]
    years = sorted(lineups_df['year'].unique().tolist())
    match_color = {str(years[i]):palette[i] for i in range(0, len(years))}
    return match_color


def hour_on_ticks(vals, order_hour_dict):
    new_ticks = []
    for v in vals:
        hour = order_hour_dict[v]
        tick = f"{v}%<br>({hour})"
        new_ticks.append(tick)
    return new_ticks


def get_order_hour_dict(lineups_df):
    df = lineups_df.copy()

    df.loc[:, 'start'] = df['hour_start_adj'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1])/60)
    df.loc[:, 'disc'] = (np.floor(df['order_in_lineup']/5)*5).astype(int)
    df = df.groupby(['disc']).agg({'start':'median'})
    df.loc[:, 'min'] = (((100*df['start'] % 100) * 60/100)//10)*10
    df.loc[:, 'hour'] = df.apply(lambda x: "{:.0f}:{:02.0f}".format(x['start']//1,  x['min']), axis=1)
    order_hour_dict=df['hour'].to_dict()
    
    return order_hour_dict


def discretize_lineup_position(raw_df, categories, order_hour_dict):
    df = raw_df.copy()

    binsize = int(100/categories)

    df.loc[:, 'start'] = (df['order_in_lineup'].apply(lambda x: min([x//binsize, categories-1]))*binsize).astype(int)
    df.loc[:, 'end'] = (df['start'] + binsize).astype(int)

    df.loc[:, 'start_adj'] = df['start'].apply(lambda x: order_hour_dict[x])
    df.loc[:, 'end_adj'] = df['end'].apply(lambda x: order_hour_dict[x])

    df.loc[:, 'lineup_moment'] = df.apply(lambda x: "{} - {}".format(x['start_adj'], x['end_adj']), axis=1)

    df = df.drop(columns=['start', 'end', 'start_adj', 'end_adj'])
    return df


def get_genres(x):
    text = ''
    if x != np.nan:
        for g, v in dict(sorted(x.items(), key=lambda item: item[1], reverse=True)).items():
            if v > 0:
                text += f'<br>{g.title()}: {100*v:.1f}%'
        return text
    else:
        return ''