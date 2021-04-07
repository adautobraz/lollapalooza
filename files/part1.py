# Setup
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import string

from sources.data_load_functions import *
from sources.visualization_functions import *
from sources.general_functions import *


def part1(cols_baseline, data_path):

    # Data definitions
    image_counter = -1

    # Load data
    data_dict = load_data(data_path)
    lineups_df = data_dict['lineups_df']
    genre_per_act_df = data_dict['genre_per_act_df']
    umap_df = data_dict['umap_df']
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

    ###### Title headline, subtitle
    center = pad_cols([cols_baseline])[0]
    center.image(image)

    text = """
    # Lollapalooza: Passado e Futuro (Parte 1)
    ## Uma série de três partes sobre os dados de 8 anos do maior festival de música alternativa do país
    Um ensaio visual por <b>Adauto Braz</b>
    <br><br><br>
    À luz da nossa atual realidade, a ideia de estar num ambiente fechado, rodeado de estranhos, 
    com corpos praticamente desrespeitando leis da física, parece um pesadelo. 
    Mas se você é como eu, sem dúvida sente falta da experiência de vivenciar um show ao vivo.  

    Cantar a plenos pulmões junto de uma multidão, sentir o reverberar do som no seu corpo, 
    chorar ao lado de desconhecidos. Estas, sensações que estamos há mais de um ano sem poder 
    experimentar, são algumas das que seguimos mais ansiosos para poder reviver.

    Reviver inclusive por volta dessa mesma época do ano, um período que costuma ser um dos 
    momentos mais esperados por amantes da música no Brasil todo, quando acontece o Lollapalooza. 
    Com sua primeira edição em 2012, o festival, que apela para um público de música mais 
    alternativa, tem tido público de mais de 200 mil pessoas nas últimas edições, 
    mas segue, assim como tantos outros, com rumos incertos sobre sua próxima edição frente 
    aos rumos da pandemia. 

    Embora 2020 fosse minha primeira vez no Lolla, tenho acompanhado de longe as impressões 
    gerais sobre o que tem acontecido com o festival nos últimos anos, em especial com o 
    perfil das atrações. Para além disso, discussões sobre igualdade de gênero em grandes 
    festivais e o florescimento da cena alternativa nacional trazem questionamentos interessantes, 
    que deveríamos entender melhor para entender os rumos futuros do festival.

    Na tentativa de matar um pouco a saudade do que não pudemos viver em 2020, 
    e nos ajudar a pensar sobre este possível futuro, vamos explorar, 
    com dados do Line-Up de 2012 a 2019, e em alguns casos até o de 2020, o que tem acontecido 
    até aqui com o Lollapalooza e, quem sabe, propor ajustes de rota para a 
    tão esperada próxima edição do festival.    
    """
    center.markdown(text, unsafe_allow_html=True)
    # space_out(1)


    # Lineup view and UMAP view
    center = pad_cols([cols_baseline])[0]

    fig = lineups_visualizer(lineups_2019_df, col_labels)
    image_name = get_image_name(image_counter)

    text = """
    ### <b>Como foi o lineup mesmo?</b>
    Mesmo tendo ido para todas as edições, é bem possível que você não lembre no detalhe de 
    todos os artistas que já tocaram ou como foi o lineup de cada ano. 
    Por isso, segue abaixo uma visão interativa de todos os lineups, em {}. 
    Para mais detalhes, basta clicar em cada elemento.
    """.format(image_name)
    center.markdown(text, unsafe_allow_html=True)

    image_counter = plot(center, fig, image_counter)
    space_out(2)

    pad = 6
    left, right = pad_cols([pad, cols_baseline - pad])

    fig = cumulative_festival_time(lineups_df, match_color_year)
    image_name_bar = get_image_name(image_counter)
    image_counter = plot(right, fig, image_counter)

    fig = acts_similarity_scatter(lineups_df, umap_df, match_color_year, col_labels)
    image_name_umap = get_image_name(image_counter)

    text = """
    Se somarmos o tempo total de música ao vivo que já foi tocada no festival, até 2019, 
    caso alguém quisesse assistir a todas as performances já realizadas, precisaria de cerca 
    de 18 dias, sem parar, como mostrado em {}.

    Para atingir esse total de horas de música, o festival já contou, até 2019, 
    com mais de 380 artistas, de diferentes nacionalidades e estilos musicais. 
    Abaixo, em {},  temos uma visão de todos os artistas que já tocaram no festival, 
    agrupados por similaridade - e os que iriam tocar em 2020 também.

    Para o gráfico de similaridade utilizamos quatro características principais: 
    nacionalidade da atração (brasileira ou internacional), se há leads femininos, tempo de carreira
    e os gêneros musicais associados a cada artista 
    (mais detalhes sobre isso nas sessões à frente).
    """.format(image_name_bar, image_name_umap)
    left.markdown(text, unsafe_allow_html=True)

    center = pad_cols([cols_baseline])[0]
    image_counter = plot(center, fig, image_counter)

    text = """
    No gráfico acima, pontos próximos são mais similares entre si. 
    Explorando os dois grandes grupos que surgem de maneira orgânica nos dados, 
    é possível ver dois tipos de atrações principais: no grupo superior, os artistas de Hip-hop, 
    Rap e Eletrônica, e no grupo inferior, os artistas Indie, Rock, Pop e Alternativo.
    """
    center.markdown(text, unsafe_allow_html=True)


    # Career
    center = pad_cols([cols_baseline])[0]
    
    pad = 4
    left, right = pad_cols([pad, cols_baseline - pad])

    fig = artist_participation_distribution(lineups_2019_df, lolla_palette)
    part_name = get_image_name(image_counter)
    image_counter = plot(left, fig, image_counter)

    fig = career_time_distribution_bar(lineups_2019_df, lolla_palette)
    career_name = get_image_name(image_counter)
    image_counter = plot(right, fig, image_counter)

    text = """
    ### <b>Artistas e tempo de carreira </b>
    De todos os artistas escalados até 2019, apenas 12,4% repetiram sua aparição em mais de 
    uma edição do festival ({}). Entre os artistas que tocaram mais vezes, temos Steve Aoki, 
    Chemical Surf, Cage the Elephant e Vintage Culture, todos com três passagens pelo lineup.
    <br><br>
    Ao considerarmos o tempo de carreira dos artistas, em {}, observamos que a maioria 
    dos artistas que tocam no festival são artistas em início de carreira - com 4 a 5 anos 
    desde o lançamento do seu primeiro single ou álbum. Há, todavia, uma participação considerável de artistas mais consolidados, 
    com cerca de 6,8 % dos artistas que já tocaram no festival com mais de 20 anos de carreira.
    """.format(part_name, career_name)
    center.markdown(text, unsafe_allow_html=True) 


    center = pad_cols([cols_baseline])[0]
    pad = 5
    left, right = pad_cols([pad, cols_baseline-pad])
    fig = career_time_per_year_line(lineups_2019_df, lolla_palette, color_dict)
    line_name = get_image_name(image_counter)
    image_counter = plot(left, fig, image_counter)

    fig = career_time_distribution_year_violin(lineups_df, match_color_year)
    viol_name = get_image_name(image_counter)
    image_counter = plot(right, fig, image_counter)

    text = """
    Ao longo do tempo ({}), observamos que o tempo de carreira média das atrações do festival tem 
    variado entre 6 e 10 anos, com destaque para os anos de 2012 e 2015 como as edições mais 
    madura e mais jovem, respectivamente. Em {}, podemos observar, ano a ano, como fica a distribuição do tempo 
    de carreira, sendo possível observar quais atrações, para o ano em questão, 
    são pontos fora da curva.
    """.format(line_name, viol_name)
    center.markdown(text, unsafe_allow_html=True) 


    center = pad_cols([cols_baseline])[0]
    pad = 4
    left, right = pad_cols([pad, cols_baseline-4])
    fig = career_time_position_bar(lineups_2019_df, lolla_palette)
    bar_name = get_image_name(image_counter)
    image_counter = plot(left, fig, image_counter)

    fig = career_time_per_position_heatmap(lineups_2019_df, category_orders, palette_name)
    heat_name = get_image_name(image_counter)
    image_counter = plot(right, fig, image_counter)

    text = """
    Dado que, como podemos ver até aqui, existem diferentes perfis de artista no festival,
    teria o tempo de carreira de cada atração alguma relação com o horário em que eles tocam?
    Em {}, vemos que existe sim uma relação entre essas características, com os horários mais
    no final do dia trazendo, na média, artistas com mais tempo de carreira do que os outros horários.

    Apesar disso, como é possível observar em {}, ano a ano, não há necessariamente
    uma progresão linear entre horário e tempo de carreira, com edições em que os artistas 
    mais maduros tocam no meio do dia, como em 2016 e 2017. Nesses anos, como destaque 
    desses horários, tocaram Bad Religion às 16:10, em 2016, e Duran Duran às 16:30, em 2017.

    Por último, um questionamento comum para os fãs todo ano é, quanto tempo depois do último
    lançamento do meu artista favorito posso esperar que ele apareça por aqui? Analisando

    """.format(bar_name, heat_name)
    center.markdown(text, unsafe_allow_html=True) 
  
    center = pad_cols([cols_baseline])[0]

    fig = last_release_time_distribution_bar(lineups_2019_df, lolla_palette)
    image_name = get_image_name(image_counter)

    text = """
    Por último, um questionamento comum para os fãs todo ano é, quanto tempo depois do último
    lançamento de um artista posso esperar até que ele apareça por aqui? Considerando o histórico
    do festival, mais da metade dos artistas não demora nem um ano completo até aparecer no lineup do festival,
    como indica {}.

    Este talvez seja o sinal de que, caso seu artista favorito lance um novo projeto, 
    vale a pena manter as esperanças por até duas edições. Depois disso, as chances se reduzem
    consideravelmente.
    """.format(image_name)
    center.markdown(text, unsafe_allow_html=True) 

    image_counter = plot(center, fig, image_counter)


    # # Genre view  
    center = pad_cols([cols_baseline])[0]

    df = genre_per_act_df.loc[genre_per_act_df['year'] <= 2019]
    fig = genre_per_artist(df, tag_dict_map)
    image_name = get_image_name(image_counter)

    text = """
    ### <b>Gêneros musicais</b>
    Para entender os diferentes gêneros musicais presentes no lineup, podemos usar dados 
    encontrados no LastFM. Nele, cada artista tem um conjunto de tags associadas, 
    como por exemplo alt-rock, indie-pop, indie, rock, house, pop, etc.

    Extraindo o conjunto de termos mais comuns entre todas as tags possíveis, 
    identificamos que existem cerca de 8 termos que sumarizam a maior parte das tags, 
    são estes: Rock, Indie, Alt, Electro, Pop, House, Rap e Hop. Apesar de haver uma quantidade 
    bem maior de gêneros musicais possíveis para descrever um artista, 
    vamos nos ater a esse conjunto de termos.

    Podemos, então, contar a frequência de aparição de cada um dos termos
    entre as tags de cada artista, chegando, ao final, a um nível percentual de quanto cada 
    artista é associado com cada um desses termos. Alguns exemplos podem ser observados no gráfico
    {}.
    """.format(image_name)
    center.markdown(text, unsafe_allow_html=True) 

    image_counter = plot(center, fig, image_counter)

    # Genre evolution views
    center = pad_cols([cols_baseline])[0]

    df = genre_per_act_df.loc[genre_per_act_df['year'] <= 2019]
    fig = genre_year_trend(df, tag_dict_map, category_orders)
    image_name = get_image_name(image_counter)

    text = """
    Com estes valores em mãos, podemos fazer uma média geral de todos os artistas presentes por ano,
    e entender como a presença estilo musical tem evoluído ao longo do tempo dentro do 
    lineup do Lollapalooza ({}).
    """.format(image_name)
    center.markdown(text, unsafe_allow_html=True) 

    image_counter = plot(center, fig, image_counter)

    text = """
    Em destaque, podemos observar que as duas principais tendências são a vertiginosa queda 
    do Rock e a ascensão vertiginosa do Pop.
    Alguns outros gêneros em queda são o Electro, o Indie e o Alt. 
    Do outro lado, outros gêneros em alta são o House e o Rap, enquanto que o Hop parece 
    estar estável.

    No gráfico acima, algumas tendências são mais consistentes do que outras - o valor R2, 
    observável clicando nas retas, nos informa quão próximo de fato de uma reta podemos aproximar 
    os pontos observados.
    """
    center.markdown(text, unsafe_allow_html=True) 

    df = genre_per_act_df.loc[genre_per_act_df['year'] <= 2019]
    fig = genre_year_position_trend(df, 4, category_orders, order_hour_dict)
    image_name = get_image_name(image_counter)

    text = """
    Mas será que essa mudança tem diferenças quando olhamos para os diferentes horário dos 
    artistas no lineup? Quando separamos as tendências em quatro grandes grupos, em {},
    percebemos algumas diferenças das tendências gerais.
    """.format(image_name)
    center.markdown(text, unsafe_allow_html=True) 

    image_counter = plot(center, fig, image_counter)

    text = """
    * O Rock, embora tenha diminuído sua participação em todos os horários, caiu mais 
    intensamente entre os headliners;
    * O Electro cresceu entre os artistas maiores, mas caiu nos demais horários
    * Hop e House tiveram as maiores quedas entre entre os artistas do primeiro horário
    * Rap, que diminuiu entre os artistas maiores
    """
    center.markdown(text, unsafe_allow_html=True) 