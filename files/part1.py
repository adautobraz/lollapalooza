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

    ###### Title headline, subtitle
    center = pad_cols([cols_baseline])[0]
    center.image(image)

    text = """
    # Lollapalooza: Passado e Futuro (Parte 1)
    ## Uma série de três partes sobre os dados de 8 anos do maior festival de música alternativa do país
    Um ensaio visual por <b>Adauto Braz</b>
    <br><br><br>
    À medida que países com vacinação em massa têm voltado à normalidade, presenciamos um dos momentos mais 
    esperados pelos amantes da música ao vivo: o retorno de grandes festivais no mundo todo, 
    com alguns acontecendo ainda este ano. Apesar de uma realidade ainda distante para o Brasil, 
    um dos principais festivais do país - o Lollapalooza - já tem data marcada, e deve acontecer em março de 2022. 
    
    A próxima edição do Lolla, que acontece desde 2012 e tem alcançado um público de mais 
    de 200 mil pessoas, deve se aproveitar da enorme saudade coletiva por shows, e tem potencial para ser 
    histórica. Enquanto aguardamos o desenrolar dos anúncios sobre seu futuro, vamos investigar, 
    com dados, quais tendências têm marcado seus últimos 8 anos, na esperança de entender o que podemos 
    esperar da próxima edição deste que é um dos maiores momentos da música no Brasil.
  
    """
    center.markdown(text, unsafe_allow_html=True)
    # space_out(1)


    # Lineup view and UMAP view
    center = pad_cols([cols_baseline])[0]

    fig = lineups_visualizer(lineups_2019_df, col_labels)
    image_name = get_image_name(image_counter)

    text = """
    ### <b>Como foi o lineup mesmo?</b>
    É bem possível que, mesmo tendo ido para todas as edições, você não lembre no detalhe de 
    todos os artistas que já tocaram ou como foi o lineup de cada ano. Por isso, segue abaixo uma visão 
    interativa de todos os lineups, em {}. Para mais detalhes, basta clicar em cada elemento.
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
    <br>
    Se somarmos o tempo total de música ao vivo que já foi tocada no festival, até 2019, 
    caso alguém quisesse assistir a todas as performances já realizadas, precisaria de cerca 
    de 18 dias, sem parar, como mostrado em {}.

    Para atingir esse total de horas de música, o festival já contou, até 2019, 
    com mais de 380 artistas, de diferentes nacionalidades e estilos musicais. 
    Abaixo, em {},  temos uma visão de todos os artistas que já tocaram no festival, 
    agrupados por similaridade - e os que iriam tocar em 2020 também.

    Para o gráfico de similaridade utilizamos quatro características principais: 
    nacionalidade da atração (brasileira ou internacional), se há leads femininos, tempo de carreira
    e os gêneros musicais associados a cada artista.
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
    com cerca de 6,8% dos artistas que já tocaram no festival com mais de 20 anos de carreira.
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
    Ao longo do tempo ({}), observamos que o tempo médio de carreira das atrações do festival tem 
    variado entre 6 e 10 anos, com destaque para os anos de 2012 e 2015 como as edições mais 
    madura e mais jovem, respectivamente. Em {}, podemos entender, como fica a distribuição do tempo 
    de carreira ano a ano, sendo possível visualizar quais atrações foram pontos fora da curva nas suas edições.
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
    Dado que, como pudemos ver até aqui, existem diferentes perfis de artista no festival,
    teria o tempo de carreira de cada atração alguma relação com o horário em que eles tocam?
    Em {}, vemos que existe sim uma relação entre essas características, com os horários mais
    no final do dia trazendo, na média, artistas com mais tempo de carreira do que os outros horários.

    Apesar disso, como é possível observar em {}, ano a ano, há claras exceções à regra,
    com edições em que os artistas mais maduros tocam no meio do dia, como em 2016 e 2017. 
    Nesses anos, como destaque desses horários, tocaram, em 2016, Bad Religion (34 anos de carreira) e, em 2017,
    Duran Duran (36 anos de carreira).

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
    Uma visão comum entre frequentadores assíduos do festival é a de que o estilo dos 
    artistas que tocam anda diferente. Até aí, nenhuma surpresa. Afinal, se a maioria dos artistas 
    que participam do festival tem até 5 anos de carreira, como mostrado em <b>E</b>, e os últimos 9 anos 
    foram marcados por uma enorme mudança de paradigma em como se consome e distribui música, por conta das 
    plataformas de streaming, nada mais normal do que uma mudança do que se considera, também, 'alternativo'.
 
    Desde 2012, a interação entre tipo de distribuição e tipo de sonoridade deixou de ter uma correlação 
    obrigatória, para se tornar uma visão de dois aspectos diferentes, liberando a expressão criativa e 
    sonora dos artistas. Como consequência, é possível observar uma difusão dos limites de cada gênero musical, 
    com sonoridades extremamente específicas ganhando público, ao mesmo tempo que sons antes de nicho chegam 
    às rádios.
 
    Mas como observar se essas mudanças de contexto no mundo da música tiveram efeito sobre o que se escuta 
    no Lolla?
 
    Temos ao nosso dispor, via LastFM, um conjunto de tags que descrevem o gênero musical de cada artista, 
    como por exemplo alt-rock, indie-pop, hip-hop, alt-pop, etc. Extraindo o conjunto de termos mais 
    comuns nas tags, identificamos que existem cerca de oito que representam a maior parte dos gêneros 
    musicais: Rock, Indie, Alt, Electro, Pop, House, Rap e Hop.

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
    Dada esta visão, podemos começar a entender o que tem acontecido com a presença geral de cada 
    estilo musical ao longo do tempo dentro do lineup do Lollapalooza ({}).
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
    os pontos observados, e quanto mais próximo de 1 este valor, mais consistente é a tendência.
    """
    center.markdown(text, unsafe_allow_html=True) 

    df = genre_per_act_df.loc[genre_per_act_df['year'] <= 2019]
    fig = genre_year_position_trend(df, 4, category_orders, order_hour_dict)
    image_name = get_image_name(image_counter)

    text = """
    Mas será que essa mudança tem diferenças quando olhamos para os diferentes horários dos 
    artistas no lineup? Quando separamos as tendências em quatro grandes grupos, em {},
    percebemos algumas diferenças das tendências gerais.
    """.format(image_name)
    center.markdown(text, unsafe_allow_html=True) 

    image_counter = plot(center, fig, image_counter)

    text = """
    * O Rock, embora tenha diminuído sua participação em todos os horários, caiu mais 
    intensamente entre os headliners;
    * O Electro, apesar de ter uma pequena queda no geral, cresceu consideravelmente entre os artistas maiores; 
    * O Indie trocou de horário, saindo do fim para o começo da tarde;
    * O Pop cresceu sobretudo no primeiro horário;

    Seja por qual ângulo observamos, é impossível negar a evolução musical do festival, 
    tanto um reflexo da expansão do que significa um som 'alternativo' quanto do que é 'pop', e como 
    isso está refletido na demanda por música das gerações atuais.

    ### <b>E o futuro?</b>

    Eventos incomuns tendem a trazer mudanças inesperadas. 
    Por mais que tenhamos identificado tendências consistentes na história do festival, 
    é de se esperar que a próxima edição do Lolla traga surpresas e mude alguns dos panoramas 
    que identificamos.

    Quais serão essas mudanças e o que a pandemia deve representar para a evolução do festival e para o 
    futuro da música, só o tempo dirá. Sejam estas quais forem, já estou com meu ingresso garantido. 
    Nos vemos em 2022!
    
    """
    center.markdown(text, unsafe_allow_html=True)


# # Data path
# data_path = Path('./data/prep')

# # Layout definitions
# cols_baseline=10

# # Print text
# part1(cols_baseline, data_path)