import pandas as pd
from difflib import SequenceMatcher
import numpy as np 

## Lineup data
def format_hour(hour_str):
    h_array = hour_str.split('h')
    correct_hour = h_array[0] + ':'
    if h_array[1] == '':
        correct_hour += '00'
    else:
        correct_hour += h_array[1]
    return correct_hour


def get_order_in_lineup(time, order_lineup):
    festival_day = time.strftime('%Y-%m-%d')
    limits_day = order_lineup[festival_day]
    porcentage = 100*((time - limits_day['start']).seconds)/((limits_day['end'] - limits_day['start']).seconds)
    return porcentage


## LastFM
def get_all_tags_act(key, acts_dict):
    all_tags = {}
    act_array = acts_dict[key]
    acts_amount = len(act_array)
    for a in act_array:
        if a:     
            tags_dict = a['tags']
            for tag, value in tags_dict.items():
                if tag.lower() in all_tags:
                    all_tags[tag.lower()] += value/(100*acts_amount)
                else:
                    all_tags[tag.lower()] = value/(100*acts_amount)
    return all_tags


def check_genre(artist_tags, genre_dict):
    genre_tags_count = {}
    for genre, all_tags_of_genre in genre_dict.items():
        genre_tags_count[genre] = 0
        for tag, value in artist_tags.items():
            if tag in all_tags_of_genre:
                genre_tags_count[genre] += value
    
    return genre_tags_count


def check_tags(tags_dict, words_to_check):
    probability = 0
    for tag, importance in tags_dict.items(): 
        for w in words_to_check:
                if w == tag:
                    probability += importance 
    return probability


## Spotify Discography

def get_first_release(first_album, first_single, second_release):

    if first_album == -1 and first_single == -1:
        return np.nan
    elif first_single == -1:
        return first_album
    elif first_album == -1:
        return first_single
    else:
        # return min([first_album, first_single])
        delta_album = abs(second_release - first_album)
        delta_single = abs(second_release - first_single)

        if delta_single < delta_album:
            return first_single
        
        else:
            return first_album
        


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
