#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import csv

if len(sys.argv) != 2:
    exit('please enter a split date')

split_limit = sys.argv[1]


print('///////////////////LOAD GAMES////////////////////////////')
## train on all the games played before a given date
games_raw = pd.read_csv('tech-data/games.csv') # read csv

games = games_raw[games_raw['DATE'] <= split_limit] # filter by date

games_count_before_split = games_raw.shape[0]
games_count_after_split = games.shape[0]
print('train on : {0} / {1} ({2:.1f}%)'.format(games_count_after_split, games_count_before_split, games_count_after_split / games_count_before_split * 100))

#print(games.shape)
#print(games.head(4))

print('////////////////////GAMES PLAYED BY TEAM///////////////////////////')
games_home = games.groupby('HOME_FOOTBALL_TEAM_ID').size()
games_home.name = 'count'

games_away = games.groupby('AWAY_FOOTBALL_TEAM_ID').size()
games_away.name = 'count'

# GAMES PLAYED
games_played = pd.concat([games_home, games_away])
games_played = games_played.to_frame()
games_played.index.names = ['FOOTBALL_TEAM_ID']
games_played.columns = ['COUNT']

games_played = games_played.groupby(games_played.index).sum()
games_played = games_played.reset_index()
games_played.sort_values('COUNT', ascending=0, inplace=1)

#print(games_played.shape)
#print(games_played.head(20))

print('////////////////////LOAD GOALS///////////////////////////')
goals_raw = pd.read_csv('tech-data/goals.csv')
goals_raw = goals_raw.replace(np.nan, 'MISSING') # need to remove NaN for further groupby

# only keep goals of the train set
goals = pd.merge(goals_raw, games[['ID']], left_on='FOOTBALL_GAME_ID', right_on='ID', how='inner')
goals.drop('ID_y', axis=1, inplace=True)
goals.rename(columns={'ID_x': 'ID'}, inplace=True)
#print(goals.shape)
#print(goals.head(4))

print('////////////////////COMPUTE GAMES RESULTS///////////////////////////')
games_teams_goals = goals.groupby(['FOOTBALL_GAME_ID', 'FOOTBALL_TEAM_ID']).size()
games_teams_goals.name = 'goals'
games_teams_goals = games_teams_goals.reset_index()

games_score = pd.merge(games, games_teams_goals, left_on=['ID', 'HOME_FOOTBALL_TEAM_ID'], right_on=['FOOTBALL_GAME_ID', 'FOOTBALL_TEAM_ID'], how='left')
games_score = games_score[['ID', 'HOME_FOOTBALL_TEAM_ID', 'AWAY_FOOTBALL_TEAM_ID', 'goals']]
games_score.columns = ['ID', 'HOME_FOOTBALL_TEAM_ID', 'AWAY_FOOTBALL_TEAM_ID', 'HOME_GOALS']
games_score = pd.merge(games_score, games_teams_goals, left_on=['ID', 'AWAY_FOOTBALL_TEAM_ID'], right_on=['FOOTBALL_GAME_ID', 'FOOTBALL_TEAM_ID'], how='left')
games_score = games_score[['ID', 'HOME_FOOTBALL_TEAM_ID', 'AWAY_FOOTBALL_TEAM_ID', 'HOME_GOALS', 'goals']]
games_score.columns = ['ID', 'HOME_FOOTBALL_TEAM_ID', 'AWAY_FOOTBALL_TEAM_ID', 'HOME_GOALS', 'AWAY_GOALS']
games_score = games_score.replace(np.nan, 0.0)

def is_1_N_2(row):
    if row['HOME_GOALS'] > row['AWAY_GOALS']:
        return '1'
    if row['HOME_GOALS'] < row['AWAY_GOALS']:
        return '2'
    else:
        return 'N'

games_score['RESULT'] = games_score.apply (lambda row: is_1_N_2(row), axis=1)

#print(games_score.shape)
#print(games_score.head(4))

print('////////////////////COMPUTE TEAMS RESULTS///////////////////////////')
games_score_h = games_score.groupby(['HOME_FOOTBALL_TEAM_ID', 'RESULT']).size()
games_score_h.name = 'count'
games_score_h = games_score_h.reset_index()
games_score_h.columns = ['FOOTBALL_TEAM_ID', 'RESULT', 'COUNT']
games_score_h_victory = games_score_h[games_score_h['RESULT'] == '1']
games_score_h_draw = games_score_h[games_score_h['RESULT'] == 'N']
games_score_h_defeat = games_score_h[games_score_h['RESULT'] == '2']

games_score_a = games_score.groupby(['AWAY_FOOTBALL_TEAM_ID', 'RESULT']).size()
games_score_a.name = 'count'
games_score_a = games_score_a.reset_index()
games_score_a.columns = ['FOOTBALL_TEAM_ID', 'RESULT', 'COUNT']
games_score_a_victory = games_score_a[games_score_a['RESULT'] == '2']
games_score_a_draw = games_score_a[games_score_a['RESULT'] == 'N']
games_score_a_defeat = games_score_a[games_score_a['RESULT'] == '1']

game_score_victory = pd.concat([games_score_h_victory, games_score_a_victory])
game_score_victory = game_score_victory.groupby('FOOTBALL_TEAM_ID')['COUNT'].sum()
game_score_victory.name = 'VICTORY_COUNT'
game_score_victory = game_score_victory.reset_index()

game_score_draw = pd.concat([games_score_h_draw, games_score_a_draw])
game_score_draw = game_score_draw.groupby('FOOTBALL_TEAM_ID')['COUNT'].sum()
game_score_draw.name = 'DRAW_COUNT'
game_score_draw = game_score_draw.reset_index()

game_score_defeat = pd.concat([games_score_h_defeat, games_score_a_defeat])
game_score_defeat = game_score_defeat.groupby('FOOTBALL_TEAM_ID')['COUNT'].sum()
game_score_defeat.name = 'DEFEAT_COUNT'
game_score_defeat = game_score_defeat.reset_index()

team_perf = pd.merge(game_score_victory, game_score_draw, on='FOOTBALL_TEAM_ID', how='outer')
team_perf = pd.merge(team_perf, game_score_defeat, on='FOOTBALL_TEAM_ID', how='outer')
team_perf = team_perf.replace(np.nan, 0.0)

#print(team_perf.shape)
#print(team_perf.head(4))

def rename_col(df, prefix):
    df_cols = list(df.columns.values)
    df_cols_to_rename = {}
    for col in df_cols:
        if col != 'FOOTBALL_TEAM_ID':
            df_cols_to_rename[col] = prefix + '_' + col
    return df_cols_to_rename

print('////////////////////STATS///////////////////////////')

team_stats = pd.merge(team_perf, games_played, on='FOOTBALL_TEAM_ID', how='inner')
team_stats.rename(columns={'COUNT': 'GAMES_COUNT'}, inplace=True)

def per_game_calc(val, total):
    if total > 0:
        return val / total
    else:
        return 0.0

team_stats['VICTORY_PERC'] = team_stats.apply (lambda row: per_game_calc(row['VICTORY_COUNT'], row['GAMES_COUNT']), axis=1)
team_stats['DRAW_PERC'] = team_stats.apply (lambda row: per_game_calc(row['DRAW_COUNT'], row['GAMES_COUNT']), axis=1)
team_stats['DEFEAT_PERC'] = team_stats.apply (lambda row: per_game_calc(row['DEFEAT_COUNT'], row['GAMES_COUNT']), axis=1)
team_stats.drop('VICTORY_COUNT', axis=1, inplace=True)
team_stats.drop('DRAW_COUNT', axis=1, inplace=True)
team_stats.drop('DEFEAT_COUNT', axis=1, inplace=True)
team_stats.drop('GAMES_COUNT', axis=1, inplace=True)

# compute number of "feature" per game
# add "prefix" to the column name
def goals_features(goals, feature, prefix):
    for_by_feature = goals.groupby(['FOOTBALL_TEAM_ID', feature], as_index=False).size()
    for_by_feature.name = 'size'
    for_by_feature = for_by_feature.reset_index()

    for_by_feature_pivot = pd.pivot_table(for_by_feature, values='size', index='FOOTBALL_TEAM_ID', columns=[feature])
    for_by_feature_pivot.reset_index(level=0, inplace=1)
    for_by_feature_pivot.rename(columns=rename_col(for_by_feature_pivot, prefix + '_' + feature), inplace=True)
    for_by_feature_pivot = for_by_feature_pivot.replace(np.nan, 0.0)
    for_by_feature_pivot = pd.merge(for_by_feature_pivot, games_played, on='FOOTBALL_TEAM_ID', how='left')
    for col in list(for_by_feature_pivot.columns.values):
        if col != 'FOOTBALL_TEAM_ID' and col != 'COUNT':
            for_by_feature_pivot[col + '_PER_M'] = for_by_feature_pivot.apply (lambda row: per_game_calc(row[col], row['COUNT']), axis=1)
            for_by_feature_pivot.drop(col, axis=1, inplace=True)
    for_by_feature_pivot.drop('COUNT', axis=1, inplace=True)
    return for_by_feature_pivot

## FOR BY BODY PART
print('////////////FOR BY BODY PART///////////////////////////////////')
for_by_body_part = goals_features(goals, 'BODY_PART', 'FOR')
# print(for_by_body_part.shape)
# print(for_by_body_part.head(4))

## FOR BY AREA
print('////////////FOR BY AREA///////////////////////////////////')
for_by_area = goals_features(goals, 'AREA', 'FOR')
#print(for_by_area.shape)
#print(for_by_area.head(4))

## FOR BY TYPE
print('////////////FOR BY TYPE///////////////////////////////////')
for_by_type = goals_features(goals, 'TYPE', 'FOR')
#print(for_by_type.shape)
#print(for_by_type.head(4))

print('////////////GOAL AGAINST///////////////////////////////////')
# compute the same DataFrame as goals but replace the team that scores the goal by the team that concedes it
games_and_goals_home = pd.merge(games, goals, left_on=['HOME_FOOTBALL_TEAM_ID', 'ID'], right_on=['FOOTBALL_TEAM_ID', 'FOOTBALL_GAME_ID'], how='inner')
goals_against_home = games_and_goals_home[['ID_y','MINUTE','PERIOD', 'FOOTBALL_PLAYER_SCORER_ID', 'FOOTBALL_GAME_ID', 'AWAY_FOOTBALL_TEAM_ID', 'BODY_PART', 'AREA', 'TYPE']]
goals_against_home.columns = ['ID','MINUTE','PERIOD', 'FOOTBALL_PLAYER_SCORER_ID', 'FOOTBALL_GAME_ID', 'FOOTBALL_TEAM_ID', 'BODY_PART', 'AREA', 'TYPE']

games_and_goals_away = pd.merge(games, goals, left_on=['AWAY_FOOTBALL_TEAM_ID', 'ID'], right_on=['FOOTBALL_TEAM_ID', 'FOOTBALL_GAME_ID'], how='inner')
goals_against_away = games_and_goals_away[['ID_y','MINUTE','PERIOD', 'FOOTBALL_PLAYER_SCORER_ID', 'FOOTBALL_GAME_ID', 'HOME_FOOTBALL_TEAM_ID', 'BODY_PART', 'AREA', 'TYPE']]
goals_against_away.columns = ['ID','MINUTE','PERIOD', 'FOOTBALL_PLAYER_SCORER_ID', 'FOOTBALL_GAME_ID', 'FOOTBALL_TEAM_ID', 'BODY_PART', 'AREA', 'TYPE']

goals_against = pd.concat([goals_against_home, goals_against_away])

#print(goals_against.shape)
#print(goals_against.head(4))

## AGAINST BY BODY PART
print('////////////AGAINST BY BODY PART///////////////////////////////////')
against_by_body_part = goals_features(goals_against, 'BODY_PART', 'AGAINST')
#print(against_by_body_part.shape)
#print(against_by_body_part.head(4))

## AGAINST BY AREA
print('////////////AGAINST BY AREA///////////////////////////////////')
against_by_area = goals_features(goals_against, 'AREA', 'AGAINST')
#print(against_by_area.shape)
#print(against_by_area.head(4))

## AGAINST BY TYPE
print('////////////AGAINST BY TYPE///////////////////////////////////')
against_by_type = goals_features(goals_against, 'TYPE', 'AGAINST')
#print(against_by_type.shape)
#print(against_by_type.head(4))

team_stats = pd.merge(team_stats, for_by_body_part, on='FOOTBALL_TEAM_ID', how='left')
team_stats = pd.merge(team_stats, for_by_area, on='FOOTBALL_TEAM_ID', how='left')
team_stats = pd.merge(team_stats, for_by_type, on='FOOTBALL_TEAM_ID', how='left')
team_stats = pd.merge(team_stats, against_by_body_part, on='FOOTBALL_TEAM_ID', how='left')
team_stats = pd.merge(team_stats, against_by_area, on='FOOTBALL_TEAM_ID', how='left')
team_stats = pd.merge(team_stats, against_by_type, on='FOOTBALL_TEAM_ID', how='left')
team_stats = team_stats.replace(np.nan, 0.0)

#print(team_stats.shape)
#print(team_stats.head(4))

print('////////////////////COMPUTE TRAIN DATA///////////////////////////')
# team_stats contains all the features of a team

x_train = pd.merge(games_score, team_stats, left_on='HOME_FOOTBALL_TEAM_ID', right_on='FOOTBALL_TEAM_ID', how='inner')
x_train.rename(columns=rename_col(team_stats, 'HOME'), inplace=True)

x_train = pd.merge(x_train, team_stats, left_on='AWAY_FOOTBALL_TEAM_ID', right_on='FOOTBALL_TEAM_ID', how='inner')
x_train.rename(columns=rename_col(team_stats, 'AWAY'), inplace=True)

x_train.drop('ID', axis=1, inplace=True)
x_train.drop('HOME_FOOTBALL_TEAM_ID', axis=1, inplace=True)
x_train.drop('AWAY_FOOTBALL_TEAM_ID', axis=1, inplace=True)
x_train.drop('HOME_GOALS', axis=1, inplace=True)
x_train.drop('AWAY_GOALS', axis=1, inplace=True)
x_train.drop('FOOTBALL_TEAM_ID_x', axis=1, inplace=True)
x_train.drop('FOOTBALL_TEAM_ID_y', axis=1, inplace=True)

y_train = x_train['RESULT']
x_train.drop('RESULT', axis=1, inplace=True)

#print(x_train.shape)
#print(x_train.head(4))

#print(y_train.shape)
#print(y_train.head(4))

def to_csv(filename, obj):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        if isinstance(obj, pd.DataFrame):
            writer.writerow(list(obj.columns.values))
        if isinstance(obj, pd.Series):
            writer.writerow([obj.name])
        writer.writerows(np.array(obj))

print('////////////////////EXPORT TRAIN DATA///////////////////////////')
to_csv('input/x_train.csv', x_train)
to_csv('input/y_train.csv', y_train)

print('////////////////////COMPUTE TEST DATA///////////////////////////')
t_games = games_raw[games_raw['DATE'] > split_limit]
t_goals = pd.merge(goals_raw, t_games[['ID']], left_on='FOOTBALL_GAME_ID', right_on='ID', how='inner')
t_goals.drop('ID_y', axis=1, inplace=True)
t_goals.rename(columns={'ID_x': 'ID'}, inplace=True)

t_games_teams_goals = t_goals.groupby(['FOOTBALL_GAME_ID', 'FOOTBALL_TEAM_ID']).size()
t_games_teams_goals.name = 'goals'
t_games_teams_goals = t_games_teams_goals.reset_index()

t_games_score = pd.merge(t_games, t_games_teams_goals, left_on=['ID', 'HOME_FOOTBALL_TEAM_ID'], right_on=['FOOTBALL_GAME_ID', 'FOOTBALL_TEAM_ID'], how='left')
t_games_score = t_games_score[['ID', 'HOME_FOOTBALL_TEAM_ID', 'AWAY_FOOTBALL_TEAM_ID', 'goals']]
t_games_score.columns = ['ID', 'HOME_FOOTBALL_TEAM_ID', 'AWAY_FOOTBALL_TEAM_ID', 'HOME_GOALS']
t_games_score = pd.merge(t_games_score, t_games_teams_goals, left_on=['ID', 'AWAY_FOOTBALL_TEAM_ID'], right_on=['FOOTBALL_GAME_ID', 'FOOTBALL_TEAM_ID'], how='left')
t_games_score = t_games_score[['ID', 'HOME_FOOTBALL_TEAM_ID', 'AWAY_FOOTBALL_TEAM_ID', 'HOME_GOALS', 'goals']]
t_games_score.columns = ['ID', 'HOME_FOOTBALL_TEAM_ID', 'AWAY_FOOTBALL_TEAM_ID', 'HOME_GOALS', 'AWAY_GOALS']
t_games_score = t_games_score.replace(np.nan, 0.0)

t_games_score['RESULT'] = t_games_score.apply (lambda row: is_1_N_2(row), axis=1)

# left merge in order to include team that doesn't play a game, to be test with or without
x_test = pd.merge(t_games_score, team_stats, left_on='HOME_FOOTBALL_TEAM_ID', right_on='FOOTBALL_TEAM_ID', how='left')
x_test.rename(columns=rename_col(team_stats, 'HOME'), inplace=True)

x_test = pd.merge(x_test, team_stats, left_on='AWAY_FOOTBALL_TEAM_ID', right_on='FOOTBALL_TEAM_ID', how='left')
x_test.rename(columns=rename_col(team_stats, 'AWAY'), inplace=True)

x_test.drop('ID', axis=1, inplace=True)
x_test.drop('HOME_FOOTBALL_TEAM_ID', axis=1, inplace=True)
x_test.drop('AWAY_FOOTBALL_TEAM_ID', axis=1, inplace=True)
x_test.drop('HOME_GOALS', axis=1, inplace=True)
x_test.drop('AWAY_GOALS', axis=1, inplace=True)
x_test.drop('FOOTBALL_TEAM_ID_x', axis=1, inplace=True)
x_test.drop('FOOTBALL_TEAM_ID_y', axis=1, inplace=True)

y_test = x_test['RESULT']
x_test.drop('RESULT', axis=1, inplace=True)

x_test = x_test.replace(np.nan, 0.0) # manage team without games

#print(x_test.shape)
#print(x_test.head(4))

#print(y_test.shape)
#print(y_test.head(4))

print('////////////////////EXPORT TEST DATA///////////////////////////')
to_csv('input/x_test.csv', x_test)
to_csv('input/y_test.csv', y_test)

print('////////////////////EXPORT FEATURES///////////////////////////')
to_csv('input/features.csv', team_stats)
