#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np

from sklearn.externals import joblib

if len(sys.argv) != 3:
    exit('please enter 2 teams ID')

home_team = sys.argv[1]
away_team = sys.argv[2]

print('Preidction for the game: {0} vs {1}'.format(home_team, away_team))

# load feature for each team
team_stats = pd.read_csv('input/features.csv')

home_stats = team_stats[team_stats['FOOTBALL_TEAM_ID'] == home_team]
away_stats = team_stats[team_stats['FOOTBALL_TEAM_ID'] == away_team]

print(home_stats.shape)
print(away_stats.shape)

for col in list(home_stats.columns.values):
    if col != 'FOOTBALL_TEAM_ID':
        print('{0} | {1} | {2}'.format(col, np.array(home_stats[col])[0], np.array(away_stats[col])[0]))

home_stats = home_stats.drop('FOOTBALL_TEAM_ID', axis=1)
away_stats = away_stats.drop('FOOTBALL_TEAM_ID', axis=1)

# create X to be predicted
X = pd.concat([home_stats, away_stats])
X = np.array(X).ravel().reshape(1, -1)

# load model
clf = joblib.load('model/model.pkl')

# predict
pred = clf.predict(X)
print('Prediction for this game: {0}'.format(pred))
