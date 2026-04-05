## created from Grok
## it was doing a terrible job of iteration through conversation.  couldn't trust it anymore.
## attempting to predict 6 golfers for fanduel
## creates course similarity matrix for weightings

import pandas as pd
import numpy as np
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
from scipy.spatial.distance import cosine
from datetime import datetime, timedelta

# Load data (adjust paths to your sources folder)
dfs_data = pd.read_csv('historical_dfs_all.csv')
round_data = pd.read_csv('historical_round_all.csv')
upcoming = pd.read_csv('upcoming.csv')

# Preprocess dates
round_data['date'] = pd.to_datetime(round_data['date'])
current_date = datetime.now()  # Use current time or set to event date
recent_cutoff = current_date - timedelta(days=60)

# Impute NAs in SG columns with mean
sg_cols = ['sg_total', 'sg_t2g', 'sg_putt', 'sg_ott', 'sg_arg', 'sg_app']
for col in sg_cols:
    round_data[col] = round_data[col].fillna(round_data[col].mean())

# Baseline Cosine Similarity Matrix
course_agg = round_data.groupby('course_name')[sg_cols + ['birdies', 'bogies', 'score', 'driving_dist', 'gir']].mean()
course_agg = course_agg.fillna(course_agg.mean(numeric_only=True))
features = course_agg.values
features_norm = (features - features.mean(axis=0)) / features.std(axis=0)
courses = course_agg.index.tolist()
similarity_matrix_cosine = pd.DataFrame(index=courses, columns=courses)
for i in range(len(courses)):
    for j in range(len(courses)):
        similarity_matrix_cosine.iloc[i, j] = 1 - cosine(features_norm[i], features_norm[j]) if i != j else 1.0
similarity_matrix_cosine.to_csv('course_similarity_matrix_cosine.csv')

# Mixed-Effects Similarity Matrix
X = sm.add_constant(round_data[sg_cols + ['birdies', 'score']])
y = round_data['pts_per_round']  # Assuming pts_per_round already apportioned by score
groups = round_data['course_name']
model_mixed = sm.MixedLM(y, X, groups=groups)
result_mixed = model_mixed.fit(method='nm', maxiter=200)
random_effects = result_mixed.random_effects  # Dict of course intercepts
re_df = pd.DataFrame.from_dict(random_effects, orient='index', columns=['re_intercept'])
re_df = re_df.reindex(courses).fillna(0)  # Pad with 0 for missing
course_agg_re = course_agg.join(re_df)
features_re = course_agg_re.values
features_re_norm = (features_re - features_re.mean(axis=0)) / features_re.std(axis=0)
similarity_matrix_mixed = pd.DataFrame(index=courses, columns=courses)
for i in range(len(courses)):
    for j in range(len(courses)):
        similarity_matrix_mixed.iloc[i, j] = 1 - cosine(features_re_norm[i], features_re_norm[j]) if i != j else 1.0
similarity_matrix_mixed.to_csv('course_similarity_matrix_mixed.csv')

# Use mixed matrix for weighting
similarity_matrix = similarity_matrix_mixed

# Apportion dfs_total by round score (revised: inverse to score for better rounds getting more points)
merged = pd.merge(round_data, dfs_data, on=['event_name', 'id_event', 'date', 'player_name', 'dg_id'], how='left')
merged['rounds_played'] = merged.groupby(['dg_id', 'id_event'])['round'].transform('count')
merged['score_inverse'] = 1 / merged['score']  # Lower score = higher proportion
merged['score_prop'] = merged['score_inverse'] / merged.groupby(['dg_id', 'id_event'])['score_inverse'].transform('sum')
merged['pts_per_round'] = merged['total_pts'] * merged['score_prop']

# Weightings
exact_courses = ['Torrey Pines Golf Course (South Course)', 'Torrey Pines Golf Course (North Course)']  # For Farmers
merged['is_recent'] = (merged['date'] >= recent_cutoff).astype(int)
merged['recency_weight'] = np.exp(-0.2 * (current_date - merged['date']).dt.days / 365)
merged['course_weight'] = 0.3
for course in exact_courses:
    merged.loc[merged['course_name'] == course, 'course_weight'] = 3.0
for idx, row in merged.iterrows():
    if row['course_name'] not in exact_courses:
        sim_scores = [similarity_matrix.loc[row['course_name'], ec] for ec in exact_courses if ec in similarity_matrix.columns]
        if sim_scores:
            merged.at[idx, 'course_weight'] = max(0.3, max(sim_scores))
merged['total_weight'] = merged['recency_weight'] * (merged['is_recent'] * 5 + 1) * merged['course_weight']

# Models (e.g., WLS as primary)
X = merged[sg_cols + ['birdies', 'score']]
X = sm.add_constant(X)
y = merged['pts_per_round']
weights = merged['total_weight']
model_wls = sm.WLS(y, X, weights=weights).fit()

# Predict for upcoming (use player means)
player_means = merged.groupby('dg_id')[sg_cols + ['birdies', 'score']].mean().reset_index()
upcoming = upcoming.merge(player_means, on='dg_id', how='left').fillna(0)
X_pred = sm.add_constant(upcoming[sg_cols + ['birdies', 'score']])
upcoming['proj_per_round'] = model_wls.predict(X_pred)
upcoming['proj_total'] = upcoming['proj_per_round'] * 4

# PuLP Optimization
prob = LpProblem("FanDuel_Opt", LpMaximize)
players = upcoming['dg_id'].tolist()
x = {p: LpVariable(f"select_{p}", cat='Binary') for p in players}
prob += lpSum([x[p] * upcoming.loc[upcoming['dg_id'] == p, 'proj_total'].values[0] for p in players])
prob += lpSum([x[p] for p in players]) == 6
prob += lpSum([x[p] * upcoming.loc[upcoming['dg_id'] == p, 'salary'].values[0] for p in players]) <= 60000
prob.solve()
selected = [p for p in players if x[p].value() == 1]
lineup = upcoming[upcoming['dg_id'].isin(selected)]
print(lineup[['player_name', 'salary', 'proj_total']])
print(f"Total Salary: {lineup['salary'].sum()}, Proj Pts: {lineup['proj_total'].sum()}")

# Monte Carlo (100 sims)
lineups = []
for _ in range(100):
    upcoming['proj_sim'] = upcoming['proj_total'] + np.random.normal(0, upcoming['dg_id'].map(player_std))  # Per-golfer std
    prob = LpProblem("FanDuel_Sim", LpMaximize)
    x = {p: LpVariable(f"select_{p}", cat='Binary') for p in players}
    prob += lpSum([x[p] * upcoming.loc[upcoming['dg_id'] == p, 'proj_sim'].values[0] for p in players])
    prob += lpSum([x[p] for p in players]) == 6
    prob += lpSum([x[p] * upcoming.loc[upcoming['dg_id'] == p, 'salary'].values[0] for p in players]) <= 60000
    prob.solve()
    selected = [p for p in players if x[p].value() == 1]
    lineups.append(sorted(upcoming[upcoming['dg_id'].isin(selected)]['player_name'].tolist()))

from collections import Counter
top_lineups = Counter(map(tuple, lineups)).most_common(20)
print(top_lineups)