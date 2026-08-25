# World Cup Predictor


## Executive Summary
The aim of this project is to predict the FIFA World Cup 2026 winner using the ranking algorithm CatBoostRanker. Results correctly predicted Spain, the eventual winner, to come out on top with all current 6 confirmed quarter finalists featuring within the top 10 (likely to be all 8 quarter finalists based on favourites) suggesting the ranking methodology performed well.

The full workbook and details can be found in [CatBoostRanker.ipynb](https://github.com/alex-wilson3/aw_ds/blob/main/world_cup_predictor/ranking_methodology/CatBoostRanker.ipynb) file.
Below is a brief overview of the process.

## Data pre-processing and feature engineering

### Load in ELO data
```
elo_team_ratings = pd.read_csv('elo_wc2026.csv')
```
The dataset required minor cleaning and 'win_rate', 'draw_rate' and 'loss_rate' were created.

### Load in international fixtures
```
internationals = pd.read_csv('international_results.csv')
```
The dataset required cleaning, e.g. creating one column for team by having two rows per fixture and creating the 'result' column.
New features were created such as 'total_win_rate', 'goals_scored_per_game_avg' and 'goals_conceded_per_game_avg' as well as rolling goals scored to create goals scored and conceded across different time bands.

### Merge both datasets into one
e.g.
```
pd.merge(international_games_team_summary_2026, elo_team_ratings_2026, left_on='team', right_on='country', how='left')
```
### Defining tournament groups
The groups for each tournament were defined within dictionaries so each team could be mapped to its group. This also aided the creation of 'group_mean_rating' feature.

## Ranking methodology - CatBoostRanker
Training dataset was World Cup 2022, validation test set was Euros 2024 and then the model was applied to 2026 World Cup.
```
from catboost import CatBoostRanker, Pool

X = data_final_2022.drop(columns = ['goals_scored','goals_conceded','target']).select_dtypes(include=np.number)
y = data_final_2022['target']
data_final_2022_sorted = data_final_2022.sort_values(by='index')
group_id = data_final_2022_sorted['index']

train_pool = Pool(data=X, label=y, group_id=group_id)

model = CatBoostRanker(
    iterations=700,
    depth=4,
    learning_rate=0.54,
    loss_function='YetiRank',
    eval_metric='FilteredDCG',
    random_seed=42
```

## Results and Post World Cup Validation
Spain were correctly predicted as winners by the model with England being just one place away from their true ranking. All eight quarter finalists were included within the top ten with three out of four of the teams who didn’t progress to the semi finals in their correct ranking places. This shows the model’s predicting power that it not only places the predicted winner highly but also the teams that have gone far in the competition.

The Spearman’s rho is 0.72 for this model which suggests strong predictive power.

Spearman's Rho was applied by comparing the model's predictions to each team's actual placing.
```
from scipy.stats import spearmanr
rho, p_value = spearmanr(wc_2026_validation['actual_rank'], wc_2026_validation['predicted_rank'])

print(f"Spearman's Rho: {rho:.4f}")
```
