# World Cup Predictor

The full workbook and details can be found in [CatBoostRanker.ipynb](https://github.com/alex-wilson3/aw_ds/blob/main/world_cup_predictor/ranking_methodology/CatBoostRanker.ipynb) file.
Below is a brief overview of the process.

### Load in Elo data
```
elo_team_ratings = pd.read_csv('elo_wc2026.csv')
```
The dataset required minor cleaning and 'win_rate', 'draw_rate' and 'loss_rate' were created.

### Load in International Fixtures data
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
### Defining tournament groups and mapping teams to groups
This also aided the creation of 'group_mean_rating' feature

### Ranking methodology

### Post World Cup Validation
