# mcf-test

This repo is a light prediction model on games data.

### Requirements

1. Python 3.5

2. Dependencies: numpy, pandas, scikit-learn (you can install it with Miniconda3: http://conda.pydata.org/miniconda.html)


### Data

tech-data/games.csv:

| ID	| DATE	DURATION |	HOME_FOOTBALL_TEAM_ID |	AWAY_FOOTBALL_TEAM_ID |
| --	| -------------- |	--------------------- |	--------------------- |
| 000f9323-6f2d-4fc5-8ee1-817978aecf82 |	2016-02-28 14:00:00 |	90 |	fdb4c45b-4caf-11e5-a731-0242ac110002 |	4db49a6d-4eed-4161-94f8-3605560fba79 |

tech-data/goals.csv:

| ID |	MINUTE |	PERIOD |	FOOTBALL_PLAYER_SCORER_ID |	FOOTBALL_GAME_ID |	FOOTBALL_TEAM_ID |	BODY_PART |	AREA |	TYPE |
| -- |	------ |	------ |	------------------------- |	---------------- |	---------------- |	--------- |	---- |	---- |
| 473b84eb-bc01-430a-b536-3229e464bfb8 |	41 |	2 |	b881bfd8-8600-5fe2-bf79-20e272bae555 |	004ba555-5d58-4500-bd91-14fc51e833de |	8353b47a-2c30-5b68-8614-37c178afb5e2 |	PIED_DROIT |	EXTERIEUR_SURFACE |	COUP_FRANC_DIRECT |

### Train & Predict

1. Init: `python init_train.py max_date` (for instance: `python init_train.py '2016-01-31 14:00:00'`), `max_date` splits data in train and test data sets. Games that occured before `max_date` is used in the train set, otherwise it is used in test set.

2. Train: `python train.py`. You can change the algorithm by modify the file (`classifier_to_export`)? Currently configured: `Linear SVM`.

3. Predict: `python predict.py team1 team22` (for instance: `python predict.py 84551c6a-25a6-58e6-aba4-d94855a754c2 62a481f1-4cb0-11e5-a731-0242ac110002`). Return 1, N or 2.
