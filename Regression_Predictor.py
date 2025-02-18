# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# %%
# Load the dataset
ball_by_ball_2008_to_2022_df = pd.read_csv('Datasets/IPL_Ball_by_Ball_2008_2022.csv')
#second_innings_run_rates_df = pd.read_csv('Datasets/dataset.csv')
#second_innings_2023_df = pd.read_csv('Datasets/csv2023.csv')
matches_df = pd.read_csv('Datasets/IPL_Matches_2008_2022.csv')
#deliveries = pd.read_csv('Datasets/deliveries.csv')

# %%
total_runs = ball_by_ball_2008_to_2022_df.groupby(['ID','innings']).sum()['total_run'].add(1).reset_index()
match_and_tot_runs_df = matches_df.merge(total_runs[['ID','total_run']],left_on='ID',right_on='ID')

# Data pre-processing
match_and_tot_runs_df['Team1'] = match_and_tot_runs_df['Team1'].replace('Delhi Daredevils','Delhi Capitals')
match_and_tot_runs_df['Team2'] = match_and_tot_runs_df['Team2'].replace('Delhi Daredevils','Delhi Capitals')

match_and_tot_runs_df['Team1'] = match_and_tot_runs_df['Team1'].replace('Deccan Chargers','Sunrisers Hyderabad')
match_and_tot_runs_df['Team2'] = match_and_tot_runs_df['Team2'].replace('Deccan Chargers','Sunrisers Hyderabad')

match_and_tot_runs_df['Team1'] = match_and_tot_runs_df['Team1'].replace('Kings XI Punjab','Punjab Kings')
match_and_tot_runs_df['Team2'] = match_and_tot_runs_df['Team2'].replace('Kings XI Punjab','Punjab Kings')


ball_and_match_data_df = match_and_tot_runs_df.merge(ball_by_ball_2008_to_2022_df,on='ID')
ball_and_match_data_df = ball_and_match_data_df.rename(columns={'total_run_y' : 'total_runs_ball', 'total_run_x' : 'innings_total', 'kind' : 'wicket_type'})
#ball_and_match_data_df['Team1Players'] = ball_and_match_data_df['Team1Players'].apply(ast.literal_eval)
#ball_and_match_data_df['Team2Players'] = ball_and_match_data_df['Team2Players'].apply(ast.literal_eval)
#ball_and_match_data_df['Team1Players'] = ball_and_match_data_df['Team1Players'].apply(set)
#ball_and_match_data_df['Team2Players'] = ball_and_match_data_df['Team2Players'].apply(set)

ball_and_match_data_df

# %%
# Select columns of interest
selected_cols = ['ID', 'City', 'Season', 'Team1', 'Team2', 'innings_total', 'innings',
                 'overs', 'ballnumber', 'batter', 'bowler', 'extra_type',
                 'batsman_run', 'extras_run', 'total_runs_ball',
                 'isWicketDelivery', 'player_out', 'wicket_type', 'fielders_involved',
                 'BattingTeam']

ball_and_match_data_df = ball_and_match_data_df[selected_cols]
ball_by_ball_df_subset = ball_and_match_data_df.copy()

# Replace NaN values with zero
ball_by_ball_df_subset.fillna(0, inplace=True)

# Encode categorical variables
cat_vars = ['City', 'Season', 'Team1', 'Team2', 'batter', 'bowler', 'extra_type', 
            'player_out', 'wicket_type', 'fielders_involved', 'BattingTeam']
encoded_df = pd.get_dummies(ball_by_ball_df_subset, columns=cat_vars, drop_first=True)

# Scale numerical variables
scaler = MinMaxScaler()
num_vars = ['innings_total', 'innings', 'overs', 'ballnumber', 'batsman_run', 
            'extras_run', 'total_runs_ball']
encoded_df[num_vars] = scaler.fit_transform(encoded_df[num_vars])


# %%
# Split the dataset into training and testing sets
X = encoded_df.drop(columns=['ID'])
y = encoded_df['total_runs_ball']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Train the multiple linear regression model
model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)


# %%
# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# %%
# Function to predict ball-by-ball records for a cricket game
city = 'Mumbai'
season = '2022'
team1 = 'Mumbai Indians'
team2 = 'Chennai Super Kings'

def predict_ball_by_ball(city, season, team1, team2):
    # Create a dataframe for prediction
    pred_df = pd.DataFrame(columns=X.columns)
    pred_df.loc[0] = 0  # Initialize all values to 0
    
    # Set the values for categorical variables based on input
    pred_df['City_' + city] = 1
    pred_df['Season_' + season] = 1
    pred_df['Team1_' + team1] = 1
    pred_df['Team2_' + team2] = 1
    
    # Add missing columns if not present in training data
    missing_cols = set(X.columns) - set(pred_df.columns)
    for col in missing_cols:
        pred_df[col] = 0
    
    # Predict ball-by-ball records
    ball_predictions = model.predict(pred_df)
    return ball_predictions

# Example usage: Predict ball-by-ball records for a cricket game

predictions = predict_ball_by_ball(city, season, team1, team2)
print("Predicted ball-by-ball records:", predictions)


