# %%
#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
import stats
import random

# %%
# Load the datasets
second_innings_run_rates_df = pd.read_csv('Datasets/dataset.csv')
ball_by_ball_2008_to_2022_df = pd.read_csv('Datasets/IPL_Ball_by_Ball_2008_2022.csv')
second_innings_2023_df = pd.read_csv('Datasets/csv2023.csv')
matches_df = pd.read_csv('Datasets/IPL_Matches_2008_2022.csv')
deliveries = pd.read_csv('Datasets/deliveries.csv')

# %%
# Define a function to get the extra type and runs to merge with the main DataFrame
def get_extras(row):
    for extra_type in ['wides', 'noballs', 'byes', 'legbyes']:
        if pd.notna(row[extra_type]):
            return pd.Series([extra_type, row[extra_type]])
    return pd.Series([np.nan, np.nan])

# Apply the function to each row
deliveries[['extra_type', 'extras_run']] = deliveries.apply(get_extras, axis=1)

# Split the 'ball' column into 'overs' and 'ballnumber'
deliveries['overs'], deliveries['ballnumber'] = np.divmod(deliveries['ball'], 1)

# Convert 'ballnumber' to actual ball number by multiplying by 10 and converting to integer
deliveries['ballnumber'] = (deliveries['ballnumber'] * 10).astype(int)

deliveries['isWicketDelivery'] = ~deliveries['wicket_type'].isna().astype(int)

deliveries.drop(columns=['wides', 'noballs', 'byes', 'legbyes', 'season', 'start_date', 'venue', 'other_wicket_type', 'other_player_dismissed', 'ball', 'penalty'], inplace=True)

# Rename the columns
deliveries.rename(columns={'match_id': 'ID', 'player_dismissed': 'player_out', 'runs_off_bat': 'batsman_run', 'striker': 'batter', 'batting_team' : 'BattingTeam'}, inplace=True)
#deliveries = deliveries[['ID', 'innings', 'overs', 'ballnumber', 'batter', 'bowler',
 #      'non-striker', 'extra_type', 'batsman_run', 'extras_run', 'total_run',
  #     'non_boundary', 'isWicketDelivery', 'player_out', 'kind', 'BattingTeam']]

# Stack the two DataFrames
merged_df = pd.concat([ball_by_ball_2008_to_2022_df, deliveries], ignore_index=True)
merged_df

# %%
# Merging the match data and total runs dataframes
total_runs = merged_df.groupby(['ID','innings']).sum()['total_run'].add(1).reset_index()
match_and_tot_runs_df = matches_df.merge(total_runs[['ID','total_run']],left_on='ID',right_on='ID')

# %%
# Data pre-processing
match_and_tot_runs_df['Team1'] = match_and_tot_runs_df['Team1'].replace('Delhi Daredevils','Delhi Capitals')
match_and_tot_runs_df['Team2'] = match_and_tot_runs_df['Team2'].replace('Delhi Daredevils','Delhi Capitals')

match_and_tot_runs_df['Team1'] = match_and_tot_runs_df['Team1'].replace('Deccan Chargers','Sunrisers Hyderabad')
match_and_tot_runs_df['Team2'] = match_and_tot_runs_df['Team2'].replace('Deccan Chargers','Sunrisers Hyderabad')

match_and_tot_runs_df['Team1'] = match_and_tot_runs_df['Team1'].replace('Kings XI Punjab','Punjab Kings')
match_and_tot_runs_df['Team2'] = match_and_tot_runs_df['Team2'].replace('Kings XI Punjab','Punjab Kings')


ball_and_match_data_df = match_and_tot_runs_df.merge(ball_by_ball_2008_to_2022_df,on='ID')
ball_and_match_data_df = ball_and_match_data_df.rename(columns={'total_run_y' : 'total_runs_ball', 'total_run_x' : 'innings_total', 'kind' : 'wicket_type'})
ball_and_match_data_df['Team1Players'] = ball_and_match_data_df['Team1Players'].apply(ast.literal_eval)
ball_and_match_data_df['Team2Players'] = ball_and_match_data_df['Team2Players'].apply(ast.literal_eval)
#ball_and_match_data_df['Team1Players'] = ball_and_match_data_df['Team1Players'].apply(set)
#ball_and_match_data_df['Team2Players'] = ball_and_match_data_df['Team2Players'].apply(set)

# %%
# Define a class to compute batting statistics
class BattingStatistics:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def batting_average(self, player_name, venue=None):
        if venue:
            player_data = self.dataframe[(self.dataframe['batter'] == player_name) & (self.dataframe['City'] == venue)]
        else:
            player_data = self.dataframe[self.dataframe['batter'] == player_name]
        runs_scored = player_data['batsman_run'].sum()
        total_outs = player_data['isWicketDelivery'].sum()
        if total_outs == 0:
            return np.nan  # Avoid division by zero error
        else:
            return runs_scored / total_outs

    def batting_strike_rate(self, player_name, venue=None):
        if venue:
            player_data = self.dataframe[(self.dataframe['batter'] == player_name) & (self.dataframe['City'] == venue)]
        else:
            player_data = self.dataframe[self.dataframe['batter'] == player_name]
        balls_faced = player_data['batter'].count() - player_data[player_data['extra_type'].notna()].shape[0]
        runs_scored = player_data['batsman_run'].sum()
        if balls_faced == 0:
            return np.nan  # Avoid division by zero error
        else:
            return (runs_scored / balls_faced) * 100
    
    def average_balls_faced(self, player_name, venue=None):
        if venue:
            player_data = self.dataframe[(self.dataframe['batter'] == player_name) & (self.dataframe['City'] == venue)]
        else:
            player_data = self.dataframe[self.dataframe['batter'] == player_name]
        
        total_balls_faced = player_data['ID'].count() - player_data[player_data['extra_type'] =='wides'].shape[0] - player_data[player_data['extra_type'] =='noballs'].shape[0]

        if venue:
            matches_with_batter = self.dataframe[
                (self.dataframe['Team1Players'].apply(lambda x: player_name in x if isinstance(x, list) else False)) & (self.dataframe['City'] == venue) |
                (self.dataframe['Team2Players'].apply(lambda x: player_name in x if isinstance(x, list) else False)) & (self.dataframe['City'] == venue)
            ]
        else:
            matches_with_batter = self.dataframe[
                (self.dataframe['Team1Players'].apply(lambda x: player_name in x if isinstance(x, list) else False)) |
                (self.dataframe['Team2Players'].apply(lambda x: player_name in x if isinstance(x, list) else False))
        ]        
        total_matches_played = len(matches_with_batter['ID'].unique())
        
        if total_matches_played == 0:
            return 0
        
        average_balls_faced = total_balls_faced / (2*total_matches_played)
        return average_balls_faced
        

    def player_venue_batting_stats(self, player_name, venue):
        avg = self.batting_average(player_name, venue)
        strike_rate = self.batting_strike_rate(player_name, venue)
        return avg, strike_rate


# Example usage:
# Initialize the class with the dataframe
batting_stats = BattingStatistics(ball_and_match_data_df)

# Compute batting average for a specific player
avg = batting_stats.batting_average('V Kohli') # Virat Kohli is one of the most prolific batsmen in the IPL

# Compute batting strike rate for a specific player
strike_rate = batting_stats.batting_strike_rate('MS Dhoni') # MS Dhoni is known for his high strike rate and finishing skills

# Example usage of the class:
avg_balls = batting_stats.average_balls_faced('V Kohli', 'Bangalore') # Virat Kohli is known for his consistency and ability to play long innings

print(avg_balls)

# %%
# Define a class to compute bowling statistics
class BowlingStatistics:
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def wickets_taken(self, player_name, venue=None):
        if venue:
            player_data = self.dataframe[(self.dataframe['bowler'] == player_name) & (self.dataframe['City'] == venue)]
        else:
            player_data = self.dataframe[self.dataframe['bowler'] == player_name]
        return player_data['isWicketDelivery'].sum()

    def bowling_average(self, player_name, venue=None):
        if venue:
            player_data = self.dataframe[(self.dataframe['bowler'] == player_name) & (self.dataframe['City'] == venue)]
        else:
            player_data = self.dataframe[self.dataframe['bowler'] == player_name]
        runs_conceded = player_data['total_runs_ball'].sum()
        total_wickets = player_data['isWicketDelivery'].sum()
        if total_wickets == 0:
            return np.nan  # Avoid division by zero error
        else:
            return runs_conceded / total_wickets

    def economy_rate(self, player_name, venue=None):
        if venue:
            player_data = self.dataframe[(self.dataframe['bowler'] == player_name) & (self.dataframe['City'] == venue)]
        else:
            player_data = self.dataframe[self.dataframe['bowler'] == player_name]
        balls_bowled = player_data['bowler'].count() - player_data[(player_data['extra_type'] == 'wides') | player_data['extra_type'] == 'noballs'].shape[0]
        overs_bowled = balls_bowled/6
        runs_conceded = player_data['total_runs_ball'].sum() - player_data['extras_run'].sum()
        if balls_bowled == 0:
            return np.nan  # Avoid division by zero error
        else:
            return (runs_conceded / overs_bowled)

    def bowling_strike_rate(self, player_name, venue=None):
        if venue:
            player_data = self.dataframe[(self.dataframe['bowler'] == player_name) & (self.dataframe['City'] == venue)]
        else:
            player_data = self.dataframe[self.dataframe['bowler'] == player_name]
        total_wickets = player_data['isWicketDelivery'].sum()
        balls_bowled = player_data['bowler'].count() - player_data[(player_data['extra_type'] == 'wides') | player_data['extra_type'] == 'noballs'].shape[0]
        if total_wickets == 0:
            return np.nan  # Avoid division by zero error
        else:
            return (balls_bowled / total_wickets)


    def wides_and_no_balls_per_over(self, bowler_name):
        bowler_data = self.dataframe[self.dataframe['bowler'] == bowler_name]
        total_wides_no_balls = bowler_data[bowler_data['extra_type']=='wides'].shape[0] + bowler_data[bowler_data['extra_type']=='noballs'].shape[0]
        total_balls_bowled = bowler_data['bowler'].count()  - bowler_data[(bowler_data['extra_type'] == 'wides') | bowler_data['extra_type'] == 'noballs'].shape[0]
        total_overs_bowled = total_balls_bowled/6

        if total_overs_bowled == 0:
            return np.nan  # Avoid division by zero error
        else:
            return total_wides_no_balls / total_overs_bowled

    def byes_and_leg_byes_per_over(self, bowler_name):
        bowler_data = self.dataframe[self.dataframe['bowler'] == bowler_name]
        total_byes_leg_byes = bowler_data[bowler_data['extra_type']=='byes'].shape[0] + bowler_data[bowler_data['extra_type']=='legbyes'].shape[0]
        total_overs_bowled = (bowler_data['bowler'].shape[0] - bowler_data[bowler_data['extra_type'] == 'wides'].shape[0] - bowler_data[bowler_data['extra_type'] =='noballs'].shape[0])/6

        if total_overs_bowled == 0:
            return np.nan  # Avoid division by zero error
        else:
            return total_byes_leg_byes / total_overs_bowled


    def bowling_probability_score(self, bowler_name):
        bowler_data = self.dataframe[self.dataframe['bowler'] == bowler_name]
        total_matches = bowler_data['ID'].nunique()
        total_overs_bowled = (bowler_data['bowler'].shape[0] - bowler_data[bowler_data['extra_type'] == 'wides'].shape[0] - bowler_data[bowler_data['extra_type'] =='noballs'].shape[0])/6

        if total_matches == 0:
            return 0
        else:
            return total_overs_bowled / total_matches

    def classify_bowlers(self):
        bowlers = self.dataframe['bowler'].unique()
        bowling_scores = [self.bowling_probability_score(bowler) for bowler in bowlers]
        quantiles = np.quantile(bowling_scores, [0.8, 0.6])

        classifications = {}
        for bowler, score in zip(bowlers, bowling_scores):
            if score >= quantiles[0]:
                classifications[bowler] = 'Frontline'
            elif score >= quantiles[1]:
                classifications[bowler] = 'Reliable'
            else:
                classifications[bowler] = 'Part-time'

        return classifications
    
    def get_bowling_classification(self, bowler_name):
        classifications = self.classify_bowlers()
        return classifications[bowler_name]
    

    def probability_bowling_overs(self, bowler_name):
        classifications = self.classify_bowlers()

        if bowler_name not in classifications:
            return {}

        classification = classifications[bowler_name]

        # Assign baseline probabilities based on bowler classification
        if classification == 'Frontline':
            probabilities = {4: 0.75, 3: 0.2, 2: 0.05, 1: 0}
        elif classification == 'Reliable':
            probabilities = {4: 0.5, 3: 0.25, 2: 0.2, 1: 0}
        else:
            # Less probability for part-time bowlers to bowl 3 or 4 overs
            probabilities = {4: 0.05, 3: 0.15, 2: 0.4, 1: 0.4}

        return probabilities


    def player_venue_bowling_stats(self, player_name, venue):
        avg = self.bowling_average(player_name, venue)
        economy = self.economy_rate(player_name, venue)
        strike_rate = self.bowling_strike_rate(player_name, venue)
        return avg, economy, strike_rate

# Example usage:
# Initialize the class with the dataframe

bowling_stats = BowlingStatistics(ball_and_match_data_df)

classifytest = bowling_stats.classify_bowlers()

front_line_test = classifytest['JJ Bumrah'] # Jasprit Bumrah is a frontline bowler for the Mumbai Indians who almost always bowls his full quota of 4 overs

reliable_test = classifytest['R Ashwin'] # Ravichandran Ashwin is a reliable bowler who bowls 3-4 overs in most matches

part_time_test = classifytest['V Kohli'] # Virat Kohli is a very occasional bowler for the Royal Challengers Bangalore

print(front_line_test)
print(reliable_test)
print(part_time_test)

# %%
# Define a class to compute fielding statistics
class FieldingStatistics:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def avg_catches_per_match(self, fielder_name):
        matches_with_fielder = self.dataframe[
            (self.dataframe['fielders_involved'].apply(lambda x: isinstance(x, list) and any(fielder_name.strip().lower() == p.strip().lower() for p in x if isinstance(p, str)))) |
            (self.dataframe['fielders_involved'].str.lower().str.contains(fielder_name.strip().lower(), na=False))
        ]
        total_catches = matches_with_fielder[(matches_with_fielder['wicket_type']== 'caught') | (matches_with_fielder['wicket_type'] == 'caught and bowled') & (matches_with_fielder['fielders_involved'] == fielder_name)].shape[0]
        total_matches = len(matches_with_fielder)
        if total_matches == 0:
            return np.nan  # Avoid division by zero error
        else:
            return total_catches / total_matches

    def avg_stumpings_per_match(self, wicketkeeper_name):
        matches_with_keeper = self.dataframe[
            (self.dataframe['fielders_involved'].apply(lambda x: isinstance(x, list) and any(wicketkeeper_name.strip().lower() == p.strip().lower() for p in x if isinstance(p, str)))) |
            (self.dataframe['fielders_involved'].str.lower().str.contains(wicketkeeper_name.strip().lower(), na=False))
        ]
        total_stumpings = matches_with_keeper['wicket_type'].apply(lambda x: 1 if 'stumped' in str(x).lower() else 0).sum()
        total_matches = len(matches_with_keeper)
        if total_matches == 0:
            return np.nan  # Avoid division by zero error
        else:
            return total_stumpings / total_matches

    def avg_run_outs_per_match(self, fielder_name):
        matches_with_fielder = self.dataframe[
            (self.dataframe['fielders_involved'].apply(lambda x: isinstance(x, list) and any(fielder_name.strip().lower() == p.strip().lower() for p in x if isinstance(p, str)))) |
            (self.dataframe['fielders_involved'].str.lower().str.contains(fielder_name.strip().lower(), na=False))
        ]
        total_run_outs = matches_with_fielder['wicket_type'].apply(lambda x: 1 if 'run out' in str(x).lower() else 0).sum()
        total_matches = len(matches_with_fielder)
        if total_matches == 0:
            return np.nan  # Avoid division by zero error
        else:
            return total_run_outs / total_matches


# Example usage:
# Initialize the class with the dataframe
fielding_stats = FieldingStatistics(ball_and_match_data_df)

# Compute average catches per match for a specific fielder
avg_catches = fielding_stats.avg_catches_per_match('SK Raina') # Suresh Raina is known for his excellent catching ability

# Compute average run outs per match for a specific fielder
avg_run_outs = fielding_stats.avg_run_outs_per_match('RA Jadeja') # Ravindra Jadeja is known for his exceptional fielding skills and direct hit run outs

# Compute average stumpings per match for a specific wicketkeeper
avg_stumpings = fielding_stats.avg_stumpings_per_match('MS Dhoni') # MS Dhoni is known for his lightning quick stumpings

print(avg_catches, avg_run_outs, avg_stumpings)


# %%
# Define a class to calculate venue statistics
class VenueStatistics:
    def __init__(self, df, venue):
        self.df = df
        self.venue = venue


    def average_score_at_venue(self):
        # Filter the dataframe for the specific venue and season range
        venue_df = self.df[(self.df['City'] == self.venue)]

        # Calculate the average score at the venue
        avg_score = venue_df['innings_total'].mean() #Average score per inning

        return avg_score
    
    def innings_score_stdev(self):
        # Filter the dataframe for the specific venue and season range
        venue_df = self.df[(self.df['City'] == self.venue)]

        # Calculate the standard deviation of scores at the venue
        score_stdev = venue_df['innings_total'].std()

        return score_stdev

    def average_number_of_wickets(self):
        # Filter the dataframe for the specific venue and season range
        venue_df = self.df[(self.df['City'] == self.venue)]

        # Calculate the average number of wickets fallen at the venue
        avg_wickets = venue_df['isWicketDelivery'].sum() / (2*venue_df['ID'].nunique()) #Average wickets per inning

        return avg_wickets

venue_stats_mumbai = VenueStatistics(ball_and_match_data_df, 'Mumbai') # Mumbai is known to be a high-scoring venue with good batting conditions and short boundaries
venue_stats_bangalore = VenueStatistics(ball_and_match_data_df, 'Bangalore') # Bangalore is known to be a high-scoring venue with flat pitches and small boundaries
venue_stats_chennai = VenueStatistics(ball_and_match_data_df, 'Chennai') # Chennai is known to be a low-scoring venue with slow and turning pitches and large boundaries
venue_stats_hyderabad = VenueStatistics(ball_and_match_data_df, 'Hyderabad') # At Hyderabad the conditions are generally favourable for bowling

avg_score_mumbai = venue_stats_mumbai.average_score_at_venue()
avg_wickets_mumbai = venue_stats_mumbai.average_number_of_wickets()

avg_score_bangalore = venue_stats_bangalore.average_score_at_venue()
avg_wickets_bangalore = venue_stats_bangalore.average_number_of_wickets()

avg_score_chennai = venue_stats_chennai.average_score_at_venue()
avg_wickets_chennai = venue_stats_chennai.average_number_of_wickets()

avg_score_hyderabad = venue_stats_hyderabad.average_score_at_venue()
avg_wickets_hyderabad = venue_stats_hyderabad.average_number_of_wickets()

print('Chennai stats:', avg_score_chennai, avg_wickets_chennai)
print('Mumbai stats:', avg_score_mumbai, avg_wickets_mumbai)
print('Bangalore stats:', avg_score_bangalore, avg_wickets_bangalore)
print('Hyderabad stats:', avg_score_hyderabad, avg_wickets_hyderabad)

# %%
import random

def assign_overs_innings1(playing_11, bowling_stats):
    # Shuffle the players
    random.shuffle(playing_11)
    num_players = len(playing_11)
    overs_assigned = np.zeros(num_players, dtype=int)
    remaining_overs = 20
    overs_dict = {player: [] for player in playing_11}  # Dictionary to store overs assigned to each bowler

    # Calculate the bowling probability score for each player
    scores = np.array([bowling_stats.bowling_probability_score(player) for player in playing_11])

    # Pre-calculate bowler classifications
    bowler_classifications = bowling_stats.classify_bowlers()

    last_bowler_idx = None

    while remaining_overs > 0:
        # Only consider players who haven't bowled 4 overs yet and didn't bowl the last over
        can_bowl = (overs_assigned < 4) & (np.arange(num_players) != last_bowler_idx)

        # If no players can bowl, break the loop
        if not np.any(can_bowl):
            break

        # Normalize the scores for the players who can bowl
        current_scores = scores.copy()
        current_scores[~can_bowl] = 0
        current_scores /= current_scores.sum()

        # Choose a player to bowl the next over based on their scores
        chosen_player_idx = np.random.choice(np.arange(num_players), p=current_scores)
        chosen_player = playing_11[chosen_player_idx]

        # Assign the over to the chosen player
        overs_assigned[chosen_player_idx] += 1
        remaining_overs -= 1

        # Classify the chosen player as a frontline, support, or part-time bowler
        bowler_classification = bowler_classifications[chosen_player]

        # Store the over number for the chosen player
        overs_dict[chosen_player].append(20 - remaining_overs)

        # Update the last bowler index
        last_bowler_idx = chosen_player_idx

    return overs_dict

playing_11 = ['PK Garg', 'Abhishek Sharma', 'RA Tripathi', 'AK Markram', 'N Pooran', 'Washington Sundar', 'R Shepherd', 'J Suchith', 'B Kumar', 'Umran Malik', 'Fazalhaq Farooqi']
bowling_stats = BowlingStatistics(ball_and_match_data_df)
overs_assigned = assign_overs_innings1(playing_11, bowling_stats)
print(overs_assigned)

# %%
Team_Playing_11_dict = {

    'Chennai Super Kings' : ['RD Gaikwad', 'DP Conway', 'MM Ali', 'RV Uthappa', 'AT Rayudu', 'MS Dhoni', 'S Dube', 'DJ Bravo', 'Simarjeet Singh', 'M Theekshana', 'Mukesh Choudhary'],

    'Delhi Capitals' : ['Mandeep Singh', 'DA Warner', 'MR Marsh', 'RR Pant', 'R Powell', 'Lalit Yadav', 'RV Patel', 'SN Thakur', 'Kuldeep Yadav', 'A Nortje', 'KK Ahmed'],

    'Gujarat Titans' : ['WP Saha','Shubman Gill', 'MS Wade', 'HH Pandya', 'DA Miller', 'R Tewatia', 'Rashid Khan', 'R Sai Kishore', 'LH Ferguson', 'Yash Dayal', 'Mohammed Shami'],

    'Kolkata Knight Riders' : ['VR Iyer', 'AM Rahane', 'N Rana', 'SS Iyer', 'SW Billings', 'RK Singh', 'AD Russell', 'SP Narine', 'UT Yadav', 'TG Southee', 'CV Varun'],

    'Lucknow Super Giants' : ['Q de Kock', 'KL Rahul', 'E Lewis', 'DJ Hooda', 'M Vohra', 'MP Stoinis', 'JO Holder', 'K Gowtham', 'Mohsin Khan', 'Avesh Khan', 'Ravi Bishnoi'],

    'Mumbai Indians' : ['Ishan Kishan', 'RG Sharma', 'SA Yadav', 'Tilak Varma', 'KA Pollard', 'TH David', 'DR Sams', 'M Ashwin', 'K Kartikeya', 'JJ Bumrah', 'RP Meredith'],

    'Punjab Kings' : ['JM Bairstow', 'S Dhawan', 'PBB Rajapaksa', 'LS Livingstone', 'MA Agarwal', 'JM Sharma', 'Harpreet Brar', 'R Dhawan', 'RD Chahar', 'K Rabada', 'Arshdeep Singh'],

    'Rajasthan Royals' : ['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D Padikkal', 'SO Hetmyer', 'R Parag', 'R Ashwin', 'TA Boult', 'YS Chahal', 'M Prasidh Krishna', 'OC McCoy'],

    'Royal Challengers Bangalore' : ['V Kohli', 'F du Plessis', 'RM Patidar', 'GJ Maxwell', 'MK Lomror', 'KD Karthik', 'Shahbaz Ahmed', 'PWH de Silva', 'HV Patel', 'JR Hazlewood', 'Mohammed Siraj'],

    'Sunrisers Hyderabad' : ['Abhishek Sharma', 'PK Garg', 'RA Tripathi', 'N Pooran', 'AK Markram', 'KS Williamson', 'Washington Sundar', 'B Kumar', 'Umran Malik', 'T Natarajan', 'Fazalhaq Farooqi']

}

Venue_list = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
       'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Mumbai', 'Hyderabad',
       'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
       'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack',
       'Dharamsala', 'Kochi', 'Nagpur']


# %%
#Basic scorecard predictor

# Get user input for team1, team2, and venue
team1_name = input("Enter the name of the first team: ")
team2_name = input("Enter the name of the second team: ")
venue_name = input("Enter the name of the venue: ")

# Check if the inputted teams and venue are valid
if team1_name not in Team_Playing_11_dict or team2_name not in Team_Playing_11_dict:
    print("Invalid team name. Please enter a valid team name.")
    exit(1)

if venue_name not in Venue_list:
    print("Invalid venue name. Please enter a valid venue name.")
    exit(1)

# Get the players for team1 and team2
team1 = Team_Playing_11_dict[team1_name]
team2 = Team_Playing_11_dict[team2_name]

# Assume venue_stats is a predefined VenueStatistics object for the inputted venue
# Predefined VenueStatistics object for the inputted venue
venue_stats = VenueStatistics(ball_and_match_data_df, venue_name)
batting_stats = BattingStatistics(ball_and_match_data_df)
bowling_stats = BowlingStatistics(ball_and_match_data_df)

# Calculate average total score and standard deviation at the venue
avg_total_score = venue_stats.average_score_at_venue()
std_dev = venue_stats.innings_score_stdev()

# Calculate average number of wickets at the venue
avg_wickets = venue_stats.average_number_of_wickets()

team1_bowling_lineup = assign_overs_innings1(team1, bowling_stats)
team2_bowling_lineup = assign_overs_innings1(team2, bowling_stats)

#Calculate extras bowled by bowlers in team1:
extras_dict_team1 = {}
for bowler in team1_bowling_lineup.keys():
    extras_dict_team1[bowler] = math.floor(np.nan_to_num(bowling_stats.byes_and_leg_byes_per_over(bowler)) + np.nan_to_num(bowling_stats.wides_and_no_balls_per_over(bowler)))

extras_dict_team2 = {}
for bowler in team2_bowling_lineup.keys():
    extras_dict_team2[bowler] = math.floor(np.nan_to_num(bowling_stats.byes_and_leg_byes_per_over(bowler)) + np.nan_to_num(bowling_stats.wides_and_no_balls_per_over(bowler)))
    

# Function to calculate runs and wickets for a team
def calculate_score(team, avg_total_score, std_dev, avg_wickets):
    total_batting_avg = sum([np.nan_to_num(batting_stats.batting_average(player)) for player in team])
    total_wickets_avg = sum([np.nan_to_num(bowling_stats.wickets_taken(player)) for player in team])
    total_strike_rate = sum([np.nan_to_num(batting_stats.batting_strike_rate(player)) for player in team])

    runs = {}
    wickets = {}
    balls_faced = {}
    max_strike_rate = 0
    max_strike_rate_player = None
    for player in team:
        player_batting_avg = np.nan_to_num(batting_stats.batting_average(player))
        player_wickets_avg = np.nan_to_num(bowling_stats.wickets_taken(player))
        player_strike_rate = np.nan_to_num(batting_stats.batting_strike_rate(player))

        # Calculate runs for the player
        player_runs = (avg_total_score + np.random.normal(0, std_dev)) * (player_batting_avg / total_batting_avg)
        runs[player] = math.floor(player_runs)

        # Calculate wickets for the player
        player_wickets = avg_wickets * (player_wickets_avg / total_wickets_avg)
        wickets[player] = math.floor(player_wickets)

        # Calculate balls faced by the player
        player_balls_faced = 120 * (player_strike_rate / total_strike_rate)
        balls_faced[player] = math.floor(player_balls_faced)

        # Keep track of the player with the highest strike rate
        if player_strike_rate > max_strike_rate:
            max_strike_rate = player_strike_rate
            max_strike_rate_player = player

    # Assign residual balls to the player with the highest strike rate
    total_balls_faced = sum(balls_faced.values())
    residual_balls = 120 - total_balls_faced
    balls_faced[max_strike_rate_player] += residual_balls

    return runs, wickets, balls_faced
# Calculate runs and wickets for team1 and team2
# Convert numpy.str_ to integer before addition

team1_runs, team1_wickets, team1_balls_faced = calculate_score(team1, avg_total_score=avg_total_score, std_dev=std_dev, avg_wickets=avg_wickets)
team2_runs, team2_wickets, team2_balls_faced = calculate_score(team2, avg_total_score=avg_total_score, std_dev=std_dev, avg_wickets=avg_wickets)

# Calculate total runs for team1 and team2

total_runs_team1 = sum(team1_runs.values())


total_runs_team2 = sum(team2_runs.values())

# Concatenate balls and runs dictionary for team1 and team2
team1_runs_balls = {k: [v, team1_balls_faced[k]] for k, v in team1_runs.items()}
team2_runs_balls = {k: [v, team2_balls_faced[k]] for k, v in team2_runs.items()}



# If team2 scored more than team1 + 6, adjust the scores
np.random.seed(44)
if total_runs_team2 > total_runs_team1 + 6:
    diff = total_runs_team2 - total_runs_team1 - 6
    total_runs_team2 -= diff + int(np.random.uniform(1, 6)) # Add a random number between 0 and 6 to the total runs of team2

    # Subtract the difference from the runs of each player in team2
    for player in team2:
        if team2_runs[player] >= diff + int(np.random.uniform(1, 6)):
            team2_runs[player] -= (diff + int(np.random.uniform(1, 6)))
            break

# Print the scorecard
print('1st Innings scorecard:\n\n')
print(f"{team1_name} total runs:", total_runs_team1)
print('Batting Scorecard: [Runs, Balls]\n')
print(team1_runs_balls)
print('Bowling Scorecard:\n')
print(f"{team2_name} wickets:", min(10, sum(team2_wickets.values())))
print('Extras conceded:\n')
print(sum(extras_dict_team2.values()))
print(f"{team2_name} overs bowled:", team2_bowling_lineup)
print('2nd Innings scorecard:\n\n')
print(f"{team2_name} total runs:", total_runs_team2)
print('Batting Scorecard: [Runs, Balls]\n')
print(team2_runs_balls)
print('Bowling Scorecard:\n')
print(f"{team1_name} wickets:", min(10, sum(team1_wickets.values())))
print('Extras conceded:\n')
print(sum(extras_dict_team1.values()))
print(f"{team1_name} overs bowled:", team1_bowling_lineup)

# Determine the winner
print('Result:')
if total_runs_team1 > total_runs_team2:
    print(f"{team1_name} wins!")
elif total_runs_team1 < total_runs_team2:
    print(f"{team2_name} wins!")
else:
    print("It's a tie!")
    


