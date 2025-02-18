def display_innings_scorecard(batting_dict, bowling_dict):
    # Display batting scorecard
    print("+-------------------+--------+---------+--------+-------------------------+")
    print("| Player            | Runs   | Balls   | SR     | Out                     |")
    print("+===================+========+=========+========+=========================+")
    for player, stats in batting_dict.items():
        runs = stats['runs scored']
        balls = stats['balls faced']
        if balls > 0:
            strike_rate = round((runs / balls) * 100, 2)
        else:
            strike_rate = 0.0
        mode_of_dismissal = stats['mode of dismissal']
        if mode_of_dismissal == 'Not out':
            out = 'Not out'
        else:
            out = f"{mode_of_dismissal} {stats['wicket taking bowler']}"
            if 'fielder_involved' in stats and stats['fielder_involved']:
                out += f" b {stats['fielder_involved']}"
        print(f"| {player:17} | {runs:6} | {balls:7} | {strike_rate:6} | {out:25} |")
    print("+-------------------+--------+---------+--------+-------------------------+")

    # Display bowling scorecard
    print("+--------------+--------+---------+-----------+-------+-------+")
    print("| Player       | Runs   | Overs   | Wickets   | Eco   | Extras|")
    print("+==============+========+=========+===========+=======+=======+")
    for player, stats in bowling_dict.items():
        runs = stats['runs_conceded']
        overs = stats['overs bowled']
        wickets = stats['wickets taken']
        extras = stats['extras conceded']
        if overs > 0:
            economy_rate = round(runs / overs, 2)
        else:
            economy_rate = 'NA'
        print(f"| {player:12} | {runs:6} | {overs:7} | {wickets:9} | {economy_rate:5} | {extras:6} |")
    print("+--------------+--------+---------+-----------+-------+-------+")

# Example usage:
innings1_batting_dict = {
    'Player1': {'runs scored': 6, 'balls faced': 5, 'mode of dismissal': 'c', 'wicket taking bowler': 'MM Ali', 'fielder_involved': 'DL Chahar'},
    'Player2': {'runs scored': 1, 'balls faced': 2, 'mode of dismissal': 'c', 'wicket taking bowler': 'DL Chahar', 'fielder_involved': 'SN Thakur'},
    # Add other players' stats here
}

innings1_bowling_dict = {
    'Player1': {'overs bowled': 4, 'wickets taken': 1, 'runs_conceded': 18, 'extras conceded': 2},
    'Player2': {'overs bowled': 2, 'wickets taken': 0, 'runs_conceded': 13, 'extras conceded': 1},
    # Add other players' stats here
}

display_innings_scorecard(innings1_batting_dict, innings1_bowling_dict)
