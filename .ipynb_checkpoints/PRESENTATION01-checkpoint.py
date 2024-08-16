import pandas as pd

# Load your dataset
df = pd.read_csv('chess_games.csv')  # Replace with your actual file path

# 1 SHOWING WHAT IS IN THE DATASET
# Display the first five rows of the dataset
print(df.head())

# 2 AVERAGE RATING FOR WHITE AND BLACK PLAYERS
# Calculate the average rating for White and Black players
average_white_rating = df['white_rating'].mean()
average_black_rating = df['black_rating'].mean()

# Create a DataFrame with the average ratings
average_ratings_df = pd.DataFrame({
    'Player Color': ['White', 'Black'],
    'Average Rating': [average_white_rating, average_black_rating]
})

import matplotlib.pyplot as plt

# Plotting the average ratings for White and Black players
plt.figure(figsize=(8, 5))
average_ratings_df.plot(kind='bar', x='Player Color', y='Average Rating', color=['#1f77b4', '#ff7f0e'], legend=False)

# Adding titles and labels
plt.title('Average Ratings for White and Black Players')
plt.xlabel('Player Color')
plt.ylabel('Average Rating')
plt.ylim(0, max(average_white_rating, average_black_rating) + 100)  # Adjust y-axis for better visualization
plt.xticks(rotation=0)  # Keep x labels horizontal
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# 3 SHOWING WHAT THE MOST POPULAR FIRST MOVE IS 
import seaborn as sns

# Load the dataset
df = pd.read_csv('chess_games.csv')

# Extract the first move from the moves column (assuming moves are space-separated)
df['first_move'] = df['moves'].apply(lambda x: x.split()[0])

# Count the total number of games for each first move
total_games_per_move = df['first_move'].value_counts()

# Convert the Series to a DataFrame for easier plotting
total_games_df = total_games_per_move.reset_index()
total_games_df.columns = ['first_move', 'Total Games']

# Set up the matplotlib figure
plt.figure(figsize=(12, 6))

# Plot the total number of games for each first move
sns.barplot(x='first_move', y='Total Games', data=total_games_df, palette='viridis')

# Add labels and title
plt.xlabel('First Move')
plt.ylabel('Total Number of Games')
plt.title('Total Number of Games for Each First Move')
plt.xticks(rotation=45, ha="right")

# Show the plot
plt.tight_layout()
plt.show()



# 4 NUMBER OF WINS AND LOSSES WITH E4 AS WHITE 
# Filter the dataset for games where the first move by White was 'e4'
e4_games = df[df['moves'].str.startswith('e4')]

# Count the number of wins for White
e4_wins_as_white = e4_games[e4_games['winner'] == 'White'].shape[0]

# Count the number of losses for White (which means Black won)
e4_losses_as_white = e4_games[e4_games['winner'] == 'Black'].shape[0]

# Prepare data for plotting
data = {
    'Result': ['Wins', 'Losses'],
    'Count': [e4_wins_as_white, e4_losses_as_white]
}
df_results = pd.DataFrame(data)

# Plotting the number of wins and losses with e4 as White
plt.figure(figsize=(8, 5))
df_results.plot(kind='bar', x='Result', y='Count', color=['skyblue', 'salmon'], legend=False)

# Adding titles and labels
plt.title('Number of Wins and Losses with e4 as White')
plt.xlabel('Result')
plt.ylabel('Number of Games')
plt.ylim(0, max(e4_wins_as_white, e4_losses_as_white) + 100)  # Adjust y-axis for better visualization
plt.xticks(rotation=0)  # Keep x labels horizontal
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()



#WHITE GETS MORE WINS ANYWAYS 
# Count the number of wins for White
white_wins = df[df['winner'] == 'White'].shape[0]

# Count the number of wins for Black
black_wins = df[df['winner'] == 'Black'].shape[0]

# Prepare data for plotting
data = {
    'Player': ['White', 'Black'],
    'Wins': [white_wins, black_wins]
}
df_wins = pd.DataFrame(data)

import matplotlib.pyplot as plt

# Plotting the number of wins for White and Black
plt.figure(figsize=(8, 5))
df_wins.plot(kind='bar', x='Player', y='Wins', color=['#1f77b4', '#ff7f0e'], legend=False)

# Adding titles and labels
plt.title('Number of Wins for White and Black')
plt.xlabel('Player')
plt.ylabel('Number of Wins')
plt.ylim(0, max(white_wins, black_wins) + 100)  # Adjust y-axis for better visualization
plt.xticks(rotation=0)  # Keep x labels horizontal
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()



#WIN RATE OF EACH FIRST PIECE
# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Get the top 10 most common first moves
top_10_first_moves = df['first_move'].value_counts().head(10)

# Create a DataFrame to store win rates
win_rates = []

# Calculate the win rate for each of the top 10 first moves
for move in top_10_first_moves.index:
    # Filter games with the current first move
    move_games = df[df['first_move'] == move]
    
    # Calculate the number of wins
    white_wins = move_games[move_games['winner'] == 'White'].shape[0]
    
    # Calculate total games with this first move
    total_games = move_games.shape[0]
    
    # Calculate the win rate (considering both White and Black wins)
    win_rate = (white_wins) / total_games * 100
    
    # Append the result to the list
    win_rates.append({'First Move': move, 'Win Rate (%)': win_rate})

# Convert the list to a DataFrame
win_rates_df = pd.DataFrame(win_rates)

# Find the first move with the highest win rate
most_successful_move = win_rates_df.loc[win_rates_df['Win Rate (%)'].idxmax()]

# Display the most successful move and its win rate
print("Most Successful First Move:")
print(most_successful_move)

import matplotlib.pyplot as plt

# Plotting the win rates of the top 10 first moves
plt.figure(figsize=(10, 6))
win_rates_df.plot(kind='bar', x='First Move', y='Win Rate (%)', color='skyblue', legend=False)

# Adding titles and labels
plt.title('Win Rates of the Top 10 First Moves for White')
plt.xlabel('First Move')
plt.ylabel('Win Rate (%)')
plt.ylim(0, 100)  # Set the y-axis limit to 100%
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()




# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Count the number of games starting with 'e4'
e4_games = df[df['first_move'] == 'e4'].shape[0]

# Count the number of games starting with 'Nf3'
nf3_games = df[df['first_move'] == 'Nf3'].shape[0]

# Count the number of games starting with 'c4'
c4_games = df[df['first_move'] == 'c4'].shape[0]

# Display the results
print(f"Number of games starting with e4: {e4_games}")
print(f"Number of games starting with Nf3: {nf3_games}")
print(f"Number of games starting with c4: {c4_games}")

import matplotlib.pyplot as plt

# Prepare data for plotting
data = {
    'First Move': ['e4', 'Nf3', 'c4'],
    'Number of Games': [e4_games, nf3_games, c4_games]
}
df_moves = pd.DataFrame(data)

# Plotting the number of games with 'e4', 'Nf3', and 'c4' as the first move
plt.figure(figsize=(8, 5))
df_moves.plot(kind='bar', x='First Move', y='Number of Games', color='skyblue', legend=False)

# Adding titles and labels
plt.title('Number of Games Starting with e4, Nf3, and c4')
plt.xlabel('First Move')
plt.ylabel('Number of Games')
plt.xticks(rotation=0)  # Keep x labels horizontal
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()





# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Filter the dataset for games where the first move was 'c4'
c4_games = df[df['first_move'] == 'c4']

# Filter the c4 games for those that resulted in a win (either for White or Black)
c4_wins = c4_games[c4_games['winner'].isin(['White', 'Black'])]

# Count the occurrences of each opening in the winning c4 games
most_common_winning_opening = c4_wins['opening_fullname'].value_counts().idxmax()

# Display the most common winning opening
print(f"The most common winning opening with c4 is: {most_common_winning_opening}")







# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Filter the dataset for games where the first move was 'c4' and White won
c4_white_wins = df[(df['first_move'] == 'c4') & (df['winner'] == 'white')]

# Extract the first few moves (e.g., first 10 moves total, which is 5 moves by each side)
def extract_opening_moves(moves):
    move_list = moves.split()
    opening_moves = ' '.join(move_list[:10])  # First 10 moves (5 by White and 5 by Black)
    return opening_moves

# Apply the function to extract the opening moves for each game
c4_white_wins['opening_moves'] = c4_white_wins['moves'].apply(extract_opening_moves)

# Display the first few rows to check the extracted opening moves
print(c4_white_wins[['opening_moves']].head())



#PIE GRAPH
# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Filter the dataset for games where the first move was 'c4'
c4_games = df[df['first_move'] == 'c4']

# Define rating ranges
rating_ranges = [(1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 2400)]

# Initialize a list to store results
results = []

# Calculate the percentage of losses for each rating range
for min_rating, max_rating in rating_ranges:
    # Filter games where Black's rating is within the current range
    range_games = c4_games[(df['black_rating'] >= min_rating) & (df['black_rating'] < max_rating)]
    
    # Count the number of losses by Black in this rating range
    losses = range_games[range_games['winner'] == 'White'].shape[0]
    
    # Calculate the total number of games in this range
    total_games = range_games.shape[0]
    
    # Calculate the loss percentage
    loss_percentage = (losses / total_games) * 100 if total_games > 0 else 0
    
    # Store the results
    results.append({'Rating Range': f'{min_rating}-{max_rating}', 'Loss Percentage (%)': loss_percentage})

# Convert the results to a DataFrame
loss_percentage_df = pd.DataFrame(results)

# Display the DataFrame
print(loss_percentage_df)

import matplotlib.pyplot as plt

# Plotting the percentage of losses by rating range as a pie chart
plt.figure(figsize=(10, 6))
plt.pie(loss_percentage_df['Loss Percentage (%)'], labels=loss_percentage_df['Rating Range'], autopct='%1.1f%%', colors=plt.cm.Paired.colors)

# Adding a title
plt.title('Percentage of Losses by Different Rating Ranges to the English Opening (c4)')

# Display the pie chart
plt.show()





df = pd.read_csv('chess_games.csv')  # Replace with your actual file path

# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Filter the dataset for games where the first move was 'e4'
e4_games = df[df['first_move'] == 'e4']

# Extract the second move (first move by Black)
e4_games['black_first_move'] = e4_games['moves'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)

# Count the total games and the number of wins for each response by Black
total_games = e4_games['black_first_move'].value_counts()
black_wins = e4_games[e4_games['winner'] == 'Black']['black_first_move'].value_counts()

# Calculate the success rate (win rate) for each move
win_rate = (black_wins / total_games) * 100

# Convert to a DataFrame for better visualization
win_rate_df = pd.DataFrame({'Move': win_rate.index, 'Win Rate (%)': win_rate.values}).sort_values(by='Win Rate (%)', ascending=False)

# Display the top moves
print(win_rate_df.head(10))

import matplotlib.pyplot as plt

# Plotting the top responses by Black against 1. e4
plt.figure(figsize=(12, 8))
win_rate_df.head(10).plot(kind='barh', x='Move', y='Win Rate (%)', color='green', legend=False)

# Adding titles and labels
plt.title('Top 10 Best Responses by Black Against 1. e4')
plt.xlabel('Win Rate (%)')
plt.ylabel('Black\'s First Move')
plt.xlim(0, 100)  # Set the x-axis limit to 100%
plt.xticks(rotation=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()




# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Filter the dataset for games where the first move was 'e4'
e4_games = df[df['first_move'] == 'e4']

# Extract the second move (first move by Black)
e4_games['black_first_move'] = e4_games['moves'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)

# Count the number of games where Black responded with 'c5' (Sicilian Defense)
sicilian_games = e4_games[e4_games['black_first_move'] == 'c5'].shape[0]

# Count the total number of games where White played 1. e4
total_e4_games = e4_games.shape[0]

# Prepare data for the mirror chart
data = {
    'Category': ['Total e4 games', 'Sicilian Defense'],
    'Number of Games': [total_e4_games, sicilian_games],
    'Mirror': [-sicilian_games, -total_e4_games]  # Use negative values for the mirror effect
}

mirror_df = pd.DataFrame(data)

import matplotlib.pyplot as plt

# Plotting the mirror bar chart
plt.figure(figsize=(10, 6))

# Bar for total e4 games
plt.barh(mirror_df['Category'][0], mirror_df['Number of Games'][0], color='lightblue')

# Bar for Sicilian Defense
plt.barh(mirror_df['Category'][1], mirror_df['Mirror'][1], color='salmon')

# Adding titles and labels
plt.title('Comparison of Total e4 Games and Sicilian Defense Usage')
plt.xlabel('Number of Games')
plt.ylabel('Category')

# Adjust layout
plt.tight_layout()
plt.show()






# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Filter the dataset for games where the first move was 'e4'
e4_games = df[df['first_move'] == 'e4']

# Extract the second move (first move by Black)
e4_games['black_first_move'] = e4_games['moves'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)

# Filter for games where Black's first move was c5 (Sicilian Defense)
sicilian_games = e4_games[e4_games['black_first_move'] == 'c5']

# Define rating ranges for White players
rating_ranges = [(0, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 2400)]

# Initialize a list to store results
results = []

# Calculate the win rate for Black using the Sicilian Defense across different White rating ranges
for min_rating, max_rating in rating_ranges:
    # Filter games where White's rating falls within the current range
    range_games = sicilian_games[(sicilian_games['white_rating'] >= min_rating) & (sicilian_games['white_rating'] < max_rating)]
    
    # Count the number of games Black won
    black_wins = range_games[range_games['winner'] == 'Black'].shape[0]
    
    # Calculate the total number of games in this range
    total_games = range_games.shape[0]
    
    # Calculate the win rate for Black
    win_rate = (black_wins / total_games) * 100 if total_games > 0 else 0
    
    # Store the results
    results.append({'White Rating Range': f'{min_rating}-{max_rating}', 'Win Rate (%)': win_rate})

# Convert the results to a DataFrame
win_rate_df = pd.DataFrame(results)

# Display the win rates
print(win_rate_df)

# Plotting the win rates for Black using the Sicilian Defense across different White rating ranges
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
win_rate_df.plot(kind='barh', x='White Rating Range', y='Win Rate (%)', color='skyblue', legend=False)

# Adding titles and labels
plt.title('Win Rates for Black (1... c5) Against Different White Rating Ranges')
plt.xlabel('Win Rate (%)')
plt.ylabel('White Rating Range')
plt.xlim(0, 100)  # Set the x-axis limit to 100%
plt.xticks(rotation=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()








# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Filter the dataset for games where the first move was 'e4'
e4_games = df[df['first_move'] == 'e4']

# Extract the second move (first move by Black)
e4_games['black_first_move'] = e4_games['moves'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)

# Filter for games where Black's first move was c5 (Sicilian Defense)
sicilian_games = e4_games[e4_games['black_first_move'] == 'c5']

# Count the number of games where White resigns
white_resigns = sicilian_games[(sicilian_games['victory_status'] == 'Resign') & (sicilian_games['winner'] == 'Black')].shape[0]

# Count the number of games that ended in a draw
draw_games = sicilian_games[sicilian_games['victory_status'] == 'Draw'].shape[0]

# Count the number of games where White was checkmated
white_checkmate = sicilian_games[(sicilian_games['victory_status'] == 'Mate') & (sicilian_games['winner'] == 'Black')].shape[0]

# Prepare the data for the bar chart
data = {
    'Outcome': ['White Resigns', 'Draw', 'White Checkmate'],
    'Number of Games': [white_resigns, draw_games, white_checkmate]
}

# Convert to DataFrame
outcome_df = pd.DataFrame(data)

import matplotlib.pyplot as plt

# Plotting the outcomes
plt.figure(figsize=(10, 6))
outcome_df.plot(kind='bar', x='Outcome', y='Number of Games', color='skyblue', legend=False)

# Adding titles and labels
plt.title('Game Outcomes When Black Plays 1... c5 (Sicilian Defense)')
plt.xlabel('Outcome')
plt.ylabel('Number of Games')
plt.xticks(rotation=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()




# Extract the first move from the 'moves' column
df['first_move'] = df['moves'].apply(lambda x: x.split()[0] if pd.notnull(x) else None)

# Filter the dataset for games where White's first move was 'c4'
c4_games = df[df['first_move'] == 'c4']

# Count the number of games where Black resigns
black_resigns = c4_games[(c4_games['victory_status'] == 'Resign') & (c4_games['winner'] == 'White')].shape[0]

# Count the number of games that ended in a draw
draw_games = c4_games[c4_games['victory_status'] == 'Draw'].shape[0]

# Count the number of games where Black was checkmated
black_checkmate = c4_games[(c4_games['victory_status'] == 'Mate') & (c4_games['winner'] == 'White')].shape[0]

# Prepare the data for the bar chart
data = {
    'Outcome': ['Black Resigns', 'Draw', 'Black Checkmate'],
    'Number of Games': [black_resigns, draw_games, black_checkmate]
}

# Convert to DataFrame
outcome_df = pd.DataFrame(data)

import matplotlib.pyplot as plt

# Plotting the outcomes
plt.figure(figsize=(10, 6))
outcome_df.plot(kind='bar', x='Outcome', y='Number of Games', color='salmon', legend=False)

# Adding titles and labels
plt.title('Game Outcomes When White Plays 1. c4 (English Opening)')
plt.xlabel('Outcome')
plt.ylabel('Number of Games')
plt.xticks(rotation=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()














