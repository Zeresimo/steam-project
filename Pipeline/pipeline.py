import requests
import json
from urllib.parse import quote
from datetime import datetime, timedelta

def get_gamelist():
    gamelist_data = None # Initialize variable to store the game list
    try: # API call to get the list of all Steam apps with appIDs and names
        gamelist = requests.get("https://api.steampowered.com/ISteamApps/GetAppList/v2/") 
        gamelist.raise_for_status()  # Ensure a successful response
        gamelist_data = gamelist.json() # Parse the JSON response into a dictionary 

    except requests.HTTPError as http_err: # Handle HTTP errors
        print(f"HTTP error occurred: {http_err}")

    except requests.RequestException as e: # Handle other request-related errors
        print(f"Request error occurred: {e}")

    apps_list = gamelist_data['applist']['apps'] if gamelist_data else [] # Extract the list of apps from the response
    appid_dict = {app['appid']: app for app in apps_list} # Create a dictionary mapping app names to their details
    name_dict = {app['name'].lower(): appid_dict[app['appid']] for app in apps_list} # Create a dictionary mapping app names to their details (case-insensitive)
    return apps_list, appid_dict, name_dict

def find_exact_match(appid_dict, name_dict, input_game):
    input_game = input_game.strip() # Remove leading/trailing whitespace
    if input_game.isdigit(): # Check if the input is numeric (appID)
        appid = int(input_game) 
        if appid in appid_dict: 
            return appid_dict[appid] 
        else:
            return None 
    else: # Input is treated as a game name
        input_game_lower = input_game.lower() # Convert input to lowercase for case-insensitive comparison
        if input_game_lower in name_dict: 
            return name_dict[input_game_lower] 
        else:
            return None 
            
def find_partial_matches(name_dict, input_game):
    input_game = input_game.strip().lower() # Normalize input for case-insensitive comparison
    partial_matches = [] # Initialize a list to store partial matches
    for app in name_dict.values(): 
        if input_game in app['name'].lower(): # Check if input is a substring of the app name
            partial_matches.append(app)
    if not partial_matches:
        return None
    else:
        return partial_matches

def confirm_match(match):
    user_input = input(f"Did you mean '{match['name']}' (AppID: {match['appid']})? (y/n): ").strip().lower()
    while user_input not in ['y', 'n']:
        user_input = input("Please enter 'y' for yes or 'n' for no: ").strip().lower()
    if user_input == 'y':
        return True
    elif user_input == 'n':
        return False

def get_game_reviews(appid):
    limit = 0 # Initialize limit for pagination
    limit_type = '' # Initialize limit type for pagination
    encoded_cursor = "*" # Initial cursor for pagination

    print("Do you want to limit by number of reviews or by days? (number/days)")
    limit_type = input("Enter 'number' for number of reviews or 'days' for days: ").strip().lower()

    while limit_type not in ['number', 'days']:
        limit_type = input("Please enter 'number' or 'days': ").strip().lower()

    if limit_type == 'number':
        limit = input("Enter the number of reviews to fetch (e.g., 1000): ").strip()
        while not limit.isdigit() or int(limit) <= 0 or int(limit) > 10000:
            limit = input("Please enter a valid positive integer for the number of reviews (less than 10,000): ").strip()
        limit = int(limit)
        return fetch_by_number(appid, limit, encoded_cursor)
        

    elif limit_type == 'days':
        limit = input("Enter the number of days to fetch reviews from (e.g., 30): ").strip()
        while not limit.isdigit() or int(limit) <= 0 or int(limit) > 365:
            limit = input("Please enter a valid positive integer for the number of days: ").strip()
        limit = int(limit)
        return fetch_by_days(appid, limit, encoded_cursor)
        
    
def fetch_by_days(appid, limit, encoded_cursor="*"):
    reviews = [] # List to store all fetched reviews
    cutoff_date = datetime.now() - timedelta(days=limit) # Calculate the cutoff date since Steam reviews are timestamped in seconds
    while True:
        reviews_data, encoded_cursor = fetch_review_page(appid, encoded_cursor)
        if reviews_data and reviews_data['reviews']:
            for review in reviews_data['reviews']:
                review_date = datetime.fromtimestamp(review['timestamp_created'])
                if review_date >= cutoff_date:
                    reviews.append(review)
                    print(f"Fetched {len(reviews_data['reviews'])} reviews, total so far: {len(reviews)}")
        else:
            break
    return reviews # Return all reviews if no more pages are available

def fetch_by_number(appid, limit, encoded_cursor="*"):
    reviews = [] # List to store all fetched reviews
    while len(reviews) < limit:
        reviews_data, encoded_cursor = fetch_review_page(appid, encoded_cursor)
        if reviews_data and reviews_data['reviews']:
            reviews.extend(reviews_data['reviews'])
            print(f"Fetched {len(reviews_data['reviews'])} reviews, total so far: {len(reviews)}")
        else:
            break
    return reviews[:limit] # Return only up to the specified limit        

def fetch_review_page(appid, encoded_cursor):
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&filter=recent&language=english&purchase_type=all&review_type=all&num_per_page=100&cursor={encoded_cursor}"
    reviews = None # Initialize variable to store the reviews response
    reviews_data = None # Initialize variable to store the reviews data

    try: # API call to get the reviews for the selected game
        reviews = requests.get(url) 
        reviews.raise_for_status()  # Ensure a successful response
        reviews_data = reviews.json() # Parse the JSON response into a dictionary
    
    except requests.HTTPError as http_err: # Handle HTTP errors
        print(f"HTTP error occurred: {http_err}")

    except requests.RequestException as e: # Handle other request-related errors
        print(f"Request error occurred: {e}")
    
    if reviews_data['success'] == 1:
        next_page = reviews_data['cursor'] # Get the cursor for the next page of reviews
        encoded_cursor = quote(next_page) # URL-encode the cursor for safe transmission
        return reviews_data, encoded_cursor
    else:
        return None, None

def main():
    print("Pipeline started...")
    selected_game = None # Variable to store the selected game
    apps_list, appid_dict, name_dict = get_gamelist() # Fetch the game list and create dictionaries for lookups

    print("Game list fetched and dictionaries created.")

    input_game = input("Enter the game name or AppID: ").strip() # Get user input for the game name or appID
    exact_match = find_exact_match(appid_dict, name_dict, input_game) # Check for an exact match

    if exact_match: # If an exact match is found

        if confirm_match(exact_match): # Confirm the exact match with the user
            selected_game = exact_match
            
    else:
        partials = find_partial_matches(name_dict, input_game) # Find partial matches if no exact match

        if partials:

            for candidate in partials:

                if confirm_match(candidate):
                    selected_game = candidate
                    break
    if selected_game:
        print(f"Selected game: {selected_game['name']} (AppID: {selected_game['appid']})")
        reviews = get_game_reviews(selected_game['appid']) # Fetch reviews for the selected game
        print(f"Fetched {len(reviews)} reviews for '{selected_game['name']}'.")
    else:
        print("No game selected.")
        exit(0) # Exit if no game is selected
               
    
main()