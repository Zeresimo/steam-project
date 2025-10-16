import requests
import json
import csv
from math import ceil
from urllib.parse import quote
from datetime import datetime, timedelta, date
import os

def get_gamelist():
    gamelist_data = None # Initialize variable to store the game list
    message = None # Variable to store log error messages
    try: # API call to get the list of all Steam apps with appIDs and names
        gamelist = requests.get("https://api.steampowered.com/ISteamApps/GetAppList/v2/") 
        gamelist.raise_for_status()  # Ensure a successful response
        gamelist_data = gamelist.json() # Parse the JSON response into a dictionary 

    except requests.HTTPError as http_err: # Handle HTTP errors
        message = f"HTTP error occurred: {http_err}"
        log_error(message, gamelist) 

    except requests.RequestException as e: # Handle other request-related errors
        message = f"Request error occurred: {e}"
        log_error(message, gamelist) 

    apps_list = gamelist_data['applist']['apps'] if gamelist_data else [] # Extract the list of apps from the response
    appid_dict = {app['appid']: app for app in apps_list} # Create a dictionary mapping app names to their details
    name_dict = {app['name'].lower(): appid_dict[app['appid']] for app in apps_list} # Create a dictionary mapping app names to their details (case-insensitive)
    return apps_list, appid_dict, name_dict

def find_exact_matches(appid_dict, name_dict, input_game):
    input_game = input_game.strip() # Remove leading/trailing whitespace

    if input_game.isdigit(): # Check if the input is numeric (appID)
        appid = int(input_game)
        return [appid_dict[appid]] if appid in appid_dict else None 
        
    else: # Input is treated as a game name
        input_game_lower = input_game.lower() # Convert input to lowercase for case-insensitive comparison
        matches = [app for app in name_dict.values() if app['name'].strip().lower() == input_game_lower]
        return matches if matches else None
               
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

def display_matches(matches, max_display=5, interactive=False):
    if not matches:
        if interactive:
            print("No matches found")
        return None
    total_matches = len(matches)
    total_pages = ceil(total_matches / max_display)
    current_page = 0
    while current_page < total_pages:
        start_index = current_page * max_display
        end_index = min(start_index + max_display, total_matches)  
        
        if interactive:
            print(f"Page {current_page + 1} of {total_pages} | Showing matches {start_index+1} to {end_index} of {total_matches}")
            for i in range(start_index, end_index):
                match = matches[i]
                print(f"{i+1}. {match['name']} - appid: {match['appid']}.")
        else:
            return matches
        
        menu_choice = input("Enter number to select a game, " \
        "Press Enter to see next matches, or 'q' to quit: ")

        if menu_choice.lower() == 'q':
            break
        elif menu_choice.isdigit():
            choice = int(menu_choice)
            if start_index + 1 <= choice <= end_index:
                selected = choice - 1
                return matches[selected]
            else:
                print("Number out of range, please try again.")
        elif menu_choice == '':
            current_page += 1
        
        else:
            print("Invalid input, please enter a number, Enter, or 'q'.")

        if end_index == total_matches:
            break
    return None

def select_game(matches):
    if not matches:
        return None
    
    if len(matches) == 1:
        return matches[0] if confirm_match(matches[0]) else None
    else:
        selected = display_matches(matches, 5, True)
        return selected if selected and confirm_match(selected) else None

def confirm_match(match):
    if not match:
        return None
    user_input = input(f"Did you mean '{match['name']}' (AppID: {match['appid']})? (y/n): ").strip().lower()
    while user_input not in ['y', 'n', 'q']:
        user_input = input("Please enter 'y' or 'n' or 'q': ").strip().lower()
    if user_input == 'y':
        return True
    elif user_input == 'n':
        return False

def get_game_reviews(appid):
    limit = 0 # Initialize limit for pagination
    limit_type = '' # Initialize limit type for pagination

    print("Do you want to limit by number of reviews or by days? (number/days)")
    limit_type = input("Enter 'number' for number of reviews or 'days' for days: ").strip().lower()

    while limit_type not in ['number', 'days']:
        limit_type = input("Please enter 'number' or 'days': ").strip().lower()

    if limit_type == 'number':
        limit = input("Enter the number of reviews to fetch (e.g., 1000): ").strip()

        while not limit.isdigit() or int(limit) <= 0 or int(limit) > 10000:
            limit = input("Please enter a valid positive integer for the number of reviews (less than 10,000): ").strip()

        limit = int(limit)
        stop_condition = lambda review, reviews, limit: len(reviews) >= limit
        print(f"Fetching up to {limit} reviews...")
        return fetch_reviews(appid, limit, stop_condition)
        
    elif limit_type == 'days':
        limit = input("Enter the number of days to fetch reviews from (e.g., 30): ").strip()

        while not limit.isdigit() or int(limit) <= 0 or int(limit) > 365:
            limit = input("Please enter a valid positive integer for the number of days: ").strip()

        limit = int(limit)
        cutoff_date = datetime.now() - timedelta(days=limit)
        stop_condition = lambda review, _, __: datetime.fromtimestamp(review['timestamp_created']) < cutoff_date
        print(f"Fetching reviews from the last {limit} days...")
        return fetch_reviews(appid, limit, stop_condition)
          
def fetch_reviews(appid, limit, stop_condition):
    reviews = [] # List to store all fetched reviews
    encoded_cursor = "*" # Initial cursor for pagination
    
    while True:
        page, encoded_cursor = fetch_review_page(appid, encoded_cursor)

        if not page or not page['reviews']:
            break

        for review in page['reviews']:
            if stop_condition(review, reviews, limit):
                return reviews
            
            else:
                reviews.append(review)

    return reviews

def fetch_review_page(appid, encoded_cursor):
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&filter=recent&language=english&purchase_type=all&review_type=all&num_per_page=100&cursor={encoded_cursor}"
    reviews = None # Initialize variable to store the reviews response
    reviews_data = None # Initialize variable to store the reviews data
    message = None

    try: # API call to get the reviews for the selected game
        reviews = requests.get(url) 
        reviews.raise_for_status()  # Ensure a successful response
        reviews_data = reviews.json() # Parse the JSON response into a dictionary
    
    except requests.HTTPError as http_err: # Handle HTTP errors
        message = f"HTTP error occurred: {http_err}"
        log_error(message, reviews) 

    except requests.RequestException as e: # Handle other request-related errors
        message = f"Request error occurred: {e}"
        log_error(message, reviews) 

    if not reviews_data or 'success' not in reviews_data or reviews_data['success'] != 1:
        message = f"Failed to fetch reviews or no reviews available for {appid}."
        log_error(message, reviews)
        return None, None

    elif reviews_data['success'] == 1 and reviews_data:
        next_page = reviews_data['cursor'] # Get the cursor for the next page of reviews
        encoded_cursor = quote(next_page) # URL-encode the cursor for safe transmission
        return reviews_data, encoded_cursor

def save_reviews(selected_game, reviews, base_path="Pipeline/data/"):
    if not reviews:
        print("No reviews fetched - skipping saving step")
        return
    
    raw_path = os.path.join(base_path, "raw")
    clean_path = os.path.join(base_path, "clean")
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(clean_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename_json = f"reviews_raw_{selected_game['appid']}_{timestamp}.json"
    file_json = os.path.join(raw_path, filename_json)
    
    filename_csv = f"reviews_{selected_game['appid']}_{timestamp}.csv"
    file_csv = os.path.join(clean_path, filename_csv)

    with open(file_json, "w", encoding="utf-8") as f: # Save as json
        json.dump(reviews, f, ensure_ascii=False, indent = 4)
    
    with open(file_csv, "w", newline="", encoding="utf-8") as g:
        writer = csv.writer(g)
        writer.writerow(["review", "voted_up", "timestamp_created", "playtime_forever", "num_reviews"])
        for x in reviews:
            writer.writerow([
                x.get("review", ""),
                x.get("voted_up", ""),
                x.get("timestamp_created", ""),
                x.get("author", {}).get("playtime_forever", ""),
                x.get("author", {}).get("num_reviews", "")
            ])
    print(f"Reviews saved to: \n {file_json} \n {file_csv}")

def log_error(message, response=None): # Error logging function for possible debugging
    base_path = "Pipeline/logs/"
    os.makedirs(base_path, exist_ok=True)
    filename =  "pipeline_error_log.txt"
    log_path = os.path.join(base_path, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"

    print(formatted_message)
    if response is not None:
        print(f"Response code: {response.status_code} \n")
    
    with open(log_path, "a") as logfile:
        logfile.write(f"{formatted_message}")
        if response:
            logfile.write(f" - Status Code: {response.status_code} \n")
        logfile.write("\n" + "-" * 60 + "\n")

    
def main():
    print("Pipeline started...")
    selected_game = None # Variable to store the selected game
    apps_list, appid_dict, name_dict = get_gamelist() # Fetch the game list and create dictionaries for lookups
    
    input_game = input("Enter the game name or AppID: ").strip() # Get user input for the game name or appID
    
    exact_match = find_exact_matches(appid_dict, name_dict, input_game) # Check for an exact match
    selected_game = select_game(exact_match)

    if not selected_game:
        partials = find_partial_matches(name_dict, input_game) # Find partial matches if no exact match
        selected_game = select_game(partials)

    if selected_game:
        print(f"Selected game: {selected_game['name']} (AppID: {selected_game['appid']})")
        reviews = get_game_reviews(selected_game['appid']) # Fetch reviews for the selected game
        print(f"Fetched {len(reviews)} reviews for '{selected_game['name']}'.")
        save_reviews(selected_game, reviews)

    else:
        print("No game selected.")
        exit(0) # Exit if no game is selected
               
    
main()