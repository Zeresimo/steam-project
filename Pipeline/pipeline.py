import os
import requests
import json
import csv
from math import ceil
from urllib.parse import quote
from datetime import datetime, timedelta, date

from paths import PIPE_RAW, PIPE_CLEAN, PIPE_PROCESSED, PIPE_LOGS
from Pipeline.utils import utils


LOG = {
    "base_path": PIPE_LOGS + "/",
    "filename": "pipeline_log.txt"
}

paths = [PIPE_RAW, PIPE_CLEAN, PIPE_PROCESSED, PIPE_LOGS]


utils.ensure_directory(paths)

def search_games(query):

    if not query or not isinstance(query, str):
        utils.log("Invalid query provided to search_games.", level="ERROR", **LOG)
        return None
    
    url = "https://store.steampowered.com/api/storesearch/"
    params = {
        'term': query,
        'cc': 'US',
        'l': 'english'
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            utils.log(f"Failed to fetch game list. Status code: {response.status_code}", level="ERROR", **LOG)
            
            return None
        
        try:
            data = response.json()
            utils.log(f"Game search successful for query: {query}", level="INFO", **LOG)
            
        except Exception as e:
            utils.log(f"Error parsing JSON response: {e}", level="ERROR", **LOG)
            return None
        
        if 'items' not in data:
            utils.log(f"No 'items' key in response data for query: {query}", level="ERROR", **LOG)
            return None
        
        else:
            if isinstance(data['items'], list):
                utils.log(f"'items' key found with {len(data['items'])} results for query: {query}", level="INFO", **LOG)
                
                results = data.get('items', [])

            else:
                utils.log(f"'items' key is not a list for query: {query}", level="ERROR", **LOG)
                return None

        if not results:
            utils.log(f"No games found for query: {query}", **LOG)
            return []
        
        utils.log(f"Found {len(results)} games for query: {query}", **LOG)
        return results
    
    except Exception as e:
        utils.log(f"Exception occurred while searching games: {e}", level="ERROR", **LOG)
        return None

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
                print(f"{i+1}. {match['name']} - id: {match['id']}.")
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

def stop_pagination(page, encoded_cursor, cursor_list, page_signatures, fetched_reviews):
    if page is None: # Error fetching page
            return ("ERROR", None)

    if encoded_cursor is None: # No more pages
        return ("STOP", None)
    
    if encoded_cursor in cursor_list: # Loop detection
        return ("ERROR", None)
    
    if len(page['reviews']) == 0: # No reviews found
            if len(fetched_reviews) == 0:
                return ("ERROR", None)
            else:
                return ("STOP", None)
            
    # Duplicate page detection
    signature = utils.validate_page_signature(page["reviews"])

    if signature is None:
        return ("ERROR", None)
    if signature in page_signatures:
        return ("DUPLICATE", None)
    
    return ("CONTINUE", signature)
        
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
    user_input = input(f"Did you mean '{match['name']}' (id: {match['id']})? (y/n): ").strip().lower()
    while user_input not in ['y', 'n', 'q']:
        user_input = input("Please enter 'y' or 'n' or 'q': ").strip().lower()
    if user_input == 'y':
        utils.log("User confirmed the game match.", level="INFO", **LOG)
        return True
    elif user_input == 'n':
        return False

def get_game_reviews(id):
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
        return fetch_reviews(id, limit, stop_condition)
        
    elif limit_type == 'days':
        limit = input("Enter the number of days to fetch reviews from (e.g., 30): ").strip()

        while not limit.isdigit() or int(limit) <= 0 or int(limit) > 365:
            limit = input("Please enter a valid positive integer for the number of days: ").strip()

        limit = int(limit)
        cutoff_date = datetime.now() - timedelta(days=limit)
        stop_condition = lambda review, _, __: datetime.fromtimestamp(review['timestamp_created']) < cutoff_date
        print(f"Fetching reviews from the last {limit} days...")
        return fetch_reviews(id, limit, stop_condition)
          
def fetch_reviews(id, limit, stop_condition):
    reviews = [] # List to store all fetched reviews
    encoded_cursor = "*" # Initial cursor for pagination
    cursor_list = set() # List to track cursors for loop detection
    page_signatures = set() # Set to track page signatures for loop detection
    while True:
        page, encoded_cursor = fetch_review_page(id, encoded_cursor)

        decision, signature = stop_pagination(page, encoded_cursor, cursor_list, page_signatures, reviews)
        
        if decision == "ERROR":
            utils.log("Pagination Error", level="ERROR", **LOG)
            return None
        
        elif decision == "STOP":
            utils.log("No more pages or reviews found", level="INFO", **LOG)
            return reviews
        
        elif decision == "DUPLICATE":
            utils.log("Duplicate page found, skipping", level="INFO", **LOG)
            continue

        elif decision == "CONTINUE":
            cursor_list.add(encoded_cursor) # Add current cursor to the set
            page_signatures.add(signature)

        for review in page['reviews']: # Process each review

            reviews.append(review) # Add review to the list

            if stop_condition(review, reviews, limit): # Check stopping condition
                utils.log(f"Stopping condition met after fetching {len(reviews)} reviews.", level="INFO", **LOG)
                
                return reviews

def fetch_review_page(id, encoded_cursor):
    url = f"https://store.steampowered.com/appreviews/{id}?json=1&filter=recent&language=english&purchase_type=all&review_type=all&num_per_page=100&cursor={encoded_cursor}"
    reviews = None # Initialize variable to store the reviews response
    reviews_data = None # Initialize variable to store the reviews data
    message = None

    try: # API call to get the reviews for the selected game
        reviews = requests.get(url) 
        reviews.raise_for_status()  # Ensure a successful response
    
    except requests.HTTPError as http_err: # Handle HTTP errors
        message = f"HTTP error occurred: {http_err}"
        utils.log(message, level="ERROR", **LOG)
        return None, None

    except requests.RequestException as e: # Handle other request-related errors
        message = f"Request error occurred: {e}"
        utils.log(message, level="ERROR", **LOG)
        return None, None

    try:
        reviews_data = reviews.json() # Parse the JSON response
        
    except Exception as e:
        message = f"Error parsing JSON response: {e}"
        utils.log(message, level="ERROR", **LOG)
        return None, None

    if not utils.validate_review_page(reviews_data):
        utils.log("Invalid review page structure received from API.", 
                  level="ERROR", **LOG)
        return None, None
    
    else:
        utils.log(f"Fetched {len(reviews_data['reviews'])} reviews for game id {id}.", level="INFO", **LOG)
        next_page = reviews_data['cursor'] # Get the cursor for the next page of reviews
        encoded_cursor = quote(next_page) # URL-encode the cursor for the next page

        return reviews_data, encoded_cursor

def save_reviews(selected_game, reviews, base_path="Pipeline/data/"):
    if not reviews:
        utils.log("No reviews fetched - skipping saving step", level="INFO", **LOG)
        print("No reviews fetched - skipping saving step")
        return
    
    raw_path = os.path.join(base_path, "raw")
    clean_path = os.path.join(base_path, "clean")

    if not utils.validate_selected_game(selected_game):
        utils.log("Invalid selected game data - cannot save reviews.", level="ERROR", **LOG)
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    game_name = selected_game["name"]
    app_id = selected_game["id"]
    safe_game_name = utils.sanitize_filename(game_name)

    for review in reviews:
        review["game_name"] = game_name
        review["appid"] = app_id

    filename_json = f"reviews_raw_{selected_game['id']}_{safe_game_name}_{timestamp}.json"
    file_json = os.path.join(raw_path, filename_json)
    
    filename_csv = f"reviews_{selected_game['id']}__{safe_game_name}_{timestamp}.csv"
    file_csv = os.path.join(clean_path, filename_csv)

    for x in reviews:

        if not utils.validate_single_review(x):
            utils.log("Invalid review data detected - skipping review.", level="ERROR", **LOG)
            continue

    try:
        with open(file_json, "w", encoding="utf-8") as f: # Save as json
            json.dump(reviews, f, ensure_ascii=False, indent = 4)
            utils.log(f"Saved reviews to {file_json}.", level="INFO", **LOG)

    except Exception as e:
        utils.log(f"Error saving reviews to JSON: {e}", level="ERROR", **LOG)
        return None

    try:
        with open(file_csv, "w", newline="", encoding="utf-8") as g:
            writer = csv.writer(g)
            writer.writerow(["appid", "game_name", "review", "voted_up", "timestamp_created", "playtime_forever", "num_reviews"])

            for x in reviews:
                writer.writerow([
                    x["appid"],
                    x["game_name"],
                    x["review"],
                    x["voted_up"],
                    x["timestamp_created"],
                    x["author"]["playtime_forever"],
                    x["author"]["num_reviews"]
                ])
        utils.log(f"Saved reviews to {file_csv}.", level="INFO", **LOG)

    except Exception as e:
        utils.log(f"Error saving reviews to CSV: {e}", level="ERROR", **LOG)
        return None
    
    utils.log(f"Saved {len(reviews)} reviews to {file_json} and {file_csv}.", level="INFO", **LOG)

def main():
    print("Pipeline started...")
    selected_game = None # Variable to store the selected game
    
    query = input("Enter the game name or id: ").strip() # Get user input for the game name or id
    
    results = search_games(query) # Search for games matching the input

    selected_game = select_game(results) # Let user select the correct game from the results
    
    if not selected_game:
        print("No game selected. Exiting pipeline.")
        exit(0)
    
    appid = selected_game['id'] # Get the app id of the selected game
    reviews = get_game_reviews(appid) # Fetch reviews for the selected game
    save_reviews(selected_game, reviews) # Save the fetched reviews to files

    return 0

if __name__ == "__main__":
    main()