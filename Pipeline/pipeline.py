import requests
import json
import csv
from math import ceil
from urllib.parse import quote
from datetime import datetime, timedelta, date
from utils.logger import log
import os

def search_games(query, base_path="Pipeline/logs/", filename="pipeline_log.txt"):

    if not query or not isinstance(query, str):
        log("Invalid query provided to search_games.", level="ERROR", base_path=base_path, filename=filename)
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
            log(f"Failed to fetch game list. Status code: {response.status_code}", 
                level="ERROR", 
                base_path=base_path, 
                filename=filename)
            
            return None
        
        try:
            data = response.json()
            log(f"Game search successful for query: {query}", 
                level="INFO", 
                base_path=base_path, 
                filename=filename)
            
        except Exception as e:
            log(f"Error parsing JSON response: {e}", 
                level="ERROR", 
                base_path=base_path, 
                filename=filename)
            return None
        
        if 'items' not in data:
            log(f"No 'items' key in response data for query: {query}", 
                level="ERROR", 
                base_path=base_path, 
                filename=filename)
            return None
        
        else:
            if isinstance(data['items'], list):
                log(f"'items' key found with {len(data['items'])} results for query: {query}", 
                    level="INFO", 
                    base_path=base_path, 
                    filename=filename)
                
                results = data.get('items', [])

            else:
                log(f"'items' key is not a list for query: {query}",
                    level="ERROR", 
                    base_path=base_path, 
                    filename=filename)
                return None

        if not results:
            log(f"No games found for query: {query}", base_path=base_path, filename=filename)
            return []
        
        log(f"Found {len(results)} games for query: {query}", base_path=base_path, filename=filename)
        return results
    
    except Exception as e:
        log(f"Exception occurred while searching games: {e}", 
            level="ERROR", 
            base_path=base_path, 
            filename=filename)
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
        log("User confirmed the game match.", level="INFO", base_path="Pipeline/logs/", filename="pipeline_log.txt")
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

        if page is None: # Error fetching page
            log("No more pages to fetch or error occurred.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
            return None

        if encoded_cursor is None: # No more pages
            log("No more pages to fetch.", level="INFO", base_path="Pipeline/logs/", filename="pipeline_log.txt")
            return reviews
        
        if encoded_cursor in cursor_list: # Loop detection
            log("Detected repeated cursor - stopping pagination to avoid infinite loop.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
            return None

        if not isinstance(page['reviews'], list): # Invalid reviews format
            log("Invalid reviews format on the current page.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
            return None
        
        # Duplicate page detection
        page_signature = tuple(sorted(review['timestamp_created'] for review in page['reviews']))

        if page_signature in page_signatures:
            log("Duplicate review page detected, skipping this page.", level="INFO",
                base_path="Pipeline/logs/", filename="pipeline_log.txt")
            continue  # Skip to next page

        page_signatures.add(page_signature)

        cursor_list.add(encoded_cursor) # Add current cursor to the set


        if len(page['reviews']) == 0: # No reviews found
            if len(reviews) == 0:
                log("No reviews found for the game.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return None
            else:
                log("End of reviews for the game.", level="INFO", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return reviews

        for review in page['reviews']: # Validate each review structure
            if not isinstance(review, dict): # Invalid review format
                log("Invalid review format in the current page.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return None
            
            item_checker = ['review', 'voted_up', 'timestamp_created', 'author'] # Expected keys in each review

            if not all(key in review for key in item_checker): # Check for expected keys
                log("Missing expected keys in review data.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return None
            
            if not isinstance(review['voted_up'], bool): # Invalid voted_up format
                log("Invalid 'voted_up' format in review data.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return None
            
            if not isinstance(review['timestamp_created'], int): # Invalid timestamp format
                log("Invalid 'timestamp_created' format in review data.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return None

            if not isinstance(review['author'], dict): # Invalid author format
                log("Invalid author format in review data.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return None
            
            author_checker = ['playtime_forever', 'num_reviews'] # Expected keys in author data

            if not all(key in review['author'] for key in author_checker): # Check for expected keys
                log("Missing expected keys in author data.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return None
            
            if not isinstance(review['author']['playtime_forever'], (int,float)) or not isinstance(review['author']['num_reviews'], (int, float)):
                log("Invalid data types in author information.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                return None

        for review in page['reviews']: # Process each review

            reviews.append(review) # Add review to the list

            if stop_condition(review, reviews, limit): # Check stopping condition
                log(f"Stopping condition met after fetching {len(reviews)} reviews.", level="INFO", base_path="Pipeline/logs/", filename="pipeline_log.txt")
                
                return reviews
        
        continue

def fetch_review_page(id, encoded_cursor, base_path="Pipeline/logs/", filename="pipeline_log.txt"):
    url = f"https://store.steampowered.com/appreviews/{id}?json=1&filter=recent&language=english&purchase_type=all&review_type=all&num_per_page=100&cursor={encoded_cursor}"
    reviews = None # Initialize variable to store the reviews response
    reviews_data = None # Initialize variable to store the reviews data
    message = None

    try: # API call to get the reviews for the selected game
        reviews = requests.get(url) 
        reviews.raise_for_status()  # Ensure a successful response
    
    except requests.HTTPError as http_err: # Handle HTTP errors
        message = f"HTTP error occurred: {http_err}"
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None

    except requests.RequestException as e: # Handle other request-related errors
        message = f"Request error occurred: {e}"
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None

    try:
        reviews_data = reviews.json() # Parse the JSON response
        
    except Exception as e:
        message = f"Error parsing JSON response: {e}"
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None

    if 'success' not in reviews_data or reviews_data['success'] != 1: # Check for success key in response
        message = f"API response indicates failure for game id {id}."
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None
    
    if 'reviews' not in reviews_data: # Check for reviews key in response
        message = f"No 'reviews' key in API response for game id {id}."
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None
    
    if not isinstance(reviews_data['reviews'], list):
        message = f"'reviews' key is not a list in API response for game id {id}."
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None
    
    for review in reviews_data['reviews']:
        if not isinstance(review, dict):
            message = f"Invalid review format in API response for game id {id}."
            log(message, level="ERROR", base_path=base_path, filename=filename)
            return None, None
    
    if len(reviews_data['reviews']) == 0:
        message = f"No reviews found in API response for game id {id}."
        log(message, level="INFO", base_path=base_path, filename=filename)
        return None, None
    
    if 'cursor' not in reviews_data: # Check for cursor key in response
        message = f"No 'cursor' key in API response for game id {id}."
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None
    
    if not isinstance(reviews_data['cursor'], str):
        message = f"'cursor' key is not a string in API response for game id {id}."
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None
    
    if reviews_data['cursor'] == "":
        message = f"Empty 'cursor' in API response for game id {id}."
        log(message, level="ERROR", base_path=base_path, filename=filename)
        return None, None
    
    else:
        log(f"Fetched {len(reviews_data['reviews'])} reviews for game id {id}.", level="INFO", base_path=base_path, filename=filename)
        next_page = reviews_data['cursor'] # Get the cursor for the next page of reviews
        temp = encoded_cursor
        encoded_cursor = quote(next_page) # URL-encode the cursor for safe transmission

        if temp == encoded_cursor:
            log(f"Cursor did not change for game id {id}, stopping pagination.", level="INFO", base_path=base_path, filename=filename)
            return None, None
        
        return reviews_data, encoded_cursor

def save_reviews(selected_game, reviews, base_path="Pipeline/data/"):
    if not reviews:
        log("No reviews fetched - skipping saving step", level="INFO", base_path="Pipeline/logs/", filename="pipeline_log.txt")
        print("No reviews fetched - skipping saving step")
        return
    
    raw_path = os.path.join(base_path, "raw")
    clean_path = os.path.join(base_path, "clean")
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(clean_path, exist_ok=True)

    if 'id' not in selected_game: # Check for game ID in selected_game
        log("Selected game ID not found in selected_game dictionary - cannot save review.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
        return None

    if selected_game['id'] is None: # Check for None game ID
        log("Selected game ID is None - cannot save review.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
        return None
    
    if not isinstance(selected_game['id'], (str, int)): # Check for valid game ID type
        log("Selected game ID is not a valid type - cannot save review.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename_json = f"reviews_raw_{selected_game['id']}_{timestamp}.json"
    file_json = os.path.join(raw_path, filename_json)
    
    filename_csv = f"reviews_{selected_game['id']}_{timestamp}.csv"
    file_csv = os.path.join(clean_path, filename_csv)

    for x in reviews:

        checker = ['review', 'voted_up', 'timestamp_created', 'author']

        if not all(key in x for key in checker):
            log("Missing expected keys in review data - cannot save review.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
            return None
        
        author_checker = ['playtime_forever', 'num_reviews']

        if not isinstance(x['author'], dict):
            log("Invalid author format in review data - cannot save review.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
            return None
        
        if not all(key in x['author'] for key in author_checker):
            log("Missing expected keys in author data - cannot save review.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
            return None
        
        if not isinstance(x['author']['playtime_forever'], (int,float)) or not isinstance(x['author']['num_reviews'], (int, float)):
            log("Invalid data types in author information - cannot save review.", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
            return None

    try:
        with open(file_json, "w", encoding="utf-8") as f: # Save as json
            json.dump(reviews, f, ensure_ascii=False, indent = 4)
            log(f"Saved reviews to {file_json}.", level="INFO", base_path="Pipeline/logs/", filename="pipeline_log.txt")

    except Exception as e:
        log(f"Error saving reviews to JSON: {e}", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
        return None

    try:
        with open(file_csv, "w", newline="", encoding="utf-8") as g:
            writer = csv.writer(g)
            writer.writerow(["id", "review", "voted_up", "timestamp_created", "playtime_forever", "num_reviews"])

            for x in reviews:
                writer.writerow([
                    selected_game['id'],
                    x["review"],
                    x["voted_up"],
                    x["timestamp_created"],
                    x["author"]["playtime_forever"],
                    x["author"]["num_reviews"]
                ])

        log(f"Saved {len(reviews)} reviews to {file_json} and {file_csv}.", level="INFO", base_path="Pipeline/logs/", filename="pipeline_log.txt")

    except Exception as e:
        log(f"Error saving reviews to CSV: {e}", level="ERROR", base_path="Pipeline/logs/", filename="pipeline_log.txt")
        return None

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