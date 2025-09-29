import requests
import json

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
    name_dict = {app['name'].lower(): app for app in apps_list} # Create a dictionary mapping app names to their details (case-insensitive)
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

def confirm_match(game):


print("Pipeline started...")


matches= [] # Initialize a list to store partial matches
       
        
    
