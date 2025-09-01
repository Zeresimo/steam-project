import requests
import json

print("Pipeline started...")
input_game = input("What game do you want news for? (AppID or name is accepted)")

try: # API call to get the list of all Steam apps with appIDs and names
    gamelist = requests.get("https://api.steampowered.com/ISteamApps/GetAppList/v2/") 
    gamelist.raise_for_status()  # Ensure a successful response
    gamelist_data = gamelist.json() # Parse the JSON response

except requests.HTTPError as http_err: # Handle HTTP errors
    print(f"HTTP error occurred: {http_err}")

except requests.RequestException as e: # Handle other request-related errors
    print(f"Request error occurred: {e}")

matches= [] # Initialize a list to store partial matches
for values in gamelist_data['applist']['apps']: # Loop through the list of apps to find a match

    if str(values['appid']) == input_game or values['name'].lower() == input_game.lower(): # Check if the input matches appID or exact name
        print(f"One match found: {matches[0]['name']} with AppID: {matches[0]['appid']}")
        confirm = input("Is this the correct game? (y/n)")

        if confirm.lower() == 'y': # If user confirms the match
            input_game = str(values['appid']) # Set input_game to the matched appID
            matches.clear() # Clear the matches list
            break

    if input_game.lower() in values['name'].lower(): # Check for partial name match
        matches.append(values)   

if len(matches) > 0: # If there are partial matches, display them
    print("No exact match found, printing partial matches...") 

    for match in matches:
        print(f"{match['name']} (AppID: {match['appid']})")
    print("Exit.") # Exit if no exact match is found
    input_game = input("Please enter the AppID of the correct game from the list above: ")
