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
    """
    Search Steam for games matching a user query.

    Args:
        query (str): The game name or phrase to search.

    Returns:
        list | None:
            A list of matching game dictionaries, or None if the request fails.

    Notes:
        - Wraps the Steam storesearch API.
        - Used in both CLI pipeline and dashboard.
    """

    if not query or not isinstance(query, str):
        return utils.error("search_games: Invalid query provided.", LOG)

    url = "https://store.steampowered.com/api/storesearch/"
    params = {"term": query, "cc": "US", "l": "english"}

    # Send request to Steam
    try:
        response = requests.get(url, params=params, timeout=(5, 10))
    except Exception as err:
        return utils.error(f"search_games: Request error ({err}).", LOG)

    # Ensure response is usable
    if response.status_code != 200:
        return utils.error(
            f"search_games: API returned status {response.status_code}.", LOG)

    # Parse JSON
    try:
        data = response.json()
    except Exception as err:
        return utils.error(f"search_games: Failed to parse JSON ({err}).", LOG)

    # Validate expected structure
    if "items" not in data or not isinstance(data["items"], list):
        return utils.error("search_games: Missing or invalid 'items' in API response.", LOG)

    results = data["items"]
    utils.info(f"search_games: Found {len(results)} result(s) for query '{query}'.", LOG)

    return results or []

def display_matches(matches, max_display=5, interactive=False):
    """
    Display paginated game search results and optionally allow selection.

    Args:
        matches (list): List of game dictionaries returned by search_games().
        max_display (int): Number of results to show per page.
        interactive (bool): If False, simply returns matches without printing.

    Returns:
        dict | None | list:
            - If interactive=True: selected game dict or None
            - If interactive=False: returns matches directly
    """

    # No matches found
    if not matches:
        if interactive:
            print("No matches found")
        return None

    # Non-interactive mode: return the whole list as-is
    if not interactive:
        return matches

    total = len(matches)
    total_pages = ceil(total / max_display)
    page = 0

    while page < total_pages:
        start = page * max_display
        end = min(start + max_display, total)

        # Display current page of results
        print(
            f"Page {page + 1} of {total_pages} | "
            f"Showing results {start + 1} to {end} of {total}"
        )

        for i in range(start, end):
            entry = matches[i]
            print(f"{i + 1}. {entry['name']} - id: {entry['id']}")

        # User chooses selection or navigation
        choice = input(
            "Enter number to select a game, press Enter for next page, or 'q' to quit: "
        ).strip().lower()

        # Quit selection entirely
        if choice == "q":
            return None

        # If a game number is chosen from this page
        if choice.isdigit():
            index = int(choice)
            if start + 1 <= index <= end:
                return matches[index - 1]
            print("Number out of range. Try again.")
            continue

        # Move to next page
        if choice == "":
            page += 1
            continue

        print("Invalid input. Enter a number, press Enter, or 'q'.")

    return None

def stop_pagination(page, encoded_cursor, cursor_list, page_signatures, fetched_reviews):
    """
    Decide whether to continue fetching review pages.

    Args:
        page (dict): Parsed JSON page returned by fetch_review_page().
        encoded_cursor (str): URL-encoded next cursor from the API.
        cursor_list (set): Cursors already seen (detect infinite loops).
        page_signatures (set): Signatures of pages already processed.
        fetched_reviews (list): All reviews collected so far.

    Returns:
        tuple:
            (decision, signature)
            decision ∈ {"ERROR", "STOP", "DUPLICATE", "CONTINUE"}
            signature is only returned when decision == "CONTINUE".
    """

    # API returned no valid page
    if page is None:
        utils.log("stop_pagination: Invalid or missing page.", level="ERROR", **LOG)
        return ("ERROR", None)

    # No cursor means API reached the end
    if encoded_cursor is None:
        utils.info("stop_pagination: Cursor is None - reached last page.", LOG)
        return ("STOP", None)

    # Prevent infinite cursor loops
    if encoded_cursor in cursor_list:
        utils.log(f"stop_pagination: Duplicate cursor detected ({encoded_cursor}).",
                  level="ERROR", **LOG)
        return ("ERROR", None)

    # Empty review list - either no data or reached the end
    if len(page["reviews"]) == 0:
        if len(fetched_reviews) == 0:
            utils.log("stop_pagination: First page contained no reviews.",
                      level="ERROR", **LOG)
            return ("ERROR", None)

        utils.info("stop_pagination: Page has no reviews - stopping.", LOG)
        return ("STOP", None)

    # Create a signature to detect duplicate pages
    signature = utils.validate_page_signature(page["reviews"])
    if signature is None:
        utils.log("stop_pagination: Invalid page signature.",
                  level="ERROR", **LOG)
        return ("ERROR", None)

    # Duplicate page should be skipped
    if signature in page_signatures:
        utils.info("stop_pagination: Duplicate page signature - skipping.", LOG)
        return ("DUPLICATE", None)

    # Unique valid page - continue fetching
    utils.info("stop_pagination: Valid page - continuing.", LOG)
    return ("CONTINUE", signature)
  
def select_game(matches):
    """
    Select a single game from a list of game search results.

    Args:
        matches (list): List of game dictionaries returned by search_games().

    Returns:
        dict | None:
            The selected game dictionary, or None if user does not confirm a choice.
    """

    if not matches:
        return None

    if len(matches) == 1:
        game = matches[0]
        return game if confirm_match(game) else None

    # Let user choose from paginated results
    selected = display_matches(matches, max_display=5, interactive=True)
    if not selected:
        return None

    return selected if confirm_match(selected) else None

def confirm_match(match):
    """
    Ask the user to confirm a selected game match.

    Args:
        match (dict): A game dictionary containing at least 'name' and 'id'.

    Returns:
        bool: True if confirmed, False otherwise.
    """

    if not match:
        return False

    prompt = (
        f"Confirm selection: '{match['name']}' "
        f"(id: {match['id']})? (y/n, 'q' to cancel): "
    )

    user_input = input(prompt).strip().lower()

    while user_input not in ("y", "n", "q"):
        user_input = input("Please enter 'y', 'n', or 'q': ").strip().lower()

    return user_input == "y"

def get_game_reviews(appid):
    """
    Prompt the user to choose a review limit method (number or days)
    and construct the appropriate stop condition for pagination.

    Args:
        appid (int): Steam AppID of the selected game.

    Returns:
        list | None: List of review dictionaries, or None on failure.
    """

    # Ask how to limit review fetching
    print("Select review fetch limit: 'number' or 'days'")
    limit_type = input("Enter limit type: ").strip().lower()

    while limit_type not in ("number", "days"):
        limit_type = input("Please enter 'number' or 'days': ").strip().lower()

    # Number of reviews
    if limit_type == "number":
        limit = input("Enter number of reviews to fetch (1-10,000): ").strip()

        while not limit.isdigit() or not (1 <= int(limit) <= 10000):
            limit = input("Please enter a valid integer between 1 and 10,000: ").strip()

        limit = int(limit)
        stop_condition = lambda review, reviews, limit: len(reviews) >= limit

        print(f"Fetching up to {limit} reviews...")
        return fetch_reviews(appid, limit, stop_condition)

    # Days
    limit = input("Enter number of days to fetch from (1-365): ").strip()

    while not limit.isdigit() or not (1 <= int(limit) <= 365):
        limit = input("Please enter a valid integer between 1 and 365: ").strip()

    limit = int(limit)
    cutoff_date = datetime.now() - timedelta(days=limit)

    stop_condition = lambda review, _, __: (
        datetime.fromtimestamp(review["timestamp_created"]) < cutoff_date
    )

    print(f"Fetching reviews from the last {limit} days...")
    return fetch_reviews(appid, limit, stop_condition)

def fetch_reviews(appid, limit, stop_condition):
    """
    Fetch reviews for a game using Steam's paginated review API.

    Args:
        appid (int): Steam AppID.
        limit (int): Max number of reviews or day-range limit,
                     depending on stop_condition logic.
        stop_condition (callable): Determines when to stop pagination.

    Returns:
        list | None: List of collected review dictionaries, or None on error.
    """

    reviews = []
    encoded_cursor = "*"
    cursor_list = set()
    page_signatures = set()

    while True:
        page, encoded_cursor = fetch_review_page(appid, encoded_cursor)

        # Decide how to handle this page
        decision, signature = stop_pagination(
            page, encoded_cursor, cursor_list, page_signatures, reviews
        )

        if decision == "ERROR":
            return utils.error("fetch_reviews: Pagination error encountered.", LOG)

        if decision == "STOP":
            utils.info("fetch_reviews: No more pages available.", LOG)
            return reviews

        if decision == "DUPLICATE":
            utils.info("fetch_reviews: Duplicate page skipped.", LOG)
            continue

        # Valid unique page
        cursor_list.add(encoded_cursor)
        page_signatures.add(signature)

        # Process reviews in this page
        for review in page["reviews"]:
            reviews.append(review)

            # Stop early if limit is reached
            if stop_condition(review, reviews, limit):
                utils.info(
                    f"fetch_reviews: Stopping condition reached "
                    f"({len(reviews)} reviews collected).", LOG
                )
                return reviews

def fetch_review_page(appid, encoded_cursor):
    """
    Fetch a single page of Steam reviews for the given appid using the cursor.

    Args:
        appid (int): Steam AppID.
        encoded_cursor (str): URL-encoded cursor for pagination.

    Returns:
        tuple: (page_data, next_encoded_cursor) or (None, None) on error.
    """

    url = (
        f"https://store.steampowered.com/appreviews/{appid}"
        f"?json=1&filter=recent&language=english&purchase_type=all"
        f"&review_type=all&num_per_page=100&cursor={encoded_cursor}"
    )

    utils.info(f"Requesting review page for appid={appid}, cursor={encoded_cursor}", LOG)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as err:
        return utils.error2(f"fetch_review_page: Request error ({err}).", LOG)

    try:
        page = response.json()
    except Exception as err:
        return utils.error2(f"fetch_review_page: JSON parse failure ({err}).", LOG)
        
    if not utils.validate_review_page(page):
        return utils.error2("fetch_review_page: Invalid page structure.", LOG)

    utils.info(
        f"fetch_review_page: Retrieved {len(page.get('reviews', []))} reviews.", LOG)

    next_cursor = page.get("cursor")
    if next_cursor is None:
        utils.info("fetch_review_page: No next cursor - end of pages.", LOG)
        return page, None

    return page, quote(next_cursor)

def save_reviews(selected_game, reviews):
    """
    Save fetched reviews to raw JSON and cleaned CSV output directories.

    Args:
        selected_game (dict): Game dictionary with 'name' and 'id'.
        reviews (list): List of review dictionaries.

    Returns:
        None
    """

    if not reviews:
        utils.info("save_reviews: No reviews to save.", LOG)
        print("No reviews fetched - skipping save step.")
        return

    if not utils.validate_selected_game(selected_game):
        return utils.error("save_reviews: Invalid selected_game data.", LOG)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    game_name = selected_game["name"]
    appid = selected_game["id"]
    safe_name = utils.sanitize_filename(game_name)

    # Attach metadata to each review
    for review in reviews:
        review["game_name"] = game_name
        review["appid"] = appid

    # File paths
    json_path = os.path.join(PIPE_RAW,
        f"reviews_raw_{appid}_{safe_name}_{timestamp}.json")
    csv_path = os.path.join(PIPE_CLEAN,
        f"reviews_{appid}_{safe_name}_{timestamp}.csv")

    # Save JSON
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reviews, f, ensure_ascii=False, indent=4)
        utils.info(f"Saved raw JSON to {json_path}", LOG)
    except Exception as err:
        return utils.error(f"save_reviews: JSON save error ({err}).", LOG)

    # Save CSV
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as g:
            writer = csv.writer(g)
            writer.writerow([
                "appid", "game_name", "review", "voted_up",
                "timestamp_created", "playtime_forever", "num_reviews"
            ])

            for r in reviews:
                writer.writerow([
                    r["appid"],
                    r["game_name"],
                    r["review"],
                    r["voted_up"],
                    r["timestamp_created"],
                    r["author"]["playtime_forever"],
                    r["author"]["num_reviews"],
                ])

        utils.info(f"Saved cleaned CSV to {csv_path}", LOG)
    except Exception as err:
        return utils.error(f"save_reviews: CSV save error ({err}).", LOG)

    utils.info(f"save_reviews: Completed - {len(reviews)} reviews saved.", LOG)

def main():
    """
    CLI entry point for the review fetching pipeline.

    Workflow:
        1. Prompt user for game name or ID.
        2. Search Steam for matching games.
        3. Allow the user to select and confirm a game.
        4. Choose a fetch limit (number or days).
        5. Fetch reviews with pagination.
        6. Save reviews to raw JSON and cleaned CSV.
    """

    print("Pipeline started...")

    # Ask for game name or AppID
    query = input("Enter the game name or id: ").strip()

    # Search Steam for matching games
    results = search_games(query)

    # User selects a game from the list
    selected_game = select_game(results)
    if not selected_game:
        print("No game selected. Exiting pipeline.")
        return 1

    appid = selected_game["id"]

    # User chooses review limit type
    reviews = get_game_reviews(appid)
    if not reviews:
        print("No reviews fetched. Exiting pipeline.")
        return 1

    # Write reviews to JSON and CSV
    save_reviews(selected_game, reviews)

    return 0

if __name__ == "__main__":
    main()
