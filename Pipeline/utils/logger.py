import os
from datetime import datetime

def log(message, level="INFO",
        base_path = "Pipeline/logs/",
        filename =  "error_log.txt"): # Error logging function for possible debugging
    
    os.makedirs(base_path, exist_ok=True)
    log_path = os.path.join(base_path, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {level}: {message}"
    
    with open(log_path, "a") as logfile:
        #print(f"writing to log file at: {log_path}") # Debugging line to verify log file path
        logfile.write(formatted_message + "\n")
        
