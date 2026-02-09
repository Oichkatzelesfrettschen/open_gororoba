import json
import os

settings_path = os.path.expanduser("~/.gemini/settings.json")

try:
    with open(settings_path, 'r') as f:
        data = json.load(f)
    
    if "mcpServers" in data and "postgres" in data["mcpServers"]:
        del data["mcpServers"]["postgres"]
        print("Removed postgres from mcpServers")
    else:
        print("Postgres not found in mcpServers")
        
    with open(settings_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    with open("modification_status.txt", "w") as f:
        f.write("Success")
except Exception as e:
    with open("modification_status.txt", "w") as f:
        f.write(f"Error: {e}")
