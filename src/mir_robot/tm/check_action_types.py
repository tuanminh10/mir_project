import requests
import json
import navigationcacdiem as nav
import sys

headers = nav.api_login()
r = requests.get(f"{nav.API_URL}/actions", headers=headers)
print("actions endpoint:", r.status_code)
types = r.json()
# This returns all actions stored in the system (i.e. already put into missions)
for t in types:
    if "sound" in t.get("action_type", "").lower() or "audio" in t.get("action_type", "").lower():
        # Get details of this specific action
        guid = t.get('guid')
        r_act = requests.get(f"{nav.API_URL}/actions/{guid}", headers=headers)
        print(json.dumps(r_act.json(), indent=2))
        break
