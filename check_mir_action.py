import requests
import json
IP = "192.168.0.177"
headers = {
    "Authorization": "Basic ZGlzdHJpYnV0b3I6NjRmZmI1OWYwZTRmNjNjNGFkOTFhNWZkODA2YWI2N2E2ZTBmNDVjOTczYjBiZTAyODg3NDkxYTVkYmZmZDI4YQ==",
    "Content-Type": "application/json",
    "Accept-Language": "en-US"
}
r = requests.get(f"http://{IP}/api/v2.0.0/actions", headers=headers)
print("status_code actions", r.status_code)
# print(r.text) # Too long? Let's check keys / first items
try:
    data = r.json()
    print("Action endpoint exists. number of items:", len(data))
    for d in data[:5]:
        print(d)
except Exception as e:
    print("json failed", e)
