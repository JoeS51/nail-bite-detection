import requests
import time

BASE_URL = "http://192.168.1.36"

resp = requests.get(f"{BASE_URL}/led/on")
print(resp.status_code)
print(resp.text)
delay = 5 
time.sleep(delay)

resp = requests.get(f"{BASE_URL}/led/off")
print(resp.status_code)
print(resp.text)