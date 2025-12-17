import requests
import sys
import json

# matches streamlit logic
import os


BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
# Remove trailing slash if present
BASE_URL = BASE_URL.rstrip("/")
USERNAME = "Arnav"
PASSWORD = "1234"

def print_result(name, response):
    if response.status_code in [200, 201]:
        print(f"✅ {name}: Success ({response.status_code})")
        try:
            data = response.json()
            if isinstance(data, list):
                print(f"   Items: {len(data)}")
                if data:
                    print(f"   Example: {data[0]}")
            elif isinstance(data, dict):
                keys = list(data.keys())
                print(f"   Keys: {keys}")
                if "count" in data:
                    print(f"   Count: {data['count']}")
        except:
            print("   (No JSON content)")
    else:
        print(f"❌ {name}: Failed ({response.status_code})")
        print(f"   Response: {response.text[:200]}...")

def main():
    print(f"Testing API at: {BASE_URL}")

    # 1. Health Check
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=10)
        print_result("Health Check", resp)
    except Exception as e:
        print(f"❌ Health check failed to connect: {e}")
        return

    # 2. Login
    print("\nAttempting Login...")
    try:
        resp = requests.post(f"{BASE_URL}/auth/login", json={"username": USERNAME, "password": PASSWORD})
        
        if resp.status_code == 401:
            print("⚠️ Login failed (401). attempting to register new user...")
            reg_resp = requests.post(f"{BASE_URL}/auth/register", json={"username": USERNAME, "password": PASSWORD})
            print_result("Registration", reg_resp)
            
            # Retry login
            if reg_resp.status_code in [200, 201]:
                print("   Retrying Login...")
                resp = requests.post(f"{BASE_URL}/auth/login", json={"username": USERNAME, "password": PASSWORD})
                
        print_result("Login", resp)

    except Exception as e:
         print(f"❌ Login failed to connect: {e}")
         return
    
    if resp.status_code != 200:
        print("❌ Cannot proceed without login.")
        return

    token_data = resp.json()
    token = token_data.get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    print(f"   Token received: {token[:10]}...")

    # 3. Get Instances
    print("\nFetching Instances (Workspaces)...")
    resp = requests.get(f"{BASE_URL}/instances", headers=headers)
    print_result("Get Instances", resp)
    
    instances_data = resp.json() if resp.status_code == 200 else {}
    instances_list = instances_data.get("instances", [])
    
    # 4. Get GitHub History
    print("\nFetching GitHub History (Global)...")
    resp = requests.get(f"{BASE_URL}/github-history", headers=headers)
    print_result("Get GitHub History", resp)
    
    history_data = resp.json() if resp.status_code == 200 else {}

    if instances_list:
        first_instance = instances_list[0]
        # Check instance keys. _id vs instance_id?
        # a_streamlit_app.py uses 'instance_id'. Let's check what API returns.
        inst_id = first_instance.get('instance_id') or first_instance.get('_id')
        
        print(f"\nFetching GitHub History for Instance 1 ({inst_id})...")
        resp = requests.get(f"{BASE_URL}/github-history", params={"instance_id": inst_id}, headers=headers)
        print_result(f"History for Instance {inst_id}", resp)

    print("\nDone.")

if __name__ == "__main__":
    main()
