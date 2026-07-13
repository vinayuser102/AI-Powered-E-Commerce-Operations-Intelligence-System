import requests

# The base URL pointing directly to your live local Uvicorn engine
BASE_URL = "http://127.0.0.1:8000"

print("📡 Initiating connection to NexusRisk Core API...")

# 1. Test Case A: A high-risk customer profile (Clean Data)
customer_a_id = "CU-9982"
customer_a_payload = {
    "recency": 2,          # Purchased very recently
    "frequency": 25,       # Bought many times
    "monetary_value": 1450.50,
    "refund_rate": 0.05    # Low refunds
}

print(f"\n🔄 Sending payload for Customer {customer_a_id}...")
response_a = requests.post(f"{BASE_URL}/predict/{customer_a_id}", json=customer_a_payload)

if response_a.status_code == 200:
    print("✅ Success! Response Received from ML Engine:")
    print(response_a.json())
else:
    print(f"❌ Failed with Status Code: {response_a.status_code}")
    print(response_a.json())