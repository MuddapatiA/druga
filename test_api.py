import requests
import json

# The URL of the running FastAPI application
API_URL = "http://127.0.0.1:8000/predict"

# Example patient data matching the API's expected input
patient_data = {
  "patient_age": 68,
  "patient_gender": "f",
  "num_conditions": 2,
  "physician_type": "general practice",
  "physician_state": "ny",
  "location_type": "clinic",
  "num_contraindications": 1,
  "patient_is_high_risk": 1
}

print(f"▶️  Sending data to API at {API_URL}")
print("Input data:", json.dumps(patient_data, indent=2))

try:
    # Send the POST request
    response = requests.post(API_URL, json=patient_data)

    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # Print the JSON response from the API
    print("\n✅ Success! API Response:")
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.RequestException as e:
    print(f"\n❌ Error: Could not connect to the API. Please ensure the server is running.")
    print(f"Details: {e}")