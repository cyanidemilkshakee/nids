#!/usr/bin/env python3
"""
Quick test script to verify the API is working correctly.
"""

import requests
import json

API_BASE = "http://localhost:5000"

def test_datasets():
    """Test the datasets endpoint"""
    print("Testing /datasets endpoint...")
    response = requests.get(f"{API_BASE}/datasets")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data['datasets'])} datasets")
        for dataset in data['datasets']:
            print(f"  - {dataset['key']}: {len(dataset['labels'])} labels")
        return data['datasets']
    else:
        print(f"❌ Failed: {response.status_code}")
        return []

def test_features(dataset_key):
    """Test the features endpoint"""
    print(f"\nTesting /features/{dataset_key} endpoint...")
    response = requests.get(f"{API_BASE}/features/{dataset_key}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data['features'])} features for {dataset_key}")
        print(f"  Sample features: {data['features'][:5]}...")
        return data['features']
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return []

def test_prediction(dataset_key, features):
    """Test the prediction endpoint"""
    print(f"\nTesting /predict endpoint with {dataset_key}...")
    
    # Create realistic test data
    test_features = []
    for i, feature in enumerate(features):
        if 'dur' in feature.lower():
            test_features.append(10.5)  # duration
        elif 'byte' in feature.lower():
            test_features.append(1024.0)  # bytes
        elif 'rate' in feature.lower():
            test_features.append(50.0)  # rate
        elif 'count' in feature.lower():
            test_features.append(15.0)  # packet count
        else:
            test_features.append(0.5)  # normalized feature
    
    payload = {
        "dataset": dataset_key,
        "features": test_features
    }
    
    response = requests.post(f"{API_BASE}/predict", 
                           headers={"Content-Type": "application/json"},
                           data=json.dumps(payload))
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Prediction successful!")
        print(f"  Predicted class: {data['prediction']}")
        print(f"  Threat name: {data['threat']}")
        return data
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None

def main():
    print("🧪 Testing AI NIDS API\n")
    
    # Test datasets
    datasets = test_datasets()
    if not datasets:
        return
    
    # Test features and prediction for first dataset
    first_dataset = datasets[0]['key']
    features = test_features(first_dataset)
    if features:
        prediction = test_prediction(first_dataset, features)
        
        if prediction:
            print(f"\n🎉 All tests passed! The API is working correctly.")
            print(f"🌐 Open the dashboard at: http://localhost:5000/dashboard/")
        else:
            print(f"\n⚠️ Prediction failed - check the model loading")
    else:
        print(f"\n⚠️ Features failed - check the data files")

if __name__ == "__main__":
    main()
