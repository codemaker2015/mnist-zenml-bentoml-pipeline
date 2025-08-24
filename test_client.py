import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

# existing helper (unchanged)
def create_sample_mnist_image():
    img = np.zeros((28, 28), dtype=np.float32)
    img[8:20, 13:15] = 1.0
    img[8:10, 11:13] = 1.0
    img[18:20, 11:17] = 1.0
    return img

def test_bentoml_service():
    BASE_URL = "http://localhost:3000"

    print("🚀 Testing BentoML MNIST Service")
    print("=" * 60)

    # 1. Health check (same)
    print("1️⃣  Testing health endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Service is healthy!")
            print(f"   Response: {response.json()}")
        else:
            print("❌ Service health check failed", response.status_code)
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return

    print("-" * 60)

    # 2. Test with numpy array (same)
    print("2️⃣  Testing prediction with numpy array...")
    try:
        test_image = create_sample_mnist_image()
        plt.figure(figsize=(4, 4))
        plt.imshow(test_image, cmap='gray')
        plt.title("Test Image (Sample '1')")
        plt.axis('off')
        # plt.show()

        data = {"inp": test_image.tolist()}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful! {result}")
        else:
            print(f"❌ Prediction failed: {response.status_code} {response.text}")

    except Exception as e:
        print(f"❌ Array prediction error: {e}")

    print("-" * 60)

    # 3. Test with existing file from disk (no creation, no explicit filename)
    print("3️⃣  Testing prediction by uploading an existing file from disk (no creation)...")
    # <-- EDIT THIS to point to your absolute file path (Windows or Linux)
    FILE_PATH = r"test.png"  # e.g. "C:\\Users\\You\\mnist.png"
    # FILE_PATH = "/mnt/data/mnist_resized_28x28.png"      # or Unix-style path

    if not os.path.isfile(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
    else:
        try:
            # open file in binary mode and send as multipart/form-data
            # note: using an empty string as the form field name per your request
            with open(FILE_PATH, "rb") as f:
                files = {"": f}  # empty form-field name
                # increase timeout in case server inference is slow
                response = requests.post(f"{BASE_URL}/predict_image", files=files, timeout=30)

            if response.status_code == 200:
                print("✅ Image prediction successful!")
                print("   Response:", response.json())
            else:
                print(f"❌ Image prediction failed: {response.status_code}")
                print("   Error:", response.text)

        except Exception as e:
            print(f"❌ Image prediction error: {e}")

    print("=" * 60)
    print("🎉 Testing completed!")

if __name__ == "__main__":
    test_bentoml_service()
