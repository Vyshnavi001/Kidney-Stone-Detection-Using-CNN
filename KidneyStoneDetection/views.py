from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
import os
import random

# Load the trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'KidneyStoneDetection', 'model', 'best_kidney_stone_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Predict kidney stone presence and simulate report data
def predict_kidney_stone(img_path):
    try:
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            stone_count = random.randint(1, 3)
            details = []
            for i in range(stone_count):
                size = round(random.uniform(3.0, 12.0), 1)
                location = random.choice(["Left Kidney", "Right Kidney", "Left Ureter", "Bladder"])
                details.append({"size": f"{size} mm", "location": location})

            return {
                "status": "detected",
                "message": "✅ Kidney Stone(s) Detected",
                "stones": details
            }
        else:
            return {
                "status": "clear",
                "message": "🟢 No Kidney Stone Is Detected"
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"⚠️ Error during prediction: {str(e)}"
        }

# Homepage view for handling image upload and prediction
def homepage(request):
    user = request.session.get('user')
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'pastreports'))
        filename = fs.save(image.name, image)
        filepath = fs.path(filename)
        result = predict_kidney_stone(filepath)
        return JsonResponse(result)
    return render(request, 'homepage.html', {'user': user})

# Login view
def login(request):
    return render(request, 'login.html')
