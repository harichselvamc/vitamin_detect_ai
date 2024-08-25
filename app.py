from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil

app = FastAPI()

# Load the saved model
model = load_model('1.h5')  # Ensure your model path is correct

# Define a function to preprocess the image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to make predictions
def predict_image(img_array):
    predictions = model.predict(img_array)
    return predictions

# Define the class labels
class_labels = ['Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K']

# Define a function to process predictions
def process_predictions(predictions, class_labels):
    predicted_index = np.argmax(predictions)
    predicted_vitamin = class_labels[predicted_index]
    confidence_score = float(predictions[0][predicted_index])
    return predicted_vitamin, confidence_score

@app.post("/predict-vitamin")
async def predict_vitamin(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    img_path = f"temp_{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img_array = preprocess_image(img_path)
    predictions = predict_image(img_array)
    predicted_vitamin, confidence_score = process_predictions(predictions, class_labels)
    
    return {
        "predicted_vitamin": predicted_vitamin,
        "confidence_score": f"{confidence_score:.2f}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)
