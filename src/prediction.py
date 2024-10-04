import tensorflow as tf
import numpy as np
import cv2
import argparse

# Load the trained model
model = tf.keras.models.load_model('models/pneumonia_detection_model.h5')


# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Load image with OpenCV
    image = cv2.resize(image, (150, 150))  # Resize to match model input size
    image = image.astype('float32') / 255.0  # Scale the pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Function to make prediction
def predict(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    if prediction[0] > 0.5:
        print(f"The model predicts pneumonia with a probability of {prediction[0][0]:.2f}")
    else:
        print(f"The model predicts normal with a probability of {1 - prediction[0][0]:.2f}")


# Parse command line argument for image path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pneumonia Detection from Chest X-Ray")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the X-ray image")
    args = parser.parse_args()

    # Predict based on the provided image path
    predict(args.image_path)