import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load your acne detection model
model = load_model("C:/Users/TODAY/VSCode/Jerawat/model.h5")  # Gantilah dengan nama dan path model yang sebenarnya

class AcneDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Acne Detection App")

        # UI Components
        self.btn_camera = tk.Button(root, text="Use Camera", command=self.use_camera)
        self.btn_camera.pack(pady=10)

        self.image_path = None
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

        self.video_capture = None
        self.video_source = 0  # Change to another value if using a different camera source

    def use_camera(self):
        # Open a video capture object
        self.video_capture = cv2.VideoCapture(self.video_source)

        # Update image from camera
        self.update_camera()

    def update_camera(self):
        # Capture a frame from the camera
        ret, frame = self.video_capture.read()

        if ret:
            # Convert OpenCV image (BGR format) to Pillow image (RGB format)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            # Update image label
            self.img_label.config(image=img)
            self.img_label.image = img

            # Perform acne detection on the captured frame
            result, acne_probability = self.detect_acne_from_frame(frame)
            self.result_label.config(text=f"Deteksi: {result}, Jerawat Terdeteksi: {acne_probability:.2f}")

            # Schedule the next update
            self.root.after(10, self.update_camera)
        else:
            # Release the video capture object if there's an issue
            self.video_capture.release()

    def detect_acne_from_frame(self, frame):
        # Resize the frame to match the input size expected by the model
        frame = cv2.resize(frame, (224, 224))

        # Preprocess the frame for acne detection
        img_array = img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Perform prediction
        predictions = model.predict(img_array)
        acne_probability = predictions[0][0]

        # Threshold for acne detection (adjust as needed)
        threshold = 0.5
        result = "Acne" if acne_probability > threshold else "No Acne"

        return result, acne_probability

if __name__ == "__main__":
    root = tk.Tk()
    app = AcneDetectionApp(root)
    root.mainloop()
