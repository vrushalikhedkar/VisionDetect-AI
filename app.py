
from flask import Flask, render_template, request
from pathlib import Path
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
import io
import base64
import time

app = Flask(__name__)

# --------------------------------------------------
# 1. LOAD SAVED MODEL
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "object_detection_saved"

model = tf.saved_model.load(str(MODEL_PATH))

print("Model loaded successfully!")
print("Available signatures:", list(model.signatures.keys()))

detector = model.signatures["default"]

print("Input signature:")
print(detector.structured_input_signature)

print("Output signature:")
for name, spec in detector.structured_outputs.items():
    print(name, spec.shape, spec.dtype)


# --------------------------------------------------
# 2. HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# 3. OBJECT DETECTION
# --------------------------------------------------

@app.route("/detect", methods=["POST"])
def detect():

    uploaded_file = request.files.get("image")

    if uploaded_file is None or uploaded_file.filename == "":
        return "Please upload an image.", 400

    try:
        start_time = time.time()

        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.asarray(image)

        # Convert image to float32 in range 0-1
        input_tensor = tf.convert_to_tensor(
            image_np, dtype=tf.float32
        ) / 255.0

        # Add batch dimension
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        # Run model
        raw_results = detector(images=input_tensor)

        # Debug: show output shapes
        print("\n--- MODEL OUTPUT SHAPES ---")
        for name, value in raw_results.items():
            print(name, value.shape, value.dtype)

        # Extract scores
        scores = np.asarray(
            raw_results["detection_scores"].numpy()
        ).reshape(-1)

        # Extract bounding boxes
        boxes = np.asarray(
            raw_results["detection_boxes"].numpy()
        ).reshape(-1, 4)

        # Extract class labels safely
        raw_labels = raw_results["detection_class_entities"].numpy()

        labels = np.asarray(raw_labels).reshape(-1)

        decoded_labels = []

        for label in labels:
            if isinstance(label, bytes):
                label = label.decode("utf-8", errors="replace")

            decoded_labels.append(str(label))

        labels = decoded_labels

        print("Number of scores:", len(scores))
        print("Number of boxes:", len(boxes))
        print("Number of labels:", len(labels))

        # Prepare image for drawing
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)

        width, height = output_image.size
        detections = []

        # Confidence threshold
        confidence_threshold = 0.30

        # Process predictions from highest confidence
        for i in np.argsort(scores)[::-1]:

            score = float(scores[i])

            if score < confidence_threshold:
                continue

            # Make sure corresponding box and label exist
            if i >= len(boxes) or i >= len(labels):
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            # Convert normalized coordinates to pixels
            x1 = max(0, min(width - 1, int(xmin * width)))
            y1 = max(0, min(height - 1, int(ymin * height)))
            x2 = max(0, min(width - 1, int(xmax * width)))
            y2 = max(0, min(height - 1, int(ymax * height)))

            if x2 <= x1 or y2 <= y1:
                continue

            label = labels[i]

            # Draw bounding box
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="lime",
                width=3
            )

            # Draw label and confidence
            draw.text(
                (x1, max(0, y1 - 20)),
                f"{label}: {score:.2f}",
                fill="red",
                stroke_width=1,
                stroke_fill="white"
            )

            detections.append({
                "label": label,
                "score": round(score, 3)
            })

        # Encode output image
        buffer = io.BytesIO()

        output_image.save(
            buffer,
            format="JPEG",
            quality=85
        )

        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        processing_time = round(time.time() - start_time, 2)

        print("Detected objects:", len(detections))
        print("Processing time:", processing_time, "seconds")

        # Display results
        return render_template(
            "result.html",
            result_image=encoded_image,
            detections=detections,
            total_detections=len(detections),
            processing_time=processing_time
        )

    except Exception:
        app.logger.exception("Object detection failed")
        return "Image processing failed. Please check the PyCharm console.", 500


# --------------------------------------------------
# 4. RUN FLASK APP
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False)
