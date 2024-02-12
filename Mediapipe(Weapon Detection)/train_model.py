import os
import json
import tensorflow as tf
from mediapipe_model_maker import object_detector
from mediapipe_model_maker import quantization

assert tf.__version__.startswith('2')

# Directory paths
train_dataset_path = "Training Data"
validation_dataset_path = "Validation Data"
export_dir = 'exported_model'

# Function to print dataset categories
def print_categories(path):
    with open(os.path.join(path, "labels.json"), "r") as f:
        labels_json = json.load(f)
    for category_item in labels_json["categories"]:
        print(f"{category_item['id']}: {category_item['name']}")

# Load and print training data categories
print("Training data categories:")
print_categories(train_dataset_path)

# Load and print validation data categories
print("Validation data categories:")
print_categories(validation_dataset_path)

# Load datasets
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)

# Model specification
spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
hparams = object_detector.HParams(export_dir=export_dir)
options = object_detector.ObjectDetectorOptions(supported_model=spec, hparams=hparams)

# Train model
model = object_detector.ObjectDetector.create(train_data=train_data, validation_data=validation_data, options=options)

# Evaluate model
loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")

# Export model (without quantization)
model.export_model(export_dir=os.path.join(export_dir, "model.tflite"))

# Optionally apply and export quantized model (e.g., for float16)
quantization_config = quantization.QuantizationConfig.for_float16()
model.export_model(model_name="model_fp16.tflite", quantization_config=quantization_config, export_dir=export_dir)

print("Model and potentially quantized versions have been saved to:", export_dir)

