# Image Classification with TensorFlow and MobileNetV2

## Project Overview

This project demonstrates how to build, train, and deploy an image classification model using TensorFlow and the MobileNetV2 architecture. It utilizes transfer learning to efficiently classify images into five categories. The pipeline includes data preprocessing, augmentation, model training, validation, and prediction on unseen test data.

---

## Features

- **Custom dataset loading**: Reads image paths and labels from CSV files.
- **Data augmentation**: Applies random horizontal flips to increase training data diversity.
- **Transfer learning**: Uses MobileNetV2 pretrained on ImageNet as a feature extractor.
- **Fine-tuning**: Adds custom dense layers and trains only the classifier head.
- **Efficient training and validation**: Uses `ImageDataGenerator` for streamlined data input.
- **Test prediction and submission**: Generates predictions on test images and prepares a submission CSV.

---

## Setup

### Requirements

- Python 3.x
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn

You can install dependencies using:

```bash
pip install tensorflow pandas numpy scikit-learn
```

---

## Usage

1. **Prepare your data**  
   Organize your images in folders (e.g., `data/train`, `data/validation`, `data/test`) and prepare CSV files (`train.csv`, `validation.csv`, `test.csv`) with columns:

   - `image_id` (filename without extension)
   - `label` (integer class label, except for test where label is dummy)

2. **Run the notebook/script**  
   Execute the Jupyter Notebook or Python script to train the model and generate predictions.

3. **Output**  
   The script will save a `submission.csv` file with predicted labels for the test set, ready for evaluation or submission.

---

## Code Structure

- **Data loading and preprocessing**: Reads CSVs, builds file paths, creates data generators with augmentation.
- **Model construction**: Loads MobileNetV2 without the top layer, adds pooling and dense layers.
- **Training loop**: Compiles and trains the model using the training and validation data.
- **Prediction and export**: Runs inference on the test data and saves predictions.

---

## Future Improvements

- **Fine-tune the base model layers** for potentially better accuracy.
- **Add more augmentation techniques** (rotation, zoom, etc.) to improve robustness.
- **Implement early stopping** or learning rate schedules to optimize training.
- **Experiment with other architectures** like EfficientNet or ResNet.
