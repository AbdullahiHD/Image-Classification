This repository explores emotion detection from images using transfer learning with two pre-trained models: MobileNetV2 and ResNet18. We compare their performance on binary (happy/sad) and quadry (happy, sad, angry, neutral) classification tasks.

Key Points:

Transfer Learning: Leverages pre-trained MobileNetV2 and ResNet18 models for feature extraction.
Classification Tasks: Evaluates performance on both binary (happy/sad) and quadry (happy, sad, surprise angry) emotions.
No Ensemble Learning: Models are trained and evaluated independently, not combined.
Getting Started:

Prerequisites: Ensure you have Python, TensorFlow, and OpenCV installed.
Data: Prepare your emotion-labeled image dataset. The code assumes a specific folder structure for training and validation data. Refer to scripts for details.
Run Experiments: Execute the Python scripts in the scripts folder to train and evaluate both MobileNetV2 and ResNet18 models on binary and quadry classification tasks.
Results:

The repository includes scripts to generate performance reports comparing the accuracy, precision, recall, and F1-score for each model on both classification tasks.

Further Exploration:

Experiment with hyperparameter tuning for each model.
Explore additional pre-trained models for comparison.
Implement data augmentation techniques to improve generalization.
Consider ensemble methods to potentially improve overall performance.

Note: This repository focuses on individual model evaluation and does not explore ensemble learning techniques for combining MobileNetV2 and ResNet18.
