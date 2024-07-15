# Forest-Fire-Detection

## Overview

This project implements a deep learning model to detect forest fires from images. It uses a convolutional neural network based on the EfficientNetB0 architecture to classify images as either containing a fire or not.

## Dataset

The dataset used in this project contains over 13,000 images divided into two categories:
- Fire (images showing forest fires)
- No Fire (images without forest fires)

The dataset is split into training/validation and testing sets.

## Project Structure

The project follows this workflow:

1. Data Loading and Extraction
2. Data Augmentation and Preprocessing
3. Model Architecture Definition
4. Model Training
5. Model Evaluation
6. Results Visualization

## Key Features

- Utilizes EfficientNetB0 as the base model for transfer learning
- Implements data augmentation techniques to improve model generalization
- Uses advanced callbacks for learning rate reduction and early stopping
- Provides detailed evaluation metrics including confusion matrix and ROC curve

## Installation

To set up the project environment:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/forest-fire-detection.git
   ```
2. Navigate to the project directory:
   ```
   cd forest-fire-detection
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the project:

1. Open the Jupyter notebook:
   ```
   jupyter notebook Forest_Fire_Detection.ipynb
   ```
2. Run the cells in the notebook sequentially to perform data preprocessing, model training, and evaluation.

## Model Architecture

The model architecture is based on EfficientNetB0 with additional layers:
- Global Average Pooling
- Batch Normalization
- Dense layer with ReLU activation
- Dropout for regularization
- Output Dense layer with sigmoid activation for binary classification

## Results

The model achieves high accuracy in detecting forest fires. Key performance metrics include:

### Confusion Matrix
```
             Predicted No Fire  Predicted Fire
Actual No Fire        1740            123
Actual Fire            281           2849
```

### Classification Report
```
             Precision    Recall  F1-Score   Support
Fire            0.86       0.93      0.90      1863
No Fire         0.96       0.91      0.93      3130
Accuracy                             0.92      4993
Macro Avg       0.91       0.92      0.91      4993
Weighted Avg    0.92       0.92      0.92      4993
```

The model demonstrates strong performance with an overall accuracy of 92%. It shows particularly high precision in detecting non-fire images (96%) and high recall for fire images (93%), indicating a good balance between minimizing false positives and false negatives.

Detailed visualizations of the model's performance, including loss/accuracy curves and ROC curve, are provided in the notebook.

## Future Improvements

- Experiment with other pre-trained models or ensemble methods
- Collect more diverse data to improve model generalization
- Implement real-time fire detection using video feeds

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is sourced from [provide source if known]
- Thanks to the TensorFlow and Keras teams for their excellent deep learning frameworks
