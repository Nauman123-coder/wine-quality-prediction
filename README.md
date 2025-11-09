# Wine Quality Prediction - Multi-Output Model

## Overview
This project implements a multi-output deep learning model using TensorFlow's Keras Functional API to simultaneously predict wine quality and wine type (red or white) from physicochemical properties. The model treats wine quality prediction as a regression problem and wine type detection as a binary classification problem.

## Dataset
**Source:** UCI Machine Learning Repository - Wine Quality Dataset

The project combines two separate datasets:
- **Red Wine Dataset:** `winequality-red.csv`
- **White Wine Dataset:** `winequality-white.csv`

### Features (11 physicochemical properties):
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

### Targets:
- **Wine Quality:** Continuous values (filtered to range 5-7 after removing imbalanced classes)
- **Wine Type:** Binary classification (Red = 1, White = 0)

### Dataset Preprocessing:
- Removed duplicate entries
- Filtered out imbalanced quality ratings (kept only quality scores 5, 6, and 7)
- Combined red and white wine datasets with type labels
- Data split: 80% training, 20% testing, with 20% of training used for validation

## Model Architecture

### Base Model:
- **Input Layer:** 11 features (normalized)
- **Hidden Layer 1:** Dense layer with 128 neurons, ReLU activation
- **Hidden Layer 2:** Dense layer with 128 neurons, ReLU activation

### Output Layers:
1. **Wine Quality Output:**
   - Dense layer with 1 neuron
   - No activation (linear regression)
   - Loss function: Mean Squared Error (MSE)
   - Metric: Root Mean Squared Error (RMSE)

2. **Wine Type Output:**
   - Dense layer with 1 neuron
   - Sigmoid activation (binary classification)
   - Loss function: Binary Crossentropy
   - Metric: Accuracy

### Model Compilation:
- **Optimizer:** RMSprop (learning rate = 0.0001)
- **Training:** 40 epochs with validation monitoring

## Project Structure
```
.
├── wine_quality_model.ipynb    # Main Jupyter notebook with complete implementation
├── utils.py                    # Utility functions for testing and validation
├── winequality-red.csv         # Red wine dataset
├── winequality-white.csv       # White wine dataset
└── README.md                   # Project documentation
```

## Requirements
```python
tensorflow
keras
numpy
pandas
matplotlib
scikit-learn
```

## Model Performance

### Validation Results:
- **Overall Loss:** ~0.38
- **Wine Quality Loss (MSE):** ~0.35
- **Wine Quality RMSE:** ~0.49
- **Wine Type Loss:** ~0.03
- **Wine Type Accuracy:** ~99.5%

### Key Observations:
- The model achieves exceptional accuracy (~99.5%) in classifying wine type
- Wine quality prediction achieves an RMSE of approximately 0.49, indicating predictions are typically within half a quality point
- The binary classification task (wine type) converges much faster than the regression task (wine quality)

## Usage
1. Open `wine_quality_model.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially to:
   - Load and preprocess the datasets
   - Build and compile the multi-output model
   - Train the model
   - Evaluate performance
   - Visualize results through confusion matrices and scatter plots

## Data Normalization
Features are normalized using the formula:
```
x_norm = (x - mean) / std
```
where mean and standard deviation are computed from the training set.

## Visualization
The notebook includes several visualizations:
- Wine quality distribution histograms
- Training and validation loss curves for both outputs
- Confusion matrix for wine type classification
- Scatter plot comparing true vs. predicted wine quality values

## Model Training Details
- **Training samples:** 3,155
- **Validation samples:** 789
- **Test samples:** 989
- **Batch processing:** Full dataset per epoch
- **Early convergence:** Wine type classification reaches >98% accuracy by epoch 6

## Notes
- The model uses the Keras Functional API, which allows for flexible architectures with multiple inputs and outputs
- Data shuffling was performed during preprocessing to ensure random distribution
- The `utils.py` file contains helper functions for testing and validation purposes
