# Medical Image Diagnosis using Deep Learning

This repository contains a comprehensive demonstration of developing medical image diagnosis algorithms using Convolutional Neural Networks (CNN). It is designed for educational purposes to help students understand the complete pipeline of medical AI development and includes advanced features for scientific research.

## Project Overview

This project demonstrates a complete pipeline for medical image diagnosis (both classification and regression tasks), including:

1. **Image Preprocessing**
   - Histogram equalization
   - Contrast enhancement
   - Gaussian smoothing
   - Image resizing
   - Visualization of preprocessing effects

2. **Dataset Management**
   - Custom dataset class for medical images
   - Train/validation/test split
   - Data augmentation
   - DataLoader implementation

3. **Model Architecture**
   - Custom CNN architecture
   - Training pipeline
   - Early stopping
   - Model checkpointing
   - Class name mapping for better visualization

4. **Hyperparameter Tuning**
   - Grid search implementation
   - Parameter optimization
   - Results visualization
   - Best model selection

5. **Model Comparison & Benchmarking**
   - Deep Learning (CNN) vs Traditional ML methods
   - SVM, Random Forest, Logistic Regression comparison
   - Feature extraction from pre-trained CNN
   - Scientific-quality performance visualization

6. **Advanced Evaluation and Visualization**
   - Training history plots
   - Confusion matrix with custom class names
   - Classification metrics (Accuracy, Precision, Recall, F1)
   - ROC and Precision-Recall curves
   - Scientific paper-quality figures
   - Performance comparison charts

## Project Structure

```
├── data/                           # Medical image data
│   ├── ori_images/                # Original raw medical images
│   ├── after_processed/           # Preprocessed images
│   └── labels.csv                 # Dataset labels and metadata
├── results/                        # Results and outputs directory
│   ├── best_model/                # Best trained model artifacts
│   │   ├── best_model.pth         # Saved model weights
│   │   ├── training_history.json  # Training metrics history
│   │   ├── training_history.png   # Training curves visualization
│   │   └── confusion_matrix.png   # Model confusion matrix
│   ├── grid_search/               # Hyperparameter tuning results
│   │   ├── grid_search_summary.json    # Complete search results
│   │   ├── grid_search_results.png     # Visualization of results
│   │   ├── combination_X_results.json  # Individual combination results
│   │   └── combination_X/              # Detailed results per combination
│   ├── model_evaluation/          # Model comparison results
│   │   ├── model_comparison_results.json  # Numerical comparison data
│   │   ├── accuracy_comparison.png       # Accuracy comparison chart
│   │   ├── precision_comparison.png      # Precision comparison chart
│   │   ├── recall_comparison.png         # Recall comparison chart
│   │   └── f1_comparison.png             # F1-score comparison chart
│   └── confusion_matrix.png       # General confusion matrix
├── preprocessing.py                # Image preprocessing utilities
├── dataset.py                     # Dataset classes and data loading
├── model.py                       # CNN model, training, and comparison
├── hyperparameter_tuning.py       # Grid search implementation
├── main.ipynb                     # Main demonstration notebook
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
- Place your medical images in the `./data` directory
- Images should be in common formats (PNG, JPG, JPEG, BMP, TIFF)
- Organize by class if doing classification

### 3. Running the Pipeline

#### Basic Usage
```python
# Run main.ipynb in Jupyter Notebook
# Follow the step-by-step demonstration
```

#### For Classification Tasks
```python
# Configure for binary classification
config = {
    'task_type': 'classification',
    'num_classes': 2,
    'class_names': {0: 'Healthy', 1: 'Diseased'}
}
```

#### For Regression Tasks
```python
# Configure for regression
config = {
    'task_type': 'regression',
    'target_variable': 'age'  # or any continuous variable
}
```

#### Advanced Configuration
```python
# Set class names for better visualization
class_names = {0: 'Normal', 1: 'Abnormal', 2: 'Disease'}

# Control debug output
DEBUG_MODE = False  # Set to True for detailed logging

# Model comparison
comparison_results = compare_models_performance(
    best_cnn_trainer=trainer,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    save_dir='./results/model_evaluation',
    task_type='classification',
    class_names=class_names,
    debug=DEBUG_MODE
)
```

## Model Comparison Features

### 🤖 Supported Models
- **Deep Learning**: Custom CNN architecture
- **Traditional ML**: 
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - Linear Regression (for regression tasks)

### 📊 Generated Outputs
- **Figures** (saved to `./results/model_evaluation/`):
  - `aupr_comparison.png` - Precision-Recall curves
  - `auc_comparison.png` - ROC curves  
  - `accuracy_comparison.png` - Accuracy comparison
  - `precision_comparison.png` - Precision comparison
  - `recall_comparison.png` - Recall comparison
  - `f1_comparison.png` - F1-score comparison
  - `model_comparison_results.json` - Numerical results

### 🎯 Scientific Paper Integration
All figures are optimized for direct use in scientific papers:
- **High Resolution**: 300 DPI for print quality
- **Professional Fonts**: Times New Roman serif fonts
- **Compact Size**: 4×3 inches for bar charts, 6×5 for curves
- **Clean Styling**: Minimal, professional aesthetics
- **Proper Labeling**: Bold labels and appropriate sizing

## Educational Applications

This repository is particularly valuable for teaching:
- **Medical Image Analysis**: Preprocessing and feature extraction
- **Deep Learning**: CNN architecture and training
- **Model Comparison**: Traditional ML vs Deep Learning
- **Scientific Visualization**: Research-quality figure generation
- **Performance Evaluation**: Comprehensive metrics and interpretation
- **Best Practices**: Modern AI development workflows

## Requirements

- Python 3.10+
- PyTorch 2.0.0+
- OpenCV 4.5+
- NumPy 1.21+
- Pandas 1.3+
- Matplotlib 3.5+
- scikit-learn 1.0+
- seaborn 0.11+
- Jupyter Notebook

## Output Quality Standards

### Figure Specifications
- **Resolution**: 300 DPI (publication ready)
- **Format**: PNG with transparent backgrounds
- **Fonts**: Times New Roman (scientific standard)
- **Colors**: Professional, colorblind-friendly palette
- **Size**: Optimized for journal submissions

### Performance Metrics
- **Classification**: Accuracy, Precision, Recall, F1, AUC, AUPR
- **Regression**: MSE, MAE, R²
- **Statistical Significance**: Consistent evaluation protocols
- **Cross-Model Comparison**: Fair benchmarking methodology

## Contributing

We welcome contributions that enhance the educational value or research capabilities:
- Bug fixes and improvements
- Additional model architectures
- New visualization features
- Documentation enhancements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{AI_medicine,
  title={When AI meets medicine – algorithms are transforming healthcare issues},
  author={Hauser Zhang},
  year={2025},
  url={https://github.com/zhanghaoyu9931/AI_medicine}
}
```

## Acknowledgments

This project is designed for educational and research purposes, demonstrating state-of-the-art practices in medical AI development. It incorporates best practices from both academic research and industry applications. 