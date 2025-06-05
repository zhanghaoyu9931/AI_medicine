# Medical Image Classification using Deep Learning

This repository contains a comprehensive demonstration of developing medical image classification algorithms using Convolutional Neural Networks (CNN). It is designed for educational purposes to help students understand the complete pipeline of medical AI development.

## Project Overview

This project demonstrates a complete pipeline for medical image classification, including:

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

4. **Hyperparameter Tuning**
   - Grid search implementation
   - Parameter optimization
   - Results visualization
   - Best model selection

5. **Evaluation and Visualization**
   - Training history plots
   - Confusion matrix
   - Classification metrics
   - Performance analysis

## Project Structure

```
├── data/                      # Raw medical images
│   └── after_processed/       # Preprocessed images
├── res/                       # Results directory
│   └── models/               # Saved models and results
├── preprocessing.py          # Image preprocessing utilities
├── dataset.py               # Dataset and data loading
├── model.py                 # CNN model and training
├── hyperparameter_tuning.py # Grid search implementation
├── main.ipynb              # Main demonstration notebook
└── requirements.txt        # Project dependencies
```

## Getting Started

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   - Place your medical images in the `./data` directory
   - Images should be in common formats (PNG, JPG, JPEG, BMP, TIFF)

3. **Running the Pipeline**
   - Open and run `main.ipynb` in Jupyter Notebook
   - Follow the step-by-step demonstration
   - Each step includes explanations and visualizations

## Key Features

- **Modular Design**: Each component is separated into its own module for better understanding and maintenance
- **Educational Focus**: Detailed comments and markdown explanations throughout the code
- **Visualization**: Comprehensive visualization of preprocessing effects and model performance
- **Flexibility**: Easy to modify parameters and experiment with different configurations
- **Best Practices**: Implements modern deep learning practices and techniques

## Usage for Teaching

This repository is particularly useful for teaching:
- Medical image preprocessing techniques
- Deep learning model development
- Hyperparameter optimization
- Model evaluation and visualization
- Best practices in medical AI development

## Requirements

- Python 3.10+
- PyTorch 2.0.0+
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is designed for educational purposes to help students understand the development of medical AI algorithms. It demonstrates fundamental concepts and best practices in the field. 