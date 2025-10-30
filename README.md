# Bone Fracture Detection AI System

An AI-powered system for detecting bone fractures in X-ray images using ResNet-50, with explainable AI techniques (Grad-CAM, LIME, SHAP) to provide visual heatmaps and textual explanations for model decisions.

## Features

- **Fracture Detection**: Binary classification (fracture/no fracture) using ResNet-50 model
- **Explainability**:
  - Grad-CAM: Class-specific heatmaps highlighting fracture areas
  - LIME: Local interpretable model-agnostic explanations
  - SHAP: Global feature importance and interaction analysis
- **Visual Outputs**: Heatmaps overlaid on original X-rays
- **Textual Explanations**: Confidence-based descriptions of predictions
- **Comparative Analysis**: Jupyter notebook for comparing XAI methods
- **Training Script**: Fine-tune ResNet-50 on custom datasets

## Project Structure

```
bone_fracture_detection/
├── src/
│   ├── model.py          # ResNet-50 model implementation
│   ├── explain.py        # Explainability techniques (Grad-CAM, LIME, SHAP)
│   └── utils.py          # Utility functions for image processing and visualization
├── data/
│   ├── download.py       # Script to download MURA dataset
│   └── sample.jpg        # Placeholder for sample X-ray image
├── notebooks/
│   └── comparative_analysis.ipynb  # Jupyter notebook for XAI comparison
├── models/               # Directory for saved model weights
├── results/              # Directory for output heatmaps and explanations
├── main.py               # Main script for prediction and explanation
├── train.py              # Training script for fine-tuning the model
├── requirements.txt      # Python dependencies
├── TODO.md               # Task completion checklist
└── README.md             # This file
```

## Installation

1. Clone or navigate to the project directory:
   ```bash
   cd bone_fracture_detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

it utilizes a custom dataset containing images of both fractured and non-fractured bones.

1. The dataset structure should be:
   ```
   data/
   ├── fractured/
   └── not_fractured/
   ```

Alternatively, place your own X-ray dataset in the `data/` directory following a similar structure.

## Training

To fine-tune the ResNet-50 model on the custom dataset:

```bash
python train.py
```

This will train the model and save weights to `models/fracture_detection_model.pth`.

## Usage

### Prediction and Explanation

Run the main script with a sample X-ray image:

```bash
python main.py data/sample.jpg --save
```

- `--model_path`: Path to trained model (optional, uses pre-trained ResNet-50 if not provided)
- `--save`: Save heatmap to `results/` directory

The script will:
1. Load the image
2. Make a prediction (Fracture/No Fracture)
3. Generate Grad-CAM heatmap
4. Display results and provide textual explanation

### Comparative Analysis

Open the Jupyter notebook for comparing XAI methods:

```bash
jupyter notebook notebooks/comparative_analysis.ipynb
```

The notebook demonstrates:
- Grad-CAM heatmap generation
- LIME explanations
- SHAP value analysis
- Comparative discussion of each method

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended for training)
- See `requirements.txt` for full list of dependencies

## Model Architecture

- **Base Model**: ResNet-50 pre-trained on ImageNet
- **Fine-tuning**: Last fully connected layer replaced for binary classification
- **Input Size**: 224x224 pixels
- **Output**: 2 classes (No Fracture, Fracture)

## Explainability Techniques

### Grad-CAM
- Provides class-specific saliency maps
- Highlights regions most important for the prediction
- Fast and computationally efficient

### LIME
- Approximates model locally with interpretable model
- Shows superpixels contributing to prediction
- Model-agnostic approach

### SHAP
- Provides global feature importance
- Handles complex feature interactions
- More computationally intensive but comprehensive

## Project Status

- **Dataset:** BoneFractureDataset downloaded and organized into training/testing folders with `Fractured` and `Non-Fractured` subfolders.  
- **Model:** ResNet-50 pre-trained model integrated with `main.py` for predictions.  
- **Explainable AI:** Grad-CAM, LIME, and SHAP explainability implemented.  
- **GUI:** Gradio-based GUI is set up and running for image upload and predictions.  
- **Training:** Model fine-tuning has not been done yet, so predictions currently default to “Fracture” for all images.  

## Results

The system outputs:
- Prediction confidence score
- Visual heatmap overlaid on original X-ray
- Textual explanation based on heatmap intensity
- Comparative analysis of different XAI methods

## Future Improvements

- Support for multi-class fracture classification
- Integration with medical imaging standards (DICOM)
- Real-time inference optimization
- Clinical validation studies
- Additional XAI techniques (e.g., Integrated Gradients)

## License

This project is for educational and research purposes. Ensure compliance with dataset licenses and medical data regulations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## Contact

For questions or issues, please open an issue in the repository.
