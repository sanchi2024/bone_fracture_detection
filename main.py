import torch
import argparse
from src.model import load_model, get_transforms
from src.explain import Explainability
from src.utils import load_image, preprocess_image, predict, display_image_with_heatmap, save_heatmap
import numpy as np

def main(image_path, model_path=None, save_results=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path)
    model.to(device)

    # Load and preprocess image
    image = load_image(image_path)
    transforms = get_transforms()
    input_tensor = preprocess_image(image, transforms)

    # Predict
    predicted_class, confidence = predict(model, input_tensor, device)
    class_names = ['No Fracture', 'Fracture']
    print(f"Prediction: {class_names[predicted_class]} with confidence {confidence:.4f}")

    # Explainability
    explainer = Explainability(model, device)

    # Grad-CAM
    grad_cam_attr = explainer.grad_cam(input_tensor, predicted_class)
    heatmap = explainer.generate_heatmap(grad_cam_attr, np.array(image.resize((224, 224))))
    textual_exp = explainer.textual_explanation(grad_cam_attr)

    print(f"Textual Explanation: {textual_exp}")

    # Display results
    display_image_with_heatmap(np.array(image), heatmap, "Grad-CAM Heatmap")

    if save_results:
        save_heatmap(heatmap, 'results/grad_cam_heatmap.png')
        print("Heatmap saved to results/grad_cam_heatmap.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bone Fracture Detection with Explainability")
    parser.add_argument("image_path", help="Path to the X-ray image")
    parser.add_argument("--model_path", help="Path to the trained model (optional)")
    parser.add_argument("--save", action="store_true", help="Save results to files")
    args = parser.parse_args()

    main(args.image_path, args.model_path, args.save)
