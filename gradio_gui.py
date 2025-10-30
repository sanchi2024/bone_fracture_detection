import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
from src.model import load_model, get_transforms
from src.explain import Explainability
from src.utils import load_image, preprocess_image, predict, save_heatmap

def predict_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model()
    model.to(device)

    # Load and preprocess image
    image = load_image(image_path)
    transforms = get_transforms()
    input_tensor = preprocess_image(image, transforms)

    # Predict
    predicted_class, confidence = predict(model, input_tensor, device)
    class_names = ['No Fracture', 'Fracture']
    prediction = class_names[predicted_class]

    # Explainability
    explainer = Explainability(model, device)

    # Grad-CAM
    grad_cam_attr = explainer.grad_cam(input_tensor, predicted_class)
    heatmap = explainer.generate_heatmap(grad_cam_attr, np.array(image.resize((224, 224))))
    textual_exp = explainer.textual_explanation(grad_cam_attr)

    # Save heatmap
    heatmap_path = 'results/grad_cam_heatmap.png'
    os.makedirs('results', exist_ok=True)
    save_heatmap(heatmap, heatmap_path)

    return prediction, confidence, heatmap_path, textual_exp

def gradio_predict(image):
    if image is None:
        return "No image uploaded", 0.0, None, ""

    # Save uploaded image temporarily
    temp_path = 'temp_image.jpg'
    image.save(temp_path)

    prediction, confidence, heatmap_path, textual_exp = predict_image(temp_path)

    # Load heatmap image
    heatmap_image = Image.open(heatmap_path)

    # Clean up temp file
    os.remove(temp_path)

    return prediction, f"{confidence:.4f}", heatmap_image, textual_exp

# Gradio Interface
with gr.Blocks(title="Bone Fracture Detection") as demo:
    gr.Markdown("# Bone Fracture Detection")
    gr.Markdown("Upload an X-ray image to detect bone fractures using AI. The model will provide a prediction, confidence score, and a Grad-CAM heatmap highlighting the areas of interest.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload X-ray Image")
        submit_btn = gr.Button("Predict")

    with gr.Row():
        prediction_output = gr.Textbox(label="Prediction")
        confidence_output = gr.Textbox(label="Confidence Score")
        heatmap_output = gr.Image(label="Grad-CAM Heatmap")

    explanation_output = gr.Textbox(label="Explanation", lines=2)

    submit_btn.click(
        fn=gradio_predict,
        inputs=image_input,
        outputs=[prediction_output, confidence_output, heatmap_output, explanation_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
