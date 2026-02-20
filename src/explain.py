import torch
import numpy as np
import cv2
from captum.attr import IntegratedGradients, GuidedBackprop, LayerGradCam
from captum.attr import visualization as viz
import lime
import lime.lime_image
# import shap   # ❌ disabled
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

class Explainability:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def grad_cam(self, input_tensor, target_class):
        layer_gc = LayerGradCam(self.model, self.model.resnet.layer4[-1])
        attr = layer_gc.attribute(input_tensor, target=target_class)
        attr = attr.squeeze().cpu().detach().numpy()
        attr = np.maximum(attr, 0)
        attr = cv2.resize(attr, (224, 224))
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        return attr

    def lime_explanation(self, image, predict_fn, num_samples=1000):
        explainer = lime.lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image.astype('double'),
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples
        )
        return explanation

    # ❌ SHAP DISABLED (Windows build issue)
    # def shap_explanation(self, background, test_images):
    #     explainer = shap.DeepExplainer(self.model, background)
    #     shap_values = explainer.shap_values(test_images)
    #     return shap_values

    def generate_heatmap(self, attr, original_image):
        heatmap = cv2.applyColorMap(np.uint8(255 * attr), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        original_image = np.float32(original_image) / 255
        superimposed_img = heatmap * 0.4 + original_image * 0.6
        return superimposed_img

    def textual_explanation(self, attr, threshold=0.5):
        max_intensity = np.max(attr)
        if max_intensity > threshold:
            return "High confidence fracture detected in highlighted areas."
        else:
            return "Low confidence or no fracture detected."
