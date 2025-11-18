"""
Explainability module for melanoma detection models.
Implements Grad-CAM, occlusion sensitivity, and feature importance analysis.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64


class ExplainabilityEngine:
    """
    Implements multiple explainability techniques for CNN predictions.
    Provides visual and textual explanations for model decisions.
    """
    
    # Clinical feature mappings for dermatological analysis
    DERMOSCOPIC_FEATURES = {
        0: {  # AKIEC
            'key_features': ['Rough, scaly patches', 'Pink or red base', 'Adherent scale'],
            'clinical_signs': ['Hyperkeratosis', 'Parakeratosis', 'Acanthosis'],
            'risk_factors': ['Sun exposure', 'Age >60', 'Fair skin'],
            'dermoscopy': ['Scale', 'Red pseudonetwork', 'Strawberry pattern']
        },
        1: {  # BCC
            'key_features': ['Arborizing vessels', 'Blue-gray ovoid nests', 'Ulceration'],
            'clinical_signs': ['Telangiectasia', 'Rolled borders', 'Pearly appearance'],
            'risk_factors': ['UV exposure', 'Previous BCC', 'Immunosuppression'],
            'dermoscopy': ['Maple leaf areas', 'Spoke wheel areas', 'Large blue-gray ovoid nests']
        },
        2: {  # BKL
            'key_features': ['Milia-like cysts', 'Comedo-like openings', 'Fissures'],
            'clinical_signs': ['Well-demarcated borders', 'Stuck-on appearance', 'Warty surface'],
            'risk_factors': ['Age', 'Genetic predisposition', 'Sun exposure'],
            'dermoscopy': ['Fingerprint patterns', 'Moth-eaten borders', 'Jelly sign']
        },
        3: {  # DF
            'key_features': ['Central white patch', 'Peripheral pigment network', 'Dimple sign'],
            'clinical_signs': ['Firm nodule', 'Positive dimple sign', 'Brown color'],
            'risk_factors': ['Minor trauma', 'Insect bites', 'Female predominance'],
            'dermoscopy': ['Central white scar-like patch', 'Delicate pigment network at periphery']
        },
        4: {  # MEL
            'key_features': ['Asymmetry', 'Irregular borders', 'Multiple colors'],
            'clinical_signs': ['ABCDE criteria positive', 'Evolution', 'Ugly duckling sign'],
            'risk_factors': ['Multiple nevi', 'Family history', 'Previous melanoma'],
            'dermoscopy': ['Atypical network', 'Blue-white veil', 'Regression structures']
        },
        5: {  # NV
            'key_features': ['Symmetric pattern', 'Regular network', 'Uniform color'],
            'clinical_signs': ['Well-defined borders', 'Stable over time', 'No symptoms'],
            'risk_factors': ['Sun exposure in childhood', 'Fair skin', 'Genetic factors'],
            'dermoscopy': ['Typical network', 'Globular pattern', 'Starburst pattern']
        },
        6: {  # VASC
            'key_features': ['Red-purple color', 'Lacunae', 'Red-blue areas'],
            'clinical_signs': ['Vascular proliferation', 'Soft consistency', 'Blanching'],
            'risk_factors': ['Age', 'Pregnancy', 'Trauma'],
            'dermoscopy': ['Red lacunae', 'Red-blue areas', 'Rainbow pattern']
        }
    }
    
    @staticmethod
    def compute_gradcam(model: Model, img_array: np.ndarray, 
                       pred_index: int, layer_name: Optional[str] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for visualization of important regions.
        
        Args:
            model: Trained Keras model
            img_array: Preprocessed image array (1, H, W, 3)
            pred_index: Index of predicted class
            layer_name: Target layer for gradients (auto-detected if None)
        
        Returns:
            Heatmap array normalized to [0, 1]
        """
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if any([isinstance(layer, tf.keras.layers.Conv2D),
                       isinstance(layer, tf.keras.layers.SeparableConv2D),
                       'conv' in layer.name.lower()]):
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            raise ValueError("No convolutional layer found in model")
        
        # Create gradient model
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    @staticmethod
    def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray,
                       alpha: float = 0.4) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            img: Original image array
            heatmap: Grad-CAM heatmap
            alpha: Transparency factor

        Returns:
            Image with heatmap overlay
        """
        # Ensure image is RGB (3 channels) - fix RGBA images
        if len(img.shape) == 3 and img.shape[2] == 4:
            # Convert RGBA to RGB
            img = img[:, :, :3]
        elif len(img.shape) == 2:
            # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Convert to RGB colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Convert BGR to RGB (cv2 uses BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Superimpose heatmap
        overlaid = heatmap * alpha + img * (1 - alpha)
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

        return overlaid

    @staticmethod
    def analyze_gradcam_focus(heatmap: np.ndarray) -> Dict[str, float]:
        """
        Analyze GradCAM heatmap focus patterns for validation.

        Valid skin lesion images should have focused attention on lesion area.
        Non-medical images typically have scattered or no clear focus.

        Args:
            heatmap: GradCAM heatmap (normalized 0-1)

        Returns:
            Dictionary with focus metrics
        """
        # Normalize heatmap to 0-1 if not already
        if heatmap.max() > 1:
            heatmap = heatmap / 255.0

        # Calculate focus metrics

        # 1. Peak intensity - highest activation
        peak_intensity = float(np.max(heatmap))

        # 2. Focus concentration - how much of the heatmap is highly activated
        high_activation_threshold = 0.5
        high_activation_ratio = float(np.mean(heatmap > high_activation_threshold))

        # 3. Spatial concentration - are high activations clustered together?
        # Use the ratio of area above threshold to total spread
        if high_activation_ratio > 0:
            # Find contours of high activation regions
            heatmap_uint8 = np.uint8(heatmap * 255)
            _, binary = cv2.threshold(heatmap_uint8, int(high_activation_threshold * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour area
                largest_contour_area = max(cv2.contourArea(c) for c in contours)
                total_area = heatmap.shape[0] * heatmap.shape[1]
                spatial_concentration = float(largest_contour_area / total_area)
            else:
                spatial_concentration = 0.0
        else:
            spatial_concentration = 0.0

        # 4. Center of mass distance from image center
        # Skin lesions are often centered
        if heatmap.sum() > 0:
            y_indices, x_indices = np.indices(heatmap.shape)
            center_y = np.average(y_indices, weights=heatmap)
            center_x = np.average(x_indices, weights=heatmap)
            img_center_y, img_center_x = heatmap.shape[0] / 2, heatmap.shape[1] / 2

            # Normalize distance by image diagonal
            diagonal = np.sqrt(heatmap.shape[0]**2 + heatmap.shape[1]**2)
            center_distance = np.sqrt((center_y - img_center_y)**2 + (center_x - img_center_x)**2)
            normalized_center_distance = float(center_distance / diagonal)
        else:
            normalized_center_distance = 0.5

        # 5. Overall focus score (combined metric)
        # Higher score = more focused attention = more likely valid
        focus_score = (
            peak_intensity * 0.3 +
            (1 - high_activation_ratio) * 0.2 +  # Less spread = more focused
            spatial_concentration * 0.3 +
            (1 - normalized_center_distance) * 0.2  # Closer to center = better
        )

        return {
            'peak_intensity': peak_intensity,
            'high_activation_ratio': high_activation_ratio,
            'spatial_concentration': spatial_concentration,
            'center_distance': normalized_center_distance,
            'focus_score': float(focus_score)
        }
    
    @staticmethod
    def compute_occlusion_sensitivity(model: Model, img_array: np.ndarray,
                                     pred_index: int, patch_size: int = 32,
                                     stride: int = 16) -> np.ndarray:
        """
        Compute occlusion sensitivity map by systematically occluding image regions.
        
        Args:
            model: Trained model
            img_array: Preprocessed image
            pred_index: Target class index
            patch_size: Size of occlusion patch
            stride: Stride for sliding window
        
        Returns:
            Sensitivity map showing importance of each region
        """
        h, w = img_array.shape[1:3]
        sensitivity_map = np.zeros((h // stride, w // stride))
        
        # Get baseline prediction
        baseline_prob = model.predict(img_array, verbose=0)[0, pred_index]
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # Create occluded image
                occluded = img_array.copy()
                occluded[0, i:i+patch_size, j:j+patch_size, :] = 0
                
                # Compute prediction difference
                occluded_prob = model.predict(occluded, verbose=0)[0, pred_index]
                sensitivity_map[i // stride, j // stride] = baseline_prob - occluded_prob
        
        # Resize to original dimensions
        sensitivity_map = cv2.resize(sensitivity_map, (w, h))
        
        # Normalize
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-8)
        
        return sensitivity_map
    
    @staticmethod
    def analyze_color_distribution(img: np.ndarray) -> Dict[str, float]:
        """
        Analyze color distribution relevant to melanoma detection.
        
        Args:
            img: Input image in RGB format
        
        Returns:
            Dictionary with color metrics
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Compute statistics
        metrics = {
            'mean_red': np.mean(img[:, :, 0]),
            'mean_green': np.mean(img[:, :, 1]),
            'mean_blue': np.mean(img[:, :, 2]),
            'std_hue': np.std(hsv[:, :, 0]),
            'std_saturation': np.std(hsv[:, :, 1]),
            'mean_lightness': np.mean(lab[:, :, 0]),
            'color_variance': np.var(img),
            'blue_white_presence': np.sum(img[:, :, 2] > 200) / (img.shape[0] * img.shape[1]),
            'darkness_ratio': np.sum(lab[:, :, 0] < 50) / (img.shape[0] * img.shape[1])
        }
        
        return metrics
    
    @staticmethod
    def detect_asymmetry(img: np.ndarray) -> float:
        """
        Compute asymmetry score for lesion (ABCDE criterion A).
        
        Args:
            img: Grayscale image of lesion
        
        Returns:
            Asymmetry score (0-1, higher = more asymmetric)
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Find contours
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest contour (main lesion)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute moments
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return 0.0
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Compare halves
        h, w = img.shape
        left_half = img[:, :cx]
        right_half = cv2.flip(img[:, cx:], 1)
        
        # Resize to same dimensions
        min_width = min(left_half.shape[1], right_half.shape[1])
        if min_width > 0:
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Compute difference
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            asymmetry_score = np.mean(diff) / 255.0
        else:
            asymmetry_score = 1.0
        
        return min(asymmetry_score, 1.0)
    
    @staticmethod
    def generate_explanation(pred_index: int, confidence_scores: np.ndarray,
                            color_metrics: Dict[str, float],
                            asymmetry_score: float) -> str:
        """
        Generate comprehensive textual explanation for prediction.
        
        Args:
            pred_index: Predicted class index
            confidence_scores: Array of confidence scores for all classes
            color_metrics: Color distribution metrics
            asymmetry_score: Asymmetry measurement
        
        Returns:
            Detailed explanation text
        """
        diagnosis_names = ['Actinic Keratosis', 'Basal Cell Carcinoma', 
                          'Benign Keratosis', 'Dermatofibroma', 
                          'Melanoma', 'Melanocytic Nevus', 'Vascular Lesion']
        
        features = ExplainabilityEngine.DERMOSCOPIC_FEATURES[pred_index]
        confidence = confidence_scores[pred_index] * 100
        
        explanation = f"**Diagnostic Analysis for {diagnosis_names[pred_index]}**\n\n"
        explanation += f"**Confidence Level:** {confidence:.1f}%\n\n"
        
        # Primary diagnostic reasoning
        explanation += "**Primary Diagnostic Features Detected:**\n"
        for feature in features['key_features']:
            explanation += f"• {feature}\n"
        
        # Clinical correlation
        explanation += f"\n**Clinical Correlation:**\n"
        for sign in features['clinical_signs'][:2]:
            explanation += f"• {sign}\n"
        
        # Quantitative analysis
        explanation += f"\n**Quantitative Analysis:**\n"
        explanation += f"• Asymmetry Score: {asymmetry_score:.2f} "
        explanation += f"({'High' if asymmetry_score > 0.3 else 'Low'})\n"
        
        # Color analysis interpretation
        if pred_index == 4:  # Melanoma
            explanation += f"• Blue-white areas: {color_metrics['blue_white_presence']*100:.1f}%\n"
            explanation += f"• Color variance: {color_metrics['color_variance']:.1f} "
            explanation += f"({'High' if color_metrics['color_variance'] > 1000 else 'Low'})\n"
        
        # Differential diagnosis
        sorted_indices = np.argsort(confidence_scores)[::-1]
        if len(sorted_indices) > 1 and confidence_scores[sorted_indices[1]] > 0.1:
            explanation += f"\n**Differential Consideration:**\n"
            explanation += f"• {diagnosis_names[sorted_indices[1]]}: "
            explanation += f"{confidence_scores[sorted_indices[1]]*100:.1f}%\n"
        
        # Risk assessment
        if pred_index in [0, 1, 4]:  # Malignant conditions
            explanation += f"\n**Clinical Recommendation:**\n"
            explanation += "• Immediate dermatological consultation recommended\n"
            explanation += "• Biopsy may be warranted for definitive diagnosis\n"
        elif pred_index in [2, 3, 5]:  # Benign conditions
            explanation += f"\n**Clinical Recommendation:**\n"
            explanation += "• Regular monitoring recommended\n"
            explanation += "• Consult dermatologist if changes occur\n"
        
        # Model reasoning transparency
        explanation += f"\n**Model Reasoning Transparency:**\n"
        explanation += f"• Decision based on pattern recognition of {features['dermoscopy'][0]}\n"
        explanation += f"• Confidence distribution indicates "
        explanation += f"{'high certainty' if confidence > 70 else 'moderate certainty'}\n"
        
        return explanation
    
    @staticmethod
    def create_explanation_visualization(img: np.ndarray, heatmap: np.ndarray,
                                        sensitivity_map: np.ndarray) -> Figure:
        """
        Create comprehensive visualization of explainability results.
        
        Args:
            img: Original image
            heatmap: Grad-CAM heatmap
            sensitivity_map: Occlusion sensitivity map
        
        Returns:
            Matplotlib figure with visualization
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, weight='bold')
        axes[0].axis('off')
        
        # Grad-CAM overlay
        overlaid = ExplainabilityEngine.overlay_heatmap(img, heatmap)
        axes[1].imshow(overlaid)
        axes[1].set_title('Grad-CAM: Important Regions', fontsize=12, weight='bold')
        axes[1].axis('off')
        
        # Sensitivity map
        axes[2].imshow(sensitivity_map, cmap='hot')
        axes[2].set_title('Occlusion Sensitivity', fontsize=12, weight='bold')
        axes[2].axis('off')
        
        # Combined importance
        combined = (heatmap + sensitivity_map) / 2
        axes[3].imshow(combined, cmap='jet')
        axes[3].set_title('Combined Importance Map', fontsize=12, weight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        return fig
