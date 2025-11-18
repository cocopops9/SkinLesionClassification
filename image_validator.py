"""
Image validation module for detecting non-skin lesion images.
Uses a combination of CNN-based classification and statistical analysis.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from typing import Tuple, Dict, Optional
from PIL import Image
import io


class ImageValidator:
    """
    Validates whether uploaded images are skin lesions.
    Uses pre-trained MobileNetV2 and statistical heuristics.
    """
    
    def __init__(self):
        """Initialize the validator with pre-trained MobileNetV2."""
        self.mobilenet = MobileNetV2(weights='imagenet', include_top=True)
        
        # ImageNet classes related to skin/medical imagery
        self.medical_classes = {
            919: 'band_aid',
            445: 'syringe',
            917: 'washbasin',
            639: 'lab_coat'
        }
        
        # Classes that definitely indicate non-skin images
        self.non_skin_classes = {
            # Animals
            281: 'tabby_cat', 285: 'Egyptian_cat', 207: 'golden_retriever',
            # Objects
            504: 'coffee_mug', 546: 'electric_fan', 664: 'monitor',
            # Vehicles
            817: 'sports_car', 511: 'convertible', 656: 'minivan',
            # Food
            934: 'hotdog', 926: 'hot_pot', 927: 'trifle',
            # Landscapes
            978: 'seashore', 979: 'lakeside', 973: 'cliff'
        }
        
        # Initialize skin detection color ranges (in HSV)
        self.skin_hsv_lower = np.array([0, 20, 20], dtype=np.uint8)
        self.skin_hsv_upper = np.array([20, 255, 255], dtype=np.uint8)
        
        # Alternative skin tone range
        self.skin_hsv_lower2 = np.array([165, 20, 20], dtype=np.uint8)
        self.skin_hsv_upper2 = np.array([180, 255, 255], dtype=np.uint8)
    
    def validate_image_dimensions(self, img: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image dimensions and aspect ratio.
        
        Args:
            img: Input image array
        
        Returns:
            Tuple of (is_valid, message)
        """
        height, width = img.shape[:2]
        
        # Check minimum dimensions
        if height < 100 or width < 100:
            return False, "Image too small (minimum 100x100 pixels required)"
        
        # Check maximum dimensions
        if height > 10000 or width > 10000:
            return False, "Image too large (maximum 10000x10000 pixels)"
        
        # Check aspect ratio
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 5:
            return False, f"Unusual aspect ratio ({aspect_ratio:.1f}:1), likely not a medical image"
        
        return True, "Dimensions valid"
    
    def detect_skin_presence(self, img: np.ndarray) -> float:
        """
        Detect skin-like pixels using color-based segmentation.
        
        Args:
            img: Input image in RGB format
        
        Returns:
            Percentage of skin-like pixels (0-1)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Create masks for both skin tone ranges
        mask1 = cv2.inRange(hsv, self.skin_hsv_lower, self.skin_hsv_upper)
        mask2 = cv2.inRange(hsv, self.skin_hsv_lower2, self.skin_hsv_upper2)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate percentage of skin pixels
        skin_pixels = np.sum(skin_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        
        return skin_pixels / total_pixels
    
    def analyze_color_statistics(self, img: np.ndarray) -> Dict[str, float]:
        """
        Analyze color statistics to identify non-medical images.
        
        Args:
            img: Input image in RGB format
        
        Returns:
            Dictionary of color statistics
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        stats = {
            'mean_saturation': np.mean(hsv[:, :, 1]),
            'std_saturation': np.std(hsv[:, :, 1]),
            'mean_value': np.mean(hsv[:, :, 2]),
            'unique_colors': len(np.unique(img.reshape(-1, img.shape[-1]), axis=0)),
            'edge_density': self._compute_edge_density(img),
            'color_entropy': self._compute_color_entropy(img)
        }
        
        return stats
    
    def _compute_edge_density(self, img: np.ndarray) -> float:
        """
        Compute edge density using Canny edge detection.
        
        Args:
            img: Input image
        
        Returns:
            Edge density ratio (0-1)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    
    def _compute_color_entropy(self, img: np.ndarray) -> float:
        """
        Compute color entropy as measure of color distribution.

        Args:
            img: Input image

        Returns:
            Entropy value
        """
        # Quantize colors to reduce computational complexity
        img_quant = (img // 32) * 32

        # Compute color histogram
        unique, counts = np.unique(img_quant.reshape(-1, 3), axis=0, return_counts=True)
        probabilities = counts / counts.sum()

        # Compute entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        return entropy

    def detect_uniform_regions(self, img: np.ndarray) -> Dict[str, float]:
        """
        Detect presence of uniform color regions (characteristic of skin lesion images).
        Skin lesion images typically have large areas of uniform skin color.

        Args:
            img: Input image in RGB format

        Returns:
            Dictionary with uniform region metrics
        """
        # Convert to LAB for better color similarity detection
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(lab, 9, 75, 75)

        # Compute local color variance using a sliding window approach
        # Resize for efficiency
        scale = min(200 / img.shape[0], 200 / img.shape[1], 1.0)
        if scale < 1.0:
            small = cv2.resize(filtered, None, fx=scale, fy=scale)
        else:
            small = filtered

        # Calculate local standard deviation
        kernel_size = 15
        local_mean = cv2.blur(small.astype(np.float32), (kernel_size, kernel_size))
        local_sqr_mean = cv2.blur((small.astype(np.float32) ** 2), (kernel_size, kernel_size))
        local_variance = local_sqr_mean - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_variance, 0))

        # Average local standard deviation across all LAB channels
        avg_local_std = np.mean(local_std)

        # Use color quantization to find dominant colors
        pixels = img.reshape(-1, 3).astype(np.float32)

        # K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 5  # Number of dominant colors to find

        # Sample pixels for efficiency
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            sample_pixels = pixels[indices]
        else:
            sample_pixels = pixels

        _, labels, centers = cv2.kmeans(sample_pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        # Calculate percentage of pixels in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)

        # Largest uniform region percentage
        largest_region_pct = np.max(percentages)

        # Check if top 2 colors dominate (typical for skin + lesion)
        sorted_pcts = np.sort(percentages)[::-1]
        top2_coverage = sorted_pcts[0] + sorted_pcts[1] if len(sorted_pcts) > 1 else sorted_pcts[0]

        return {
            'avg_local_std': avg_local_std,
            'largest_region_pct': largest_region_pct,
            'top2_coverage': top2_coverage,
            'color_spread': np.std(percentages)
        }

    def analyze_texture_uniformity(self, img: np.ndarray) -> Dict[str, float]:
        """
        Analyze texture uniformity to detect non-skin images.
        Natural skin has specific texture patterns different from random images.

        Args:
            img: Input image in RGB format

        Returns:
            Dictionary with texture metrics
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Compute Laplacian variance (measure of focus/texture)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()

        # Compute gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)

        # Statistics of gradient
        gradient_mean = np.mean(gradient_mag)
        gradient_std = np.std(gradient_mag)

        # Coefficient of variation (normalized measure of texture variation)
        gradient_cv = gradient_std / (gradient_mean + 1e-10)

        # Check for periodic patterns (common in non-natural images)
        # Use FFT to detect strong frequency components
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Ratio of high-frequency to low-frequency content
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2

        # Low frequency region (center)
        low_freq_region = magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10]
        low_freq_energy = np.sum(low_freq_region ** 2)

        # High frequency region (outer)
        high_freq_energy = np.sum(magnitude_spectrum ** 2) - low_freq_energy

        freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)

        return {
            'laplacian_var': laplacian_var,
            'gradient_mean': gradient_mean,
            'gradient_cv': gradient_cv,
            'freq_ratio': freq_ratio
        }
    
    def classify_with_mobilenet(self, img: np.ndarray) -> Tuple[bool, float, str]:
        """
        Use MobileNetV2 to detect non-skin images.
        
        Args:
            img: Input image in RGB format
        
        Returns:
            Tuple of (is_likely_skin, confidence, detected_class)
        """
        # Preprocess for MobileNetV2
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Get predictions
        predictions = self.mobilenet.predict(img_array, verbose=0)
        top_indices = np.argsort(predictions[0])[::-1][:5]
        
        # Check if any top prediction is clearly non-medical
        for idx in top_indices:
            confidence = predictions[0][idx]
            if confidence > 0.3:  # Confidence threshold
                if idx in self.non_skin_classes:
                    return False, confidence, self.non_skin_classes[idx]
        
        # If no clear non-medical class, assume potentially valid
        return True, 0.0, "unknown"
    
    def detect_text_presence(self, img: np.ndarray) -> bool:
        """
        Detect if image contains significant text (screenshots, documents).
        
        Args:
            img: Input image
        
        Returns:
            Boolean indicating significant text presence
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Detect horizontal and vertical lines (common in documents)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Check for grid-like structure
        total_lines = np.sum(horizontal_lines > 0) + np.sum(vertical_lines > 0)
        total_pixels = img.shape[0] * img.shape[1]
        
        line_ratio = total_lines / total_pixels
        
        return line_ratio > 0.05  # Threshold for document detection
    
    def validate_image(self, img_input) -> Dict[str, any]:
        """
        Comprehensive image validation pipeline.
        
        Args:
            img_input: PIL Image or numpy array
        
        Returns:
            Dictionary with validation results
        """
        # Convert to numpy array if needed
        if isinstance(img_input, Image.Image):
            img = np.array(img_input.convert('RGB'))
        else:
            img = img_input
        
        results = {
            'is_valid': True,
            'confidence': 1.0,
            'reasons': [],
            'warnings': []
        }
        
        # Check 1: Validate dimensions
        dim_valid, dim_msg = self.validate_image_dimensions(img)
        if not dim_valid:
            results['is_valid'] = False
            results['reasons'].append(dim_msg)
            results['confidence'] *= 0.1
            return results
        
        # Check 2: Detect text/document
        if self.detect_text_presence(img):
            results['is_valid'] = False
            results['reasons'].append("Image appears to be a document or screenshot")
            results['confidence'] *= 0.2
            return results
        
        # Check 3: MobileNet classification
        is_skin, mobilenet_conf, detected_class = self.classify_with_mobilenet(img)
        if not is_skin:
            results['is_valid'] = False
            results['reasons'].append(f"Image detected as '{detected_class}' with {mobilenet_conf*100:.1f}% confidence")
            results['confidence'] *= (1 - mobilenet_conf)
            return results
        
        # Check 4: Skin presence detection
        skin_ratio = self.detect_skin_presence(img)
        if skin_ratio < 0.01:  # Less than 1% skin-like pixels - very lenient
            results['warnings'].append(f"Very low skin presence detected ({skin_ratio*100:.1f}%)")
            results['confidence'] *= 0.7
        elif skin_ratio < 0.1:
            results['warnings'].append(f"Low skin presence detected ({skin_ratio*100:.1f}%)")
            results['confidence'] *= 0.9
        
        # Check 5: Color statistics
        color_stats = self.analyze_color_statistics(img)

        # Check for unnatural images (cartoons, drawings)
        if color_stats['unique_colors'] < 100:
            results['is_valid'] = False
            results['reasons'].append("Image appears to be a drawing or cartoon (too few unique colors)")
            results['confidence'] *= 0.2
            return results

        # Check for overly saturated images (likely non-medical)
        if color_stats['mean_saturation'] > 200:
            results['warnings'].append("Image has unusually high saturation")
            results['confidence'] *= 0.8

        # Check edge density (text or technical drawings have high edge density)
        if color_stats['edge_density'] > 0.3:
            results['warnings'].append("High edge density detected")
            results['confidence'] *= 0.85

        # Check 6: Uniform color regions (informational only - don't reject)
        uniform_stats = self.detect_uniform_regions(img)

        # Just warn on fragmented images, don't reject
        if uniform_stats['largest_region_pct'] < 0.08:
            results['warnings'].append(
                f"Low uniform color areas ({uniform_stats['largest_region_pct']*100:.1f}% largest region)"
            )
            results['confidence'] *= 0.9

        # Just warn on busy/colorful images, don't reject
        if uniform_stats['top2_coverage'] < 0.25:
            results['warnings'].append(
                f"High color diversity ({uniform_stats['top2_coverage']*100:.1f}% top-2 color coverage)"
            )
            results['confidence'] *= 0.9

        # Check 7: Texture uniformity analysis (informational only)
        texture_stats = self.analyze_texture_uniformity(img)

        # Just warn on unusual patterns, don't reject
        if texture_stats['freq_ratio'] > 150:
            results['warnings'].append("Unusual texture patterns detected")
            results['confidence'] *= 0.95

        # Final validation - very lenient threshold
        if results['confidence'] < 0.3:
            results['is_valid'] = False
            if not results['reasons']:
                results['reasons'].append("Image characteristics inconsistent with skin lesion imagery")

        return results
    
    def generate_validation_report(self, validation_results: Dict[str, any]) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            validation_results: Results from validate_image()
        
        Returns:
            Formatted report string
        """
        if validation_results['is_valid']:
            report = "✅ **Image Validation: PASSED**\n\n"
            report += f"Confidence: {validation_results['confidence']*100:.1f}%\n"
            
            if validation_results['warnings']:
                report += "\n**Warnings:**\n"
                for warning in validation_results['warnings']:
                    report += f"⚠️ {warning}\n"
            else:
                report += "\nImage appears to be a valid skin lesion photograph.\n"
        else:
            report = "❌ **Image Validation: FAILED**\n\n"
            report += "**Invalid Image Detected:**\n"
            for reason in validation_results['reasons']:
                report += f"• {reason}\n"
            
            report += "\n**Recommendation:**\n"
            report += "Please upload a clear photograph or dermoscopic image of a skin lesion.\n"
            report += "Ensure the image:\n"
            report += "• Shows skin surface clearly\n"
            report += "• Is well-lit and in focus\n"
            report += "• Contains no text or overlays\n"
            report += "• Is a medical/clinical photograph\n"
        
        return report
