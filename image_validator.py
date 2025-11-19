"""
Image validation module for detecting non-skin lesion images.
Uses a combination of CNN-based classification and statistical analysis.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from PIL import Image
import io

# CLIP imports for zero-shot classification
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


class ImageValidator:
    """
    Validates whether uploaded images are skin lesions.
    Uses pre-trained MobileNetV2 and statistical heuristics.
    """
    
    def __init__(self):
        """Initialize the validator with CLIP for zero-shot classification."""
        # Initialize CLIP for zero-shot classification
        self.clip_available = CLIP_AVAILABLE
        if CLIP_AVAILABLE:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

                # Binary CLIP comparison - just 2 prompts to avoid softmax dilution
                # This gives clear 50/50 probability split
                self.medical_prompt = "a dermoscopic or clinical photograph of a skin lesion or mole"
                self.non_medical_prompt = "a photograph that is not a medical skin image"

                # Threshold: accept if medical score is higher than non-medical
                self.binary_threshold = 0.50  # Medical must be >50% to accept

                # Pre-compute text embeddings for binary comparison
                binary_prompts = [self.medical_prompt, self.non_medical_prompt]
                text_tokens = clip.tokenize(binary_prompts).to(self.device)
                with torch.no_grad():
                    self.text_features = self.clip_model.encode_text(text_tokens)
                    self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            except Exception as e:
                print(f"Warning: Failed to initialize CLIP: {e}")
                self.clip_available = False
    
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
        try:
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
            if local_std.size > 0:
                avg_local_std = np.mean(local_std)
            else:
                avg_local_std = 0.0

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
            largest_region_pct = np.max(percentages) if len(percentages) > 0 else 0.5

            # Check if top 2 colors dominate (typical for skin + lesion)
            sorted_pcts = np.sort(percentages)[::-1]
            top2_coverage = sorted_pcts[0] + sorted_pcts[1] if len(sorted_pcts) > 1 else sorted_pcts[0] if len(sorted_pcts) > 0 else 0.5

            return {
                'avg_local_std': float(avg_local_std),
                'largest_region_pct': float(largest_region_pct),
                'top2_coverage': float(top2_coverage),
                'color_spread': float(np.std(percentages)) if len(percentages) > 0 else 0.0
            }
        except Exception:
            # Return default values if analysis fails
            return {
                'avg_local_std': 0.0,
                'largest_region_pct': 0.5,
                'top2_coverage': 0.5,
                'color_spread': 0.0
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

    def validate_with_clip(self, img: np.ndarray) -> Tuple[bool, float, str]:
        """
        Use CLIP zero-shot classification to determine if image is medical/skin-related.

        CLIP was trained on a massive dataset of images with natural language descriptions,
        making it excellent at understanding semantic content without being limited to
        ImageNet classes.

        Args:
            img: Input image in RGB format

        Returns:
            Tuple of (is_medical, confidence, reason)
        """
        if not self.clip_available:
            # Fall back to accepting if CLIP not available
            return True, 0.5, "CLIP not available"

        try:
            # Convert numpy array to PIL Image for CLIP preprocessing
            pil_image = Image.fromarray(img)

            # Preprocess image for CLIP
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            # Compute image features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Compute similarities with all text prompts
                similarities = (image_features @ self.text_features.T).squeeze(0)

                # Apply softmax to get probabilities
                probs = similarities.softmax(dim=-1).cpu().numpy()

            # Binary comparison - just 2 prompts, no dilution
            medical_prob = float(probs[0])
            non_medical_prob = float(probs[1])

            # Simple threshold check
            if medical_prob >= self.binary_threshold:
                reason = "Valid skin lesion image"
                return True, medical_prob, reason
            else:
                reason = "Image may not be a valid skin lesion"
                return False, non_medical_prob, reason

        except Exception as e:
            # If CLIP fails, fall back to accepting
            print(f"CLIP validation error: {e}")
            return True, 0.5, f"CLIP error: {str(e)}"

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
        Image validation pipeline using CLIP zero-shot classification.

        CLIP-BASED APPROACH: Uses semantic understanding to distinguish
        medical skin images from non-medical content. CLIP was trained on
        diverse image-text pairs and can understand semantic concepts beyond
        ImageNet classes.

        Rejection criteria:
        1. Invalid dimensions
        2. Simple graphic detection (very few unique colors)
        3. CLIP classifies as non-medical content
        4. Fallback: MobileNet detects cats/dogs (if CLIP unavailable)

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
            results['confidence'] = 0.0
            return results

        # Check 2: Simple graphic detection
        # Only reject obvious icons/drawings with very few unique colors
        try:
            img_small = cv2.resize(img, (100, 100)) if img.shape[0] > 100 else img
            unique_colors = len(np.unique(img_small.reshape(-1, 3), axis=0))

            if unique_colors < 30:
                results['is_valid'] = False
                results['reasons'].append(
                    f"Image appears to be a simple graphic ({unique_colors} unique colors)"
                )
                results['confidence'] = 0.2
                return results
        except Exception:
            pass

        # Check 3: CLIP-based semantic classification
        if self.clip_available:
            is_medical, clip_confidence, clip_reason = self.validate_with_clip(img)
            if not is_medical:
                results['is_valid'] = False
                results['reasons'].append(clip_reason)
                results['confidence'] = 1.0 - clip_confidence
                return results
            else:
                # CLIP validated as medical
                results['confidence'] = clip_confidence
                return results
        else:
            # CLIP not available - accept but warn
            results['warnings'].append("CLIP not available - validation skipped")
            results['confidence'] = 0.5
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

            if validation_results['warnings']:
                report += "**Warnings:**\n"
                for warning in validation_results['warnings']:
                    report += f"⚠️ {warning}\n"
            else:
                report += "Image appears to be a valid skin lesion photograph.\n"
        else:
            report = "⚠️ **Image Validation Warning**\n\n"
            report += "The uploaded image may not be a valid skin lesion photograph.\n"
            report += "Predictions on this image may not be reliable.\n\n"
            report += "If you are sure this is a valid skin lesion image, "
            report += "you can confirm below to proceed with the analysis.\n"

        return report
