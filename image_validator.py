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

        # Expanded list of classes that definitely indicate non-skin images
        # Only reject when MobileNet is very confident (>80%)
        self.non_skin_classes = {
            # Animals - cats
            281: 'tabby_cat', 282: 'tiger_cat', 283: 'Persian_cat',
            284: 'Siamese_cat', 285: 'Egyptian_cat', 286: 'cougar',
            # Animals - dogs
            151: 'Chihuahua', 207: 'golden_retriever', 208: 'Labrador_retriever',
            209: 'Chesapeake_Bay_retriever', 210: 'German_short-haired_pointer',
            229: 'Old_English_sheepdog', 232: 'Border_collie', 235: 'German_shepherd',
            # Animals - other
            30: 'bullfrog', 31: 'tree_frog', 33: 'loggerhead_turtle',
            # Vehicles
            407: 'ambulance', 436: 'beach_wagon', 468: 'cab',
            511: 'convertible', 555: 'fire_engine', 609: 'jeep',
            656: 'minivan', 675: 'motor_scooter', 717: 'pickup',
            751: 'racer', 817: 'sports_car', 864: 'tow_truck',
            # Food
            924: 'guacamole', 925: 'consomme', 926: 'hot_pot',
            927: 'trifle', 928: 'ice_cream', 929: 'ice_lolly',
            930: 'French_loaf', 931: 'bagel', 932: 'pretzel',
            933: 'cheeseburger', 934: 'hotdog', 935: 'mashed_potato',
            936: 'head_cabbage', 937: 'broccoli', 938: 'cauliflower',
            939: 'zucchini', 940: 'spaghetti_squash', 941: 'acorn_squash',
            942: 'butternut_squash', 943: 'cucumber', 944: 'artichoke',
            945: 'bell_pepper', 946: 'cardoon', 947: 'mushroom',
            948: 'Granny_Smith', 949: 'strawberry', 950: 'orange',
            951: 'lemon', 952: 'fig', 953: 'pineapple', 954: 'banana',
            955: 'jackfruit', 956: 'custard_apple', 957: 'pomegranate',
            # Household objects
            504: 'coffee_mug', 505: 'coffeepot', 532: 'dining_table',
            546: 'electric_fan', 553: 'file_cabinet', 620: 'laptop',
            664: 'monitor', 671: 'mouse', 703: 'park_bench',
            720: 'pill_bottle', 737: 'printer', 742: 'racket',
            765: 'rocking_chair', 831: 'studio_couch', 832: 'stupa',
            # Landscapes/scenes
            970: 'alp', 971: 'bubble', 972: 'cliff', 973: 'coral_reef',
            974: 'geyser', 975: 'lakeside', 976: 'promontory',
            977: 'sandbar', 978: 'seashore', 979: 'valley',
            # Buildings
            497: 'church', 498: 'cinema', 536: 'dock', 663: 'monastery',
            698: 'palace', 833: 'submarine', 900: 'water_tower',
        }

        # Remove skin detection - not reliable for medical images
        # HSV ranges removed as they don't work for dermoscopic images
    
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
    
    def classify_with_mobilenet(self, img: np.ndarray) -> Tuple[bool, float, str]:
        """
        Use MobileNetV2 to detect non-skin images using entropy-based analysis.

        NEW APPROACH: Use prediction confidence distribution, not just top prediction.
        - High confidence on blacklisted class (>50%) → REJECT
        - Very high confidence on ANY class (>85%) → REJECT (clear object, not medical)
        - All top-5 predictions low (<20%) → ACCEPT (MobileNet confused = unusual/medical)

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
        top_indices = np.argsort(predictions[0])[::-1][:10]
        top_confidences = [predictions[0][idx] for idx in top_indices[:5]]

        # Get top prediction info
        top_idx = top_indices[0]
        top_conf = predictions[0][top_idx]

        # Check 1: If top prediction is a blacklisted class with moderate confidence
        # Lower threshold (50%) for specific non-medical objects
        if top_conf > 0.50 and top_idx in self.non_skin_classes:
            return False, top_conf, self.non_skin_classes[top_idx]

        # Check 2: If very confident about ANY class, it's a clear object
        # Medical images shouldn't strongly match any ImageNet class
        if top_conf > 0.85:
            # Get class name from ImageNet labels
            class_name = f"class_{top_idx}"
            # Check if it's in our blacklist
            if top_idx in self.non_skin_classes:
                return False, top_conf, self.non_skin_classes[top_idx]
            # Even if not blacklisted, very high confidence means clear object
            # Only reject if it's clearly not medical-related
            # (medical classes like band-aid, syringe are allowed)
            if top_idx not in self.medical_classes:
                return False, top_conf, f"clear_object_{top_idx}"

        # Check 3: Entropy-based acceptance
        # If MobileNet is very confused (all top predictions low), accept
        # This indicates unusual content like medical images
        max_top5 = max(top_confidences)
        if max_top5 < 0.25:
            # Very confused - likely unusual/medical image
            return True, 0.0, "unknown_medical"

        # Check 4: Check if any of top-10 predictions hit blacklist with decent confidence
        for idx in top_indices:
            confidence = predictions[0][idx]
            if confidence > 0.35 and idx in self.non_skin_classes:
                return False, confidence, self.non_skin_classes[idx]

        # If no clear non-medical class detected, accept
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
        Simplified image validation pipeline.

        NEW APPROACH: Only reject images when we are very confident they are
        not medical images. This avoids false rejections of valid skin lesions.

        Rejection criteria:
        1. Invalid dimensions
        2. MobileNet detects non-medical object with >80% confidence
        3. Image is a simple graphic (<50 unique colors)

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

        # Check 2: MobileNet classification
        # Only reject if very confident (>80%) it's a non-medical object
        is_valid_type, mobilenet_conf, detected_class = self.classify_with_mobilenet(img)
        if not is_valid_type:
            results['is_valid'] = False
            results['reasons'].append(
                f"Image detected as '{detected_class}' with {mobilenet_conf*100:.1f}% confidence"
            )
            results['confidence'] = 1.0 - mobilenet_conf
            return results

        # Check 3: Simple graphic detection
        # Only reject obvious icons/drawings with very few unique colors
        try:
            # Quantize and count unique colors
            img_small = cv2.resize(img, (100, 100)) if img.shape[0] > 100 else img
            unique_colors = len(np.unique(img_small.reshape(-1, 3), axis=0))

            if unique_colors < 50:
                results['is_valid'] = False
                results['reasons'].append(
                    f"Image appears to be a simple graphic ({unique_colors} unique colors)"
                )
                results['confidence'] = 0.2
                return results
        except Exception:
            # If color analysis fails, don't reject
            pass

        # Image passed all checks
        results['confidence'] = 1.0
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
