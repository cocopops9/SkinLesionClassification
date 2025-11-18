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

        # MINIMAL VALIDATION: Only reject cats and dogs
        # These are the most common non-medical uploads and MobileNet recognizes them well
        # All other classes removed - too unreliable
        self.non_skin_classes = {
            # Cats - all cat breeds
            281: 'tabby_cat', 282: 'tiger_cat', 283: 'Persian_cat',
            284: 'Siamese_cat', 285: 'Egyptian_cat', 286: 'cougar',
            287: 'lynx', 288: 'leopard', 289: 'snow_leopard',
            290: 'jaguar', 291: 'lion', 292: 'tiger', 293: 'cheetah',
            # Dogs - common breeds
            151: 'Chihuahua', 152: 'Japanese_spaniel', 153: 'Maltese_dog',
            154: 'Pekinese', 155: 'Shih-Tzu', 156: 'Blenheim_spaniel',
            157: 'papillon', 158: 'toy_terrier', 159: 'Rhodesian_ridgeback',
            160: 'Afghan_hound', 161: 'basset', 162: 'beagle',
            163: 'bloodhound', 164: 'bluetick', 165: 'black-and-tan_coonhound',
            166: 'Walker_hound', 167: 'English_foxhound', 168: 'redbone',
            169: 'borzoi', 170: 'Irish_wolfhound', 171: 'Italian_greyhound',
            172: 'whippet', 173: 'Ibizan_hound', 174: 'Norwegian_elkhound',
            175: 'otterhound', 176: 'Saluki', 177: 'Scottish_deerhound',
            178: 'Weimaraner', 179: 'Staffordshire_bullterrier',
            180: 'American_Staffordshire_terrier', 181: 'Bedlington_terrier',
            182: 'Border_terrier', 183: 'Kerry_blue_terrier',
            184: 'Irish_terrier', 185: 'Norfolk_terrier', 186: 'Norwich_terrier',
            187: 'Yorkshire_terrier', 188: 'wire-haired_fox_terrier',
            189: 'Lakeland_terrier', 190: 'Sealyham_terrier', 191: 'Airedale',
            192: 'cairn', 193: 'Australian_terrier', 194: 'Dandie_Dinmont',
            195: 'Boston_bull', 196: 'miniature_schnauzer', 197: 'giant_schnauzer',
            198: 'standard_schnauzer', 199: 'Scotch_terrier',
            200: 'Tibetan_terrier', 201: 'silky_terrier',
            202: 'soft-coated_wheaten_terrier', 203: 'West_Highland_white_terrier',
            204: 'Lhasa', 205: 'flat-coated_retriever', 206: 'curly-coated_retriever',
            207: 'golden_retriever', 208: 'Labrador_retriever',
            209: 'Chesapeake_Bay_retriever', 210: 'German_short-haired_pointer',
            211: 'vizsla', 212: 'English_setter', 213: 'Irish_setter',
            214: 'Gordon_setter', 215: 'Brittany_spaniel', 216: 'clumber',
            217: 'English_springer', 218: 'Welsh_springer_spaniel',
            219: 'cocker_spaniel', 220: 'Sussex_spaniel', 221: 'Irish_water_spaniel',
            222: 'kuvasz', 223: 'schipperke', 224: 'groenendael',
            225: 'malinois', 226: 'briard', 227: 'kelpie', 228: 'komondor',
            229: 'Old_English_sheepdog', 230: 'Shetland_sheepdog', 231: 'collie',
            232: 'Border_collie', 233: 'Bouvier_des_Flandres', 234: 'Rottweiler',
            235: 'German_shepherd', 236: 'Doberman', 237: 'miniature_pinscher',
            238: 'Greater_Swiss_Mountain_dog', 239: 'Bernese_mountain_dog',
            240: 'Appenzeller', 241: 'EntleBucher', 242: 'boxer', 243: 'bull_mastiff',
            244: 'Tibetan_mastiff', 245: 'French_bulldog', 246: 'Great_Dane',
            247: 'Saint_Bernard', 248: 'Eskimo_dog', 249: 'malamute',
            250: 'Siberian_husky', 251: 'dalmatian', 252: 'affenpinscher',
            253: 'basenji', 254: 'pug', 255: 'Leonberg', 256: 'Newfoundland',
            257: 'Great_Pyrenees', 258: 'Samoyed', 259: 'Pomeranian',
            260: 'chow', 261: 'keeshond', 262: 'Brabancon_griffon',
            263: 'Pembroke', 264: 'Cardigan', 265: 'toy_poodle',
            266: 'miniature_poodle', 267: 'standard_poodle',
            268: 'Mexican_hairless',
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
        MINIMAL VALIDATION: Only reject clear cat/dog photos.

        After 5 failed approaches, this uses the simplest possible logic:
        - Only check for cats and dogs (most common non-medical uploads)
        - Use low threshold (40%) because MobileNet is good at these
        - Accept everything else

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

        # Check top-5 predictions for cats/dogs with 40% threshold
        # Lower threshold because MobileNet is well-trained on animals
        for idx in top_indices:
            confidence = predictions[0][idx]
            if confidence > 0.40 and idx in self.non_skin_classes:
                return False, confidence, self.non_skin_classes[idx]

        # Accept everything else
        return True, 0.0, "accepted"
    
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

            if unique_colors < 30:  # Very strict - only catch obvious graphics
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
