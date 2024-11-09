import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict, List, Tuple, Any

class FormProcessor:
    def __init__(self, image_path: str, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize EasyOCR with better parameters
        self.reader = easyocr.Reader(['en'], gpu=False, 
                          # Disable paragraph mode for better granularity
                                   recognizer=True)  # Enable text recognition
        
        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize if image is too large
        max_dim = 1500
        height, width = self.original_image.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            self.original_image = cv2.resize(self.original_image, None, fx=scale, fy=scale)
        
        self.height, self.width = self.original_image.shape[:2]
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)


    def preprocess_for_text(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing specifically for text extraction"""
        self.save_histogram(image, "histogram_ini.png", "Histogram Initially")
        # 1. Denoise 
        denoised = cv2.fastNlMeansDenoising(image, None, 5, 7, 21) # Mean Filter (Linear Smoothing Filter)
        # denoised = self.visualize_denoising(image, h=15, template_window_size=7, search_window_size=21)
        self.save_image(denoised, "1_denoised_visual.png", "Denoised Image")
        
        # Apply Gaussian blur to reduce noise further || LOW PASS FILTERS | AVERAGING
        # SigmaX, the standard deviation in the x direction. Setting it to 0 allows OpenCV to automatically calculate the value based on the kernel size.
        blurred = cv2.GaussianBlur(denoised, (3, 3), 0) ## Gaussian Filtering (Linear Smoothing filter)
        self.save_image(blurred, "1_blurred.png", "Gaussian Blurred Image")

        # Save histogram before contrast enhancement
        self.save_histogram(blurred, "histogram_before_clahe.png", "Histogram After GaussianBlur")

        # 2. Enhance contrast 
        #CLAHE (Contrast Limited Adaptive Histogram Equalization) is used to enhance the local contrast in the image. This makes the text more distinguishable from the background, which improves OCR results.
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(5,5))
        enhanced = clahe.apply(blurred)
        self.save_image(enhanced, "2_contrast_enhanced.png", "Contrast Enhanced Image")

        # Save histogram after contrast enhancement
        self.save_histogram(enhanced, "histogram_after_clahe.png", "Histogram After CLAHE")

        # 3. Sharpen
        kernel = np.array([[-1,-1,-1], [-1,11,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        self.save_image(sharpened, "3_sharpened.png", "Sharpened Image")

        # 4. Binarize | Otsu's method automatically finds the optimal threshold value for binarization by minimizing intra-class intensity variance
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.save_image(binary, "4_binarized.png", "Binarized Image")

        return binary

    def extract_text_elements(self) -> List[Tuple[List[Tuple[int, int]], str, float]]:
        """Extract all text elements with their positions and confidence scores"""
        preprocessed = self.preprocess_for_text(self.gray_image)
        
        # Get all text elements with their bounding boxes and confidence scores
        results = self.reader.readtext(preprocessed)
        
        # Filter results by confidence
        filtered_results = [result for result in results if result[2] > 0.1]
        
        return filtered_results

    def classify_text_elements(self, elements: List[Tuple[List[Tuple[int, int]], str, float]]) -> Dict[str, str]:
        """Classify text elements into form fields based on position and content"""
        form_data = {
            'Patient ID': '',
            'Patient Name': '',
            'Doctor ID': '',
            'Doctor Name': '',
            'Prescription': '',
            'Observations': ''
        }

        # Sort elements by vertical position
        elements.sort(key=lambda x: (sum(p[1] for p in x[0]) / len(x[0])))  # Sort by average y-coordinate

        # Split image into regions
        height_third = self.height // 3
        width_mid = self.width // 2

        prescription_texts = []
        observations_texts = []

        for bbox, text, conf in elements:
            # Calculate center point of the text
            center_x = sum(p[0] for p in bbox) / len(bbox)
            center_y = sum(p[1] for p in bbox) / len(bbox)

            # Clean the text
            text = text.strip()
            if not text:
                continue

            # Classify based on position and content
            if center_y < height_third:  # Top section
                if center_x < width_mid:  # Left side
                    if text.isdigit() and not form_data['Patient ID']:
                        form_data['Patient ID'] = text
                    elif not form_data['Patient Name'] and text.isalpha():
                        form_data['Patient Name'] = text
                else:  # Right side
                    if text.isdigit() and not form_data['Doctor ID']:
                        form_data['Doctor ID'] = text
                    elif not form_data['Doctor Name'] and text.isalpha():
                        form_data['Doctor Name'] = text
            else:  # Lower sections
                if center_x < width_mid:
                    prescription_texts.append(text)
                else:
                    observations_texts.append(text)

        # Combine prescription and observations texts
        if prescription_texts:
            form_data['Prescription'] = '\n'.join(prescription_texts)
        if observations_texts:
            form_data['Observations'] = '\n'.join(observations_texts)

        return form_data

    def process_form(self) -> Dict[str, str]:
        """Process the form and extract all field values"""
        # Extract all text elements
        text_elements = self.extract_text_elements()
        
        # Classify and organize text elements
        form_data = self.classify_text_elements(text_elements)
        
        # Visualize results
        self.visualize_results(text_elements, form_data)
        
        return form_data

    

    def visualize_results(self, text_elements: List[Tuple[List[Tuple[int, int]], str, float]], 
                         form_data: Dict[str, str]):
        """Visualize detected text and classifications"""
        debug_image = self.original_image.copy()
        
        # Draw detected text elements
        for bbox, text, conf in text_elements:
            points = np.array(bbox).astype(np.int32)
            cv2.polylines(debug_image, [points], True, (0, 255, 0), 2)
            
            # Add text label
            x, y = points[0]
            cv2.putText(debug_image, text[:20], (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Text Elements')
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, 'text_detection.png'))
        plt.close()
        
        # Save results
        self.save_results(form_data)

    def save_results(self, data: Dict[str, str]):
        """Save results in multiple formats"""
        # Save as Excel
        df = pd.DataFrame([data])
        df.to_excel(os.path.join(self.output_dir, 'extracted_data.xlsx'), index=False)
        
        # Save as text file
        with open(os.path.join(self.output_dir, 'extracted_data.txt'), 'w', encoding='utf-8') as f:
            for field, value in data.items():
                f.write(f"{field}:\n{value}\n\n")

    def save_image(self, image: np.ndarray, filename: str, title: str):
        """Save an image with a title"""
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def save_histogram(self, image, filename, title):
        """Save histogram of an image."""
        plt.figure()
        plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray')
        plt.title(title)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

def proces_form(image_path: str, output_dir: str = "final_results"):
    """Process a medical form and extract all field values"""
    try:
        processor = FormProcessor(image_path, output_dir)
        results = processor.process_form()
        
        print("\nExtracted Form Data:")
        for field, value in results.items():
            print(f"\n{field}:")
            print(value)
            
        print(f"\nResults saved in: {output_dir}")
        return results
        
    except Exception as e:
        print(f"Error processing form: {str(e)}")
        return None

if __name__ == "__main__":
    proces_form("/Users/home/Desktop/PresCr/pp/handd.jpg")