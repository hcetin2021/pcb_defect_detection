"""
PCB Defect Detection System with Adaptive OTSU Thresholding
F1 Score: 77.78%

Performance Comparison:
- Fixed Parameters (threshold=25, kernel=3, area=60): F1 Score 52.31%
- OTSU Adaptive Method: F1 Score 77.78% (+25.47% improvement)

Attempted Methods (unsuccessful):
- SSIM + Edge Detection: Excessive false positives
- Multi-scale Detection: 23,871 FP (catastrophic)
- CLAHE Preprocessing: Increased false positive rate
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import linear_sum_assignment
import gradio as gr
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

BASE_PATH = Path(__file__).parent

# ============== DEFECT TYPES ==============

DEFECT_INFO = {
    'missing_hole': {'name': 'Missing Hole', 'desc': 'Missing drill hole on PCB'},
    'mouse_bite': {'name': 'Mouse Bite', 'desc': 'Irregular notches at PCB edge'},
    'open_circuit': {'name': 'Open Circuit', 'desc': 'Broken conductor path'},
    'short': {'name': 'Short Circuit', 'desc': 'Unwanted connection'},
    'spur': {'name': 'Spur', 'desc': 'Excess copper extension'},
    'spurious_copper': {'name': 'Spurious Copper', 'desc': 'Unwanted copper residue'}
}

# ============== CORE FUNCTIONS ==============

def get_actual_defects(xml_file: str) -> List[Tuple[int, int, int, int]]:
    """Extract ground truth coordinates from XML annotation file"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        defects = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            defects.append((xmin, ymin, xmax, ymax))
        return defects
    except:
        return []

def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate Intersection over Union (IoU)"""
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    union = (x2-x1+1)*(y2-y1+1) + (x2p-x1p+1)*(y2p-y1p+1) - inter
    return inter / union if union > 0 else 0

def calculate_metrics(detected: List, actual: List, iou_thresh: float = 0.1) -> Dict:
    """Calculate detection metrics (TP, FP, FN, precision, recall, F1)"""
    n_det, n_act = len(detected), len(actual)
    
    if n_det == 0 and n_act == 0:
        return {
            'tp': 0, 'fp': 0, 'fn': 0,
            'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
            'matched_det': [], 'matched_gt': [], 'ious': []
        }
    
    if n_det == 0:
        return {
            'tp': 0, 'fp': 0, 'fn': n_act,
            'precision': 0, 'recall': 0, 'f1': 0,
            'matched_det': [], 'matched_gt': [], 'ious': []
        }
    
    if n_act == 0:
        return {
            'tp': 0, 'fp': n_det, 'fn': 0,
            'precision': 0, 'recall': 0, 'f1': 0,
            'matched_det': [], 'matched_gt': [], 'ious': []
        }
    
    # IoU matrix
    cost = np.array([[-calculate_iou(d, a) for a in actual] for d in detected])
    row, col = linear_sum_assignment(cost)
    
    matched_det, matched_gt, ious = [], [], []
    for i, j in zip(row, col):
        iou = -cost[i, j]
        if iou > iou_thresh:
            matched_det.append(i)
            matched_gt.append(j)
            ious.append(iou)
    
    tp = len(matched_det)
    fp = n_det - tp
    fn = n_act - tp
    
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': prec, 'recall': rec, 'f1': f1,
        'matched_det': matched_det, 'matched_gt': matched_gt, 'ious': ious
    }

# ============== IMAGE PROCESSING ==============

def align_images(ref: np.ndarray, test: np.ndarray) -> Optional[np.ndarray]:
    """Align images using SIFT feature matching and homography"""
    sift = cv2.SIFT_create()
    kp1, d1 = sift.detectAndCompute(ref, None)
    kp2, d2 = sift.detectAndCompute(test, None)
    
    if d1 is None or d2 is None:
        return None
    
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    if len(good) < 4:
        return None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    
    return cv2.warpPerspective(test, H, (ref.shape[1], ref.shape[0]))

def detect_defects(ref: np.ndarray, test: np.ndarray) -> Tuple[List, np.ndarray, Dict]:
    """
    Adaptive OTSU defect detection
    
    Returns:
        detected: List of detected bounding boxes
        diff: Difference image
        params: Detection parameters used
    """
    aligned = align_images(ref, test)
    if aligned is None:
        return [], np.zeros_like(ref), {}
    
    diff = cv2.absdiff(ref, aligned)
    
    # OTSU adaptive threshold
    otsu_thresh, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive kernel (görüntü boyutuna göre)
    k = max(3, int((ref.shape[0] + ref.shape[1]) / 800))
    k = k if k % 2 == 1 else k + 1
    kernel = np.ones((k, k), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Adaptive min area
    min_area = int(ref.shape[0] * ref.shape[1] * 0.00002)
    min_area = max(30, min(min_area, 200))
    
    # Contour detection
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            detected.append((x, y, x+w, y+h))
    
    params = {
        'otsu_threshold': int(otsu_thresh),
        'kernel_size': k,
        'min_area': min_area,
        'image_size': f"{ref.shape[1]}×{ref.shape[0]}"
    }
    
    return detected, diff, params

# ============== VISUALIZATION ==============

def draw_results(img: np.ndarray, detected: List, actual: List, metrics: Dict) -> np.ndarray:
    """Draw detection results with bounding boxes and labels"""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    
    matched_det = metrics.get('matched_det', [])
    matched_gt = metrics.get('matched_gt', [])
    
    # Ground Truth (Yeşil kesikli)
    for j, (x, y, x2, y2) in enumerate(actual):
        color = (0, 255, 0) if j in matched_gt else (0, 180, 0)
        thickness = 2 if j in matched_gt else 1
        # Dashed line effect
        for px in range(x, x2, 8):
            cv2.line(out, (px, y), (min(px+4, x2), y), color, thickness)
            cv2.line(out, (px, y2), (min(px+4, x2), y2), color, thickness)
        for py in range(y, y2, 8):
            cv2.line(out, (x, py), (x, min(py+4, y2)), color, thickness)
            cv2.line(out, (x2, py), (x2, min(py+4, y2)), color, thickness)
    
    # Detected boxes (solid lines)
    for i, (x, y, x2, y2) in enumerate(detected):
        if i in matched_det:
            color = (0, 255, 0)  # True positive - Green
            label = "TP"
        else:
            color = (0, 0, 255)  # False positive - Red
            label = "FP"
        
        cv2.rectangle(out, (x, y), (x2, y2), color, 2)
        
        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y), color, -1)
        cv2.putText(out, label, (x + 3, y - 3), font, 0.5, (255, 255, 255), 1)
    
    # Mark missed defects (FN)
    for j, (x, y, x2, y2) in enumerate(actual):
        if j not in matched_gt:
            cx, cy = (x + x2) // 2, (y + y2) // 2
            cv2.drawMarker(out, (cx, cy), (0, 165, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(out, "FN", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    return out

# ============== FILE HELPERS ==============

def find_annotation(test_path: str) -> Optional[str]:
    """Find annotation file for test image"""
    if test_path is None:
        return None
    
    filename = Path(test_path).stem
    
    # Search in annotation folders
    for defect_type in ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']:
        xml_path = BASE_PATH / "Annotations" / defect_type / f"{filename}.xml"
        if xml_path.exists():
            return str(xml_path)
    
    return None

def get_defect_type(filename: str) -> Tuple[str, str]:
    """Extract defect type information from filename"""
    filename_lower = filename.lower()
    for key, info in DEFECT_INFO.items():
        if key in filename_lower:
            return info['name'], info['desc']
    return "Unknown", "Unknown defect type"

def find_reference() -> Optional[str]:
    """Find reference image"""
    for ext in ['.JPG', '.jpg', '.jpeg', '.png']:
        ref_path = BASE_PATH / "Reference" / f"01{ext}"
        if ref_path.exists():
            return str(ref_path)
    return None

# ============== MAIN DETECTION ==============

def run_detection(test_file):
    """Main detection function - load file and run automatic detection"""
    
    if test_file is None:
        return None, None, "**Error: Please upload an image**"
    
    # Read file
    test_img = cv2.imread(test_file.name)
    if test_img is None:
        return None, None, "**Error: Failed to read image**"
    
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    filename = Path(test_file.name).stem
    
    # Reference image (automatic)
    ref_path = find_reference()
    if ref_path is None:
        return None, None, "**Error: Reference image not found** (Reference/01.JPG)"
    
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        return None, None, "**Error: Failed to load reference image**"
    
    # Detection
    detected, diff, params = detect_defects(ref_img, test_gray)
    
    if len(detected) == 0 and diff.sum() == 0:
        return None, None, "**Error: Image alignment failed** - Insufficient SIFT matches"
    
    # Find annotation (automatic - from filename)
    xml_path = find_annotation(test_file.name)
    actual = get_actual_defects(xml_path) if xml_path else []
    
    # Calculate metrics
    metrics = calculate_metrics(detected, actual)
    
    # Result image
    result = draw_results(ref_img, detected, actual, metrics)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Difference image (colored)
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    diff_rgb = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
    
    # Defect type
    defect_name, defect_desc = get_defect_type(filename)
    
    # Report
    report = f"## {defect_name}\n\n"
    
    if xml_path:
        # Ground truth available - detailed report
        report += f"### Results\n\n"
        report += f"| | Count |\n|---|---|\n"
        report += f"| **True Positive (TP)** | {metrics['tp']} |\n"
        report += f"| **False Positive (FP)** | {metrics['fp']} |\n"
        report += f"| **False Negative (FN)** | {metrics['fn']} |\n\n"
        
        report += f"### Metrics\n\n"
        report += f"| Metric | Value |\n|---|---|\n"
        report += f"| Precision | {metrics['precision']*100:.1f}% |\n"
        report += f"| Recall | {metrics['recall']*100:.1f}% |\n"
        report += f"| F1 Score | {metrics['f1']*100:.1f}% |\n\n"
        
        # Missed defects locations
        if metrics['fn'] > 0:
            report += f"### Missed Defects\n\n"
            missed_indices = [j for j in range(len(actual)) if j not in metrics['matched_gt']]
            for idx, j in enumerate(missed_indices, 1):
                x1, y1, x2, y2 = actual[j]
                report += f"- **FN #{idx}**: Location ({x1}, {y1}) - ({x2}, {y2})\n"
            report += "\n"
        
        # False alarms locations
        if metrics['fp'] > 0:
            report += f"### False Alarms\n\n"
            fp_indices = [i for i in range(len(detected)) if i not in metrics['matched_det']]
            for idx, i in enumerate(fp_indices, 1):
                x1, y1, x2, y2 = detected[i]
                report += f"- **FP #{idx}**: Location ({x1}, {y1}) - ({x2}, {y2})\n"
    else:
        # No ground truth
        report += f"**{len(detected)} defects detected**\n\n"
        report += "> Note: No annotation found for this image, showing detections only.\n\n"
        
        if len(detected) > 0:
            report += "| # | Location | Size |\n|---|---|---|\n"
            for i, (x1, y1, x2, y2) in enumerate(detected, 1):
                report += f"| {i} | ({x1}, {y1}) | {x2-x1}×{y2-y1} px |\n"
    
    report += f"\n---\n### Legend\n"
    report += f"- **Green box** = True positive (TP)\n"
    report += f"- **Red box** = False positive (FP)\n"
    report += f"- **Orange X** = Missed defect (FN)\n"
    report += f"- **Dashed green** = Ground truth location\n"
    
    return result_rgb, diff_rgb, report

# ============== GRADIO UI ==============

def create_ui():
    with gr.Blocks(title="PCB Defect Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# PCB Defect Detection System

Upload a PCB image. Reference and annotations are automatically located.
""")
        
        with gr.Row():
            with gr.Column(scale=1):
                test_file = gr.File(
                    label="Upload PCB Image",
                    file_types=["image"]
                )
                detect_btn = gr.Button("Detect Defects", variant="primary", size="lg")
        
        with gr.Row():
            result_output = gr.Image(label="Detection Result")
            diff_output = gr.Image(label="Difference Map")
        
        result_text = gr.Markdown()
        
        detect_btn.click(
            fn=run_detection,
            inputs=[test_file],
            outputs=[result_output, diff_output, result_text]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(inbrowser=True)
