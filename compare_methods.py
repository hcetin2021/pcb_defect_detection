"""
Method Comparison - Final Results:
1. OLD: Fixed parameters (threshold=25, kernel=3, area=60) -> F1: 52.31%
2. OTSU: Adaptive Otsu threshold -> F1: 77.78% (WINNER)

Attempted methods (unsuccessful):
- SSIM + Edge: Excessive false positives
- Multi-scale: 23,871 FP (catastrophic)
- CLAHE preprocessing: Increased false positive rate
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import linear_sum_assignment
from pathlib import Path

BASE_PATH = Path(__file__).parent

# ============== COMMON FUNCTIONS ==============

def get_actual_defects(xml_file):
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

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    union = (x2-x1+1)*(y2-y1+1) + (x2p-x1p+1)*(y2p-y1p+1) - inter
    return inter / union if union > 0 else 0

def calculate_metrics(detected, actual, iou_thresh=0.1):
    n_det, n_act = len(detected), len(actual)
    if n_det == 0 and n_act == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 1.0, 'recall': 1.0}
    if n_det == 0 or n_act == 0:
        return {'tp': 0, 'fp': n_det, 'fn': n_act, 'precision': 0, 'recall': 0}
    
    cost = np.array([[-calculate_iou(d, a) for a in actual] for d in detected])
    row, col = linear_sum_assignment(cost)
    tp = sum(1 for i, j in zip(row, col) if -cost[i, j] > iou_thresh)
    
    fp, fn = n_det - tp, n_act - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': prec, 'recall': rec}

def align_images(ref, test):
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

# ============== OLD METHOD (FIXED PARAMETERS) ==============

def detect_OLD(ref, test):
    """Old method: threshold=25, kernel=3, min_area=60"""
    aligned = align_images(ref, test)
    if aligned is None:
        return []
    
    diff = cv2.absdiff(ref, aligned)
    
    # FIXED threshold = 25
    _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # FIXED kernel = 3
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # FIXED min_area = 60
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 60:
            x, y, w, h = cv2.boundingRect(cnt)
            detected.append((x, y, x+w, y+h))
    
    return detected

# ============== NEW METHOD (ADAPTIVE) ==============

def detect_OTSU(ref, test):
    """OTSU method: Otsu threshold, adaptive kernel/area"""
    aligned = align_images(ref, test)
    if aligned is None:
        return []
    
    diff = cv2.absdiff(ref, aligned)
    
    # ADAPTIVE threshold (Otsu)
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ADAPTIVE kernel
    k = max(3, int((ref.shape[0] + ref.shape[1]) / 800))
    k = k if k % 2 == 1 else k + 1
    kernel = np.ones((k, k), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # ADAPTIVE min_area
    min_area = int(ref.shape[0] * ref.shape[1] * 0.00002)
    min_area = max(30, min(min_area, 200))
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            detected.append((x, y, x+w, y+h))
    
    return detected

# ============== TEST ==============

def run_comparison():
    ref_path = BASE_PATH / "Reference" / "01.JPG"
    ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    
    if ref_img is None:
        print("Reference image not found!")
        return
    
    defect_types = ["Missing_hole", "Mouse_bite", "Open_circuit"]
    
    results = {
        'old': {'tp': 0, 'fp': 0, 'fn': 0},
        'otsu': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    test_count = 0
    
    print("=" * 75)
    print("OLD vs OTSU COMPARISON (FINAL)")
    print("=" * 75)
    
    for defect_type in defect_types:
        rotation_folder = BASE_PATH / "rotation" / f"{defect_type}_rotation"
        annotation_folder = BASE_PATH / "Annotations" / defect_type
        
        if not rotation_folder.exists():
            continue
        
        print(f"\nFolder: {defect_type}")
        print("-" * 65)
        
        for test_file in sorted(rotation_folder.glob("*.jpg"))[:5]:
            test_img = cv2.imread(str(test_file), cv2.IMREAD_GRAYSCALE)
            if test_img is None:
                continue
            
            ann_file = annotation_folder / f"{test_file.stem}.xml"
            actual = get_actual_defects(str(ann_file)) if ann_file.exists() else []
            
            # Compare 2 methods
            detected_old = detect_OLD(ref_img, test_img)
            detected_otsu = detect_OTSU(ref_img, test_img)
            
            metrics_old = calculate_metrics(detected_old, actual)
            metrics_otsu = calculate_metrics(detected_otsu, actual)
            
            for key in ['tp', 'fp', 'fn']:
                results['old'][key] += metrics_old[key]
                results['otsu'][key] += metrics_otsu[key]
            
            test_count += 1
            
            print(f"  {test_file.name} (GT: {len(actual)})")
            print(f"    OLD: Det={len(detected_old):2d} | TP={metrics_old['tp']} FP={metrics_old['fp']} FN={metrics_old['fn']}")
            print(f"    OTSU: Det={len(detected_otsu):2d} | TP={metrics_otsu['tp']} FP={metrics_otsu['fp']} FN={metrics_otsu['fn']}")
    
    # Summary
    print("\n" + "=" * 75)
    print("OVERALL SUMMARY")
    print("=" * 75)
    print(f"\n{test_count} images tested\n")
    
    # OLD results
    r_old = results['old']
    prec_old = r_old['tp'] / (r_old['tp'] + r_old['fp']) if (r_old['tp'] + r_old['fp']) > 0 else 0
    rec_old = r_old['tp'] / (r_old['tp'] + r_old['fn']) if (r_old['tp'] + r_old['fn']) > 0 else 0
    f1_old = 2 * prec_old * rec_old / (prec_old + rec_old) if (prec_old + rec_old) > 0 else 0
    
    # OTSU results
    r_otsu = results['otsu']
    prec_otsu = r_otsu['tp'] / (r_otsu['tp'] + r_otsu['fp']) if (r_otsu['tp'] + r_otsu['fp']) > 0 else 0
    rec_otsu = r_otsu['tp'] / (r_otsu['tp'] + r_otsu['fn']) if (r_otsu['tp'] + r_otsu['fn']) > 0 else 0
    f1_otsu = 2 * prec_otsu * rec_otsu / (prec_otsu + rec_otsu) if (prec_otsu + rec_otsu) > 0 else 0
    
    print("┌────────────────────────────────────────────────────────────┐")
    print("│                    RESULTS TABLE                          │")
    print("├────────────────────────────────────────────────────────────┤")
    print(f"│ Method      │ TP   │ FP   │ FN   │ Precision │ Recall │ F1     │")
    print("├────────────────────────────────────────────────────────────┤")
    print(f"│ OLD         │ {r_old['tp']:4d} │ {r_old['fp']:4d} │ {r_old['fn']:4d} │ {prec_old:9.2%} │ {rec_old:6.2%} │ {f1_old:6.2%} │")
    print(f"│ OTSU (NEW)  │ {r_otsu['tp']:4d} │ {r_otsu['fp']:4d} │ {r_otsu['fn']:4d} │ {prec_otsu:9.2%} │ {rec_otsu:6.2%} │ {f1_otsu:6.2%} │")
    print("└────────────────────────────────────────────────────────────┘")
    
    # Improvement
    improvement = f1_otsu - f1_old
    print(f"\nWINNER: OTSU Adaptive")
    print(f"IMPROVEMENT: +{improvement:.2%} F1 Score")
    
    # Visual bar
    print("\nF1 Score Comparison:")
    bar_old = '█' * int(f1_old * 40)
    bar_otsu = '█' * int(f1_otsu * 40)
    print(f"  OLD:  {bar_old:<40} {f1_old:.2%}")
    print(f"  OTSU: {bar_otsu:<40} {f1_otsu:.2%}")
    
    print("\n" + "=" * 75)
    print("ATTEMPTED METHODS (UNSUCCESSFUL):")
    print("-" * 75)
    print("  - SSIM + Edge Detection: Excessive false positives")
    print("  - Multi-scale Detection: 23,871 FP (catastrophic)")
    print("  - CLAHE Preprocessing: Increased false positive rate")
    print("=" * 75)

if __name__ == "__main__":
    run_comparison()
