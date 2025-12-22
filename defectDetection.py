import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import linear_sum_assignment

# XML dosyasını oku ve gerçek kusur koordinatlarını al (Doğruluk hesaplamak için)
def get_actual_defects(xml_file):
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

# IoU hesapla
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2

    xi1 = max(x1, x1p)
    yi1 = max(y1, y1p)
    xi2 = min(x2, x2p)
    yi2 = min(y2, y2p)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2p - x1p + 1) * (y2p - y1p + 1)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

# Kusurları karşılaştır ve doğruluk hesapla
def calculate_metrics(detected_defects, actual_defects, iou_threshold=0.1):  # IoU eşik değeri düşürüldü
    num_detected = len(detected_defects)
    num_actual = len(actual_defects)
    
    cost_matrix = np.zeros((num_detected, num_actual))

    for i in range(num_detected):
        for j in range(num_actual):
            cost_matrix[i, j] = -calculate_iou(detected_defects[i], actual_defects[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    true_positives = 0
    for i, j in zip(row_ind, col_ind):
        if -cost_matrix[i, j] > iou_threshold:
            true_positives += 1

    false_positives = num_detected - true_positives
    false_negatives = num_actual - true_positives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return true_positives, false_positives, false_negatives, precision, recall

def calculate_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

# Resimleri yükleyin (Gri tonlamalı olarak)
image1 = cv2.imread("Reference/original.JPG", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("rotation/Missing_hole_rotation/01_missing_hole_18.jpg", cv2.IMREAD_GRAYSCALE)

# SIFT dedektörünü oluşturun
sift = cv2.SIFT_create()
# Özellik noktalarını ve tanımlayıcıları bulun
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# FLANN tabanlı eşleştirici oluşturun
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Tanımlayıcıları eşleştirin
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Lowe'un oran testi ile iyi eşleşmeleri seçin
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Eşleşen noktaları çıkarın
points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

for i, match in enumerate(good_matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Yeterli eşleşme varsa homografi matrisini hesaplayın ve uygulayın
if len(points1) >= 4:
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width = image1.shape
    aligned_image2 = cv2.warpPerspective(image2, H, (width, height))

    # Orijinal ve hizalanmış resimleri çakıştırın
    difference_image = cv2.absdiff(image1, aligned_image2)
    
    # Eşik değeri uygulayın ve açın
    _, thresholded_diff = cv2.threshold(difference_image, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresholded_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_OPEN, kernel)

    # Küçük konturları filtrele
    min_contour_area = 60
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_defects = []
    # Farklı pikselleri kırmızı bir kare ile işaretleyin
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Kırmızı
            cv2.rectangle(aligned_image2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Kırmızı
            detected_defects.append((x, y, x + w, y + h))
    
    # Gerçek kusur koordinatlarını XML dosyasından al
    actual_defects = get_actual_defects("Annotations/Missing_hole/01_missing_hole_18.xml")
    print("Gerçek değerler :", actual_defects)
    print("Bizim bulduğumuz değerler:", detected_defects)

    # Gerçek kusur koordinatlarını sarı ile işaretleyin
    for defect in actual_defects:
        x, y, xmax, ymax = defect
        cv2.rectangle(image1, (x, y), (xmax, ymax), (0, 255, 255), 2) 
    
    # Doğruluk hesaplaması
    true_positives, false_positives, false_negatives, precision, recall = calculate_metrics(detected_defects, actual_defects)

    # mAP hesapla
    recalls = np.array([recall])
    precisions = np.array([precision])
    mAP = calculate_ap(recalls, precisions)
    
    # Her kusurun IoU'sunu hesapla ve ortalama IoU'yu bul
    iou_values = []
    for detected in detected_defects:
        max_iou = 0
        for actual in actual_defects:
            iou = calculate_iou(detected, actual)
            if iou > 0:
                print(f"IoU between {detected} and {actual}: {iou:.2f}")
            if iou > max_iou:
                max_iou = iou
        iou_values.append(max_iou)

    average_iou = sum(iou_values) / len(iou_values) if iou_values else 0
    print(f"Average IoU: {average_iou:.2f}")

    # Görselleştirmeler
    def draw_transparent_rect(image, top_left, bottom_right, color, alpha=0.4):
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    output_image = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    height, width, _ = output_image.shape

    # Tespit edilen kusurlar (kırmızı şeffaf)
    for i, (x, y, xmax, ymax) in enumerate(detected_defects):
        iou = iou_values[i]
        color = (0, 0, 255)  # Kırmızı
        draw_transparent_rect(output_image, (x, y), (xmax, ymax), color)
        cv2.putText(output_image, f'IoU: {iou:.2f}', (x-20 , y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)

    # Gerçek kusurlar (sarı şeffaf)
    for x, y, xmax, ymax in actual_defects:
        draw_transparent_rect(output_image, (x, y), (xmax, ymax), (0, 255, 255))  # Sarı
        
    cv2.namedWindow("Difference Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Difference Image", 720, 480)
    cv2.imshow("Difference Image", difference_image)

    cv2.namedWindow("Detection Results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection Results", 1024, 768)
    cv2.imshow("Detection Results", output_image)

    # Konsolda ek bilgileri yazdır
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"mAP: {mAP:.2f}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
