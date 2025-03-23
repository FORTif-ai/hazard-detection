import cv2
import numpy as np

cap = cv2.VideoCapture(0)
step_size = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()
    height, width = frame.shape[:2]

    # === EDGE SEPARATION (CANNY) ===
    img_edgerep = frame.copy()
    blur = cv2.bilateralFilter(img_edgerep, 9, 40, 40)
    edges = cv2.Canny(blur, 50, 100)

    # For edge overlay
    img_edgerep_h = height - 1
    img_edgerep_w = width - 1
    edge_array = []

    for j in range(0, img_edgerep_w, step_size):
        pixel = (j, 0)
        for i in range(img_edgerep_h - 5, 0, -1):
            if edges.item(i, j) == 255:
                pixel = (j, i)
                break
        edge_array.append(pixel)

    for x in range(len(edge_array) - 1):
        cv2.line(img_edgerep, edge_array[x], edge_array[x + 1], (0, 255, 0), 1)
    for x in range(len(edge_array)):
        cv2.line(img_edgerep, (x * step_size, img_edgerep_h), edge_array[x], (0, 255, 0), 1)

    # === THRESHOLDING + CONTOUR DETECTION ===
    blurred = cv2.bilateralFilter(frame.copy(), 9, 75, 75)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 106, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detection_result = frame.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(detection_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(detection_result, "Hazard", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === CONVERT GRAYSCALE IMAGES TO BGR FOR STACKING ===
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh_cleaned, cv2.COLOR_GRAY2BGR)

    # === STACK IMAGES ===
    top_row = cv2.hconcat([original_frame, edges_bgr])
    bottom_row = cv2.hconcat([thresh_bgr, detection_result])
    grid = cv2.vconcat([top_row, bottom_row])

    # Resize grid if it's too large for your screen (optional)
    grid_resized = cv2.resize(grid, (0, 0), fx=0.7, fy=0.7)

    cv2.imshow("Hazard Detection - Combined View", grid_resized)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
