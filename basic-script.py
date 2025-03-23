import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, threshold1=25, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detection_frame = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(detection_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(detection_frame, "Hazard", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    gray_blur_bgr = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)

    top_row = cv2.hconcat([frame, gray_blur_bgr])
    bottom_row = cv2.hconcat([edges_bgr, detection_frame])
    grid = cv2.vconcat([top_row, bottom_row])
    grid_resized = cv2.resize(grid, (0, 0), fx=0.75, fy=0.75)

    cv2.imshow("Hazard Detection - Combined View", grid_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
