# Real-World Examples of OpenCV Hazard Detection

Detecting obstacles on the floor in real-time is a common need for robotics and safety systems. OpenCV provides simple yet powerful techniques to identify hazards (like objects or debris) in a live video feed. Below we explore real-world uses of OpenCV for hazard detection and provide a sample Python OpenCV code for a basic hazard detection prototype. Finally, we suggest improvements for better accuracy.

OpenCV is often used in robots to detect floor obstacles and avoid collisions. For example, one hobbyist project used a webcam on a robot to scan for floor hazards: the camera frames were converted to grayscale, blurred, and passed through a Canny edge detector. By scanning upward from the bottom of the edge image, the system could spot where the first edge pixels appeared (indicating an object rising from the floor).  
[Source 1 – Big Face Robotics Blog](https://bigfacerobotics.wordpress.com/2014/12/18/obstacle-detection-using-opencv/#:~:text=The%20method%20I%20am%20using,as%20anything%20found%20here%20will)

Another robot navigation project applied Canny edge detection followed by contour-finding to locate obstacles in the camera view.  
[Source 2 – GitHub: Obstacle Avoidance of a 4-wheeled Robot](https://github.com/F-LAB-Systems/Obstacle-Avoidance-of-a-4-wheeled-robot-using-OPENCV)

In these cases, simple image processing allowed the robot to “see” objects on the ground and plan a path around them.

---

## Scripts

### Canny Edge Detection Script (First One)
1. Capture frame from webcam  
2. Convert frame to grayscale  
3. Apply **Gaussian blur**  
4. Apply **Canny edge detection**  
5. Find contours from the edge map  
6. Filter contours by area  
7. Draw bounding boxes around detected objects  
8. Display original, blurred, edge, and detection frames side-by-side

### More Complex Edge Detection script (Second One) 
1. Capture frame from webcam  
2. Apply **Bilateral filter** to smooth noise while preserving edges  
3. Convert to grayscale  
4. Apply **binary thresholding**  
6. Find contours from the thresholded mask  
7. Filter contours by area  
8. Draw bounding boxes around detected objects  
9. Display original, threshold, edge overlay, and detection frames side-by-side

---

## Evaluation

- **Accuracy**  
  The system performs reasonably well in detecting mid-sized objects in uncluttered environments, especially when lighting is consistent and the object contrasts well with the floor.

- **Challenges**  
  False positives can occur in cluttered backgrounds or under poor lighting. Shadows, floor textures, and edges of furniture are sometimes misidentified as hazards.

- **Improvements**  
  Adding a region of interest (ROI) focused on the floor helped reduce irrelevant detections. Further improvements could include background subtraction or using depth sensors for more robust spatial awareness.

- **Tuning Required**  
  Threshold values and contour area filters must be carefully adjusted depending on the room, lighting, and camera angle. Adaptive thresholding or learning-based segmentation could improve generalizability.

- **Next Steps**  
  Combining traditional OpenCV methods with lightweight machine learning models (e.g., YOLOv5 nano) could increase detection precision while maintaining real-time performance.





