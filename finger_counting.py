import cv2
import numpy as np
import math

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Preprocessing (masking)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 50], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 2. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find Largest Contour (Hand)
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 1000:
            
            # 3. Convex Hull & Detect Defects
            hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
            hull_points = cv2.convexHull(largest_contour, returnPoints=True)

            # Draw Contour (Green) and Hull (Red)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [hull_points], -1, (0, 0, 255), 2)

            finger_count = 0

            try:
                defects = cv2.convexityDefects(largest_contour, hull_indices)
                
                if defects is not None:
                    # Loop through defects (gaps between fingers)
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(largest_contour[s][0])
                        end = tuple(largest_contour[e][0])
                        far = tuple(largest_contour[f][0])

                        # Calculate triangle sides to filter by angle (optional but good)
                        # a: distance from start to far
                        # b: distance from end to far
                        # c: distance from start to end
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        
                        # Apply cosine rule to find angle of the defect
                        # angle = arccos((b^2 + c^2 - a^2) / (2bc))
                        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57.2958 # convert to degrees

                        # Ignore defects if the angle is > 90 degrees (too wide to be between fingers)
                        if angle <= 90:
                            finger_count += 1
                            cv2.circle(frame, far, 5, [255, 0, 0], -1)

                # Logic: fingers = defects + 1
                # If 0 defects, it could be 1 finger or fist (0). Assuming 1 as default "A".
                total_fingers = finger_count + 1
                
                # Mapping user's request:
                # 1 -> A
                # 2 -> B
                # 3 -> C
                # 4 -> D
                # 5 -> E
                if total_fingers == 1:
                    text = "A (1 Finger)"
                elif total_fingers == 2:
                    text = "B (2 Fingers)"
                elif total_fingers == 3:
                    text = "C (3 Fingers)"
                elif total_fingers == 4:
                    text = "D (4 Fingers)"
                elif total_fingers == 5:
                    text = "E (5 Fingers)"
                else:
                    text = "..."

                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            except Exception as e:
                pass

    cv2.imshow('Finger Counting', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
