import cv2
import numpy as np

url = "http://10.95.49.163:8080/video"
cap = cv2.VideoCapture(url)

# List to store the coordinates of the blue rod's tip
tip_coordinates = []

# Variable to store the initial reference point
initial_tip = None

def find_blue_tip(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the color of the blue rod
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    
    # Create a mask for the color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bottom-most point of the contour
        bottom_most_point = tuple(largest_contour[largest_contour[:,:,1].argmax()][0])
        return bottom_most_point
    
    return None

# Initialize video writer for saving the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (480, 320))

# For soft transitions
previous_frame = None
alpha = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Resize the frame to 480x320
    frame = cv2.resize(frame, (480, 320))

    tip = find_blue_tip(frame)
    if tip is not None:
        if initial_tip is None:
            initial_tip = tip  # Set the initial tip only once
        else:
            initial_tip = tip  # Update the initial tip to track movement
        tip_coordinates.append(initial_tip)
        # Draw the tip coordinate
        cv2.circle(frame, initial_tip, 5, (255, 0, 0), -1)  # Blue for chaotic rod tips

    # Create a mask to draw the path of the blue tip
    path_mask = np.zeros_like(frame)

    # Draw the path of the blue tip with smooth lines
    if len(tip_coordinates) > 1:
        pts = np.array(tip_coordinates, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(path_mask, [pts], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        if len(tip_coordinates) > 50:  # Limit the length of the drawn path
            tip_coordinates.pop(0)

    # Apply Gaussian Blur to the path mask
    path_mask = cv2.GaussianBlur(path_mask, (15, 15), 0)

    # Combine the blurred path mask with the original frame
    frame = cv2.addWeighted(frame, 1, path_mask, 1, 0)

    # For soft transitions, blend the current frame with the previous frame
    if previous_frame is not None:
        frame = cv2.addWeighted(frame, alpha, previous_frame, 1 - alpha, 0)
    previous_frame = frame.copy()
    
    # Display FPS on the frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the frame to video file
    out.write(frame)

    cv2.imshow("Frame", frame)

    q = cv2.waitKey(1)
    if q == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
