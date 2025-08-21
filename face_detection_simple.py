import cv2
import numpy as np
import time

# Load the pre-trained face detection model
def load_face_detection_model():
    # Paths to model files
    prototxt_path = "models/deploy.prototxt"
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
    
    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("Face detection model loaded successfully!")
    return net

# Process frame for face detection
def detect_faces_frame(frame, net, confidence_threshold=0.5):
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Create a blob from the frame (preprocessing)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    
    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()
    
    # List to store face coordinates
    faces = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding boxes fall within the dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Add the face coordinates and confidence to the list
            faces.append({
                "box": (startX, startY, endX, endY),
                "confidence": confidence
            })
    
    return faces

# Draw bounding boxes around detected faces
def draw_faces(frame, faces):
    # Make a copy of the frame
    output_frame = frame.copy()
    
    # Draw each face
    for face in faces:
        (startX, startY, endX, endY) = face["box"]
        confidence = face["confidence"]
        
        # Draw the bounding box
        cv2.rectangle(output_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # Draw the confidence label
        text = f"{confidence * 100:.2f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(output_frame, text, (startX, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    return output_frame

# Main function for real-time face detection
def real_time_face_detection():
    # Load the face detection model
    net = load_face_detection_model()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    print("Press 'q' to quit, 's' to save a snapshot")
    
    # For calculating FPS
    prev_time = 0
    fps = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Detect faces in the frame
        faces = detect_faces_frame(frame, net)
        
        # Draw bounding boxes around detected faces
        output_frame = draw_faces(frame, faces)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display number of faces detected
        cv2.putText(output_frame, f"Faces: {len(faces)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the result
        cv2.imshow("Real-Time Face Detection", output_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save snapshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, output_frame)
            print(f"Snapshot saved as {filename}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_face_detection()