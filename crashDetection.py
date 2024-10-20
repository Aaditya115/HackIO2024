from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO('yolov5n.pt')

# Load video
video_path = "./youtube2.mov"
cap = cv2.VideoCapture(video_path)

# Set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


velocity = 0
brake_count = 0

object_in_front = 0

# Read frames
while True:
    ret, frame = cap.read()
    
    current_time = time.time()
    if not ret:
        break  # Exit loop if no frame is captured

    # Detect objects
    results = model.track(frame, persist=True)

    # Process each detection result
    for result in results:
        boxes = result.boxes  # Get detected bounding boxes

        for box in boxes:
            object_id = box.id  # Get the object ID
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Box coordinates

            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Put the object ID on the frame
            cv2.putText(frame, f'ID: {object_id}', (int(x1), int(y1)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            

            # Calculate the height of the bounding box
            shift_in_y = ((y2+y1)/2)
  
            
            # # Calculate distance using the height of the bounding box
            
            # Print distance for object ID 4
            
            velocity = shift_in_y/current_time
            object_in_front = (x2+x1)/2
            print(velocity)
            if velocity >= .00000069:
                              
                brake_count+=1

            if brake_count == 5:
                cv2.putText(frame, 'BRAKE!', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 6, cv2.LINE_AA)
                print('brake')
                brake_count = 0



            #q Draw bounding box on the frame
            

            

    # Display the frame with bounding boxes and IDs
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()


