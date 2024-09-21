import cv2
import mediapipe as mp
import warnings
warnings.simplefilter("ignore")

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam video capture
cap = cv2.VideoCapture(0)

# Set up the hand detection model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the frame to RGB as MediaPipe uses RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform hand detection
        results = hands.process(rgb_frame)

        # Convert the frame back to BGR for OpenCV display
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # If hands are detected, draw the landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Optional: Print the coordinates of the landmarks
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    height, width, _ = frame.shape
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    print(f"Landmark {idx}: (X: {x}, Y: {y})")

        # Display the frame with the hand landmarks
        cv2.imshow('Hand and Finger Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
