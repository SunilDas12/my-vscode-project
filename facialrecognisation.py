import cv2
from deepface import DeepFace
import warnings
warnings.simplefilter("ignore")

# Initialize the webcam video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Detect faces in the frame and analyze emotions
    try:
        # Analyze the frame for facial expressions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        print(result)
        dominant_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']

        # Display the dominant emotion on the frame
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print(f"Error: {e}")

    # Show the frame with the emotion text
    cv2.imshow('Facial Expression Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
