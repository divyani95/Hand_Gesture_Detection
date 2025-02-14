import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Function to count fingers and determine the number
def count_fingers(landmarks):
    # Define the landmark indices for the tips and base of each finger
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    base_ids = [2, 5, 9, 13, 17]  # Base of each finger

    count_fingers = 0  # Initialize the finger count to 0

    # Thumb detection (special case)
    if landmarks[tip_ids[0]].x < landmarks[base_ids[0]].x:
        count_fingers += 1  # Thumb is extended

    # Check other fingers (Index, Middle, Ring, Pinky)
    for i in range(1, 5):
        if landmarks[tip_ids[i]].y < landmarks[base_ids[i]].y:
            count_fingers += 1  # Finger is extended

    return count_fingers

# Function to check odd or even
def odd_or_even(number):
    return "Even" if number % 2 == 0 else "Odd"

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers and determine the number
            num_fingers = count_fingers(hand_landmarks.landmark)
            result = odd_or_even(num_fingers)

            # Display the result
            cv2.putText(frame, f"Number: {num_fingers} ({result})", (10, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()