import cv2
import mediapipe as mp
import math
import time

fire_mode = False
fire_start_time = 0

# Function to overlay a PNG image with transparency onto the background frame
def overlay_png(background, overlay, x, y):

    h, w = overlay.shape[:2]
    # Check if the overlay goes out of bounds of the background
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return

    # Check if the overlay position is negative
    if x < 0 or y < 0:
        return

    # Separate the color and alpha channels of the overlay
    overlay_rgb = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0

    background_region = background[y:y+h, x:x+w]

    # Blend the overlay with the background using the alpha mask
    blended = (
        background_region * (1 - mask) +
        overlay_rgb * mask
    ).astype("uint8")

    # Place the blended result back onto the background
    background[y:y+h, x:x+w] = blended


def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Set up the Hands model with desired parameters
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Load the fireball image with an alpha channel and resize it to 200x200 pixels
fireball = cv2.imread("fire.png", cv2.IMREAD_UNCHANGED)
fireball = cv2.resize(fireball, (200, 200))

# Main loop to process video frames
while True:

# Read a frame from the webcam
    success, frame = cap.read()

    if not success:
        break

# Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

# Convert the frame from BGR to RGB color space for MediaPipe processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the RGB frame to detect hand landmarks
    results = hands.process(rgb)

# If hand landmarks are detected, iterate through each detected hand
    if results.multi_hand_landmarks:

        # Loop through each detected hand's landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2:
                    hand1 = results.multi_hand_landmarks[0]
                    hand2 = results.multi_hand_landmarks[1]
            
        # define new pose
            


            # # Get the thumb tip and index finger tip landmarks
            # thumb_tip = hand_landmarks.landmark[4]
            # index_tip = hand_landmarks.landmark[8]

            # # Calculate the distance between the thumb tip and index finger tip
            # pinch_distance = distance(thumb_tip, index_tip)
            # # Get the dimensions of the frame to convert normalized coordinates to pixel coordinates
            # h, w, _ = frame.shape

            # # Convert the normalized coordinates of the index finger tip to pixel coordinates
            # x = int(index_tip.x * w)
            # y = int(index_tip.y * h)

            # # If the pinch distance is below a certain threshold, overlay the fireball image at the index finger tip position
            # if pinch_distance < 0.05:

            #     overlay_png(
            #         frame,
            #         fireball,
            #         x - 100,
            #         y - 100
            #     )

            # Draw the hand landmarks and connections on the frame for visualization
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display the resulting frame with hand landmarks and fireball overlay
    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()