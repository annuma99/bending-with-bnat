# Ananya Iyer
# Project: Bending with Bnat
# This file contains the code for the bending pose detection using MediaPipe Hands. 
# with given hand landmarks, we will detect if the user is making the bending pose and 
# over lay the corresponding animation on the video feed based on the "element"
# they are giving the pose for.
# That animation will last for 5 seconds and then disappear until the user gives the pose again.

# There will 4 elements: fire, water, earth, and air. Each element will have a 
# corresponding animation that will be overlaid on the video feed when the user makes 
# the bending pose for that element.

# 5 sections to this file are:
# Utility Functions
# Pose Detection Functions
# Animation Overlay Functions
# State Variables
# Main Loop

import cv2
import mediapipe as mp
import math
import time


# Utility Functions


def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def is_index_up(hand):
    """
    Returns True if the index finger is extended upward.
    """

    index_tip = hand.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP
    ]

    index_pip = hand.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_PIP
    ]

    return index_tip.y < index_pip.y


def is_thumb_up(hand):
    """
    Returns True if the thumb is extended upward.
    """

    thumb_tip = hand.landmark[
        mp_hands.HandLandmark.THUMB_TIP
    ]

    thumb_ip = hand.landmark[
        mp_hands.HandLandmark.THUMB_IP
    ]

    return thumb_tip.y < thumb_ip.y


def are_hands_close(hand1, hand2, threshold=0.15):
    """
    Returns True if the two index fingertips are close together.
    """

    index1 = hand1.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP
    ]

    index2 = hand2.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP
    ]

    return distance(index1, index2) < threshold

def overlay_transparent(background, overlay, x, y):

    h, w = overlay.shape[:2]

    # prevent drawing outside frame
    if x < 0 or y < 0:
        return

    if x + w > background.shape[1]:
        return

    if y + h > background.shape[0]:
        return

    # split alpha channel
    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0

    # region of interest
    roi = background[y:y+h, x:x+w]

    # blend
    blended = (
        roi * (1 - alpha[:, :, None]) +
        overlay_rgb * alpha[:, :, None]
    ).astype("uint8")

    background[y:y+h, x:x+w] = blended

# Pose Detection Functions


# Earth Pose Detection
def detect_earth_pose(hand1, hand2):
    """
    Earth Pose:
    - both hands present
    - both index fingers up
    - hands close together
    """

    if not is_index_up(hand1):
        return False

    if not is_index_up(hand2):
        return False

    if not are_hands_close(hand1, hand2):
        return False

    return True



# Water Pose Detection


def detect_water_pose(hand1, hand2):
    """
    Water Pose:
    - one hand thumb up
    - one hand index up
    - hands separated slightly
    """

    hand1_thumb = is_thumb_up(hand1)
    hand2_thumb = is_thumb_up(hand2)

    hand1_index = is_index_up(hand1)
    hand2_index = is_index_up(hand2)

    # one hand acts vertical
    # one hand acts horizontal

    condition1 = hand1_thumb and hand2_index
    condition2 = hand2_thumb and hand1_index

    if not (condition1 or condition2):
        return False

    if are_hands_close(hand1, hand2, threshold=0.08):
        return False

    return True

# Animation Overlay Functions

# Load the effect images with alpha channel (transparency)
fire_effect = cv2.imread(
    "fire.png",
    cv2.IMREAD_UNCHANGED
)
if fire_effect is None:
    print("Failed to load fire.png")
    exit()

water_effect = cv2.imread(
    "water.png",
    cv2.IMREAD_UNCHANGED
)
if water_effect is None:
    print("Failed to load water.png")
    exit()


earth_effect = cv2.imread(
    "earth.png",
    cv2.IMREAD_UNCHANGED
)
if earth_effect is None:
    print("Failed to load earth.png")
    exit()

air_effect = cv2.imread(
    "air.png",
    cv2.IMREAD_UNCHANGED
)
if air_effect is None:
    print("Failed to load air.png")
    exit()

# Resize the effect images to a standard size (e.g., 200x200 pixels)
fire_effect = cv2.resize(fire_effect, (200, 200))
water_effect = cv2.resize(water_effect, (200, 200))
earth_effect = cv2.resize(earth_effect, (200, 200))
air_effect = cv2.resize(air_effect, (200, 200))

def render_effect(frame, hands, effect_image):

    h, w, _ = frame.shape

    for hand in hands:

        index_tip = hand.landmark[
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        overlay_transparent(
            frame,
            effect_image,
            x - 100,
            y - 100
        )
def render_fire(frame, hands):
    render_effect(frame, hands, fire_effect)


def render_water(frame, hands):
    render_effect(frame, hands, water_effect)


def render_earth(frame, hands):
    render_effect(frame, hands, earth_effect)


def render_air(frame, hands):
    render_effect(frame, hands, air_effect)

# State Variables
current_element = None
effect_start_time = 0
effect_duration = 5




# Main Loop

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)
while True:

    success, frame = cap.read()

    if not success:
        break

    # mirror webcam
    frame = cv2.flip(frame, 1)

    # convert BGR -> RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process hands
    results = hands.process(rgb)

    
    # Draw Hand Landmarks
    

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Pose Detection


    if results.multi_hand_landmarks:

        # only continue if BOTH hands detected
        if len(results.multi_hand_landmarks) == 2:

            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]

            
            # Earth Pose
            if detect_earth_pose(hand1, hand2):

                # only activate if nothing active
                if current_element is None:

                    current_element = "earth"

                    effect_start_time = time.time()

                    print("EARTH ACTIVATED")

            # Water Pose
            elif detect_water_pose(hand1, hand2):

                if current_element is None:

                    current_element = "water"

                    effect_start_time = time.time()

                    print("WATER ACTIVATED")


            # # Fire Pose
            # elif detect_fire_pose(hand1, hand2):

            #     if current_element is None:

            #         current_element = "fire"

            #         effect_start_time = time.time()

            #         print("FIRE ACTIVATED")


            # # Air Pose
            # elif detect_air_pose(hand1, hand2):

            #     if current_element is None:

            #         current_element = "air"

            #         effect_start_time = time.time()

            #         print("AIR ACTIVATED")

    
    # Render Active Effect
    if current_element is not None:

        elapsed = time.time() - effect_start_time

        # keep effect active for 5 seconds
        if elapsed < effect_duration:

            if current_element == "fire":

                render_fire(
                    frame,
                    results.multi_hand_landmarks
                )

            elif current_element == "water":

                render_water(
                    frame,
                    results.multi_hand_landmarks
                )

            elif current_element == "earth":

                render_earth(
                    frame,
                    results.multi_hand_landmarks
                )

            elif current_element == "air":

                render_air(
                    frame,
                    results.multi_hand_landmarks
                )

        else:

            # reset after timer expires
            current_element = None

            print("Effect Ended")

    # Show Webcam Feed

    cv2.imshow(
        "Bending with Bnat",
        frame
    )

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# cleanup
cap.release()
cv2.destroyAllWindows()