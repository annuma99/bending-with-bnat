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
# Particle Effect Images
# Utility Functions
# Pose Detection Functions
# Animation Overlay Functions
# State Variables
# Main Loop

import cv2
import mediapipe as mp
import math
import time
import random
import numpy as np

###################
# Utility Functions
###################

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

def is_hand_fist(hand):
    fingers_curled = 0
    for finger in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]:
        tip = hand.landmark[finger]
        pip = hand.landmark[finger - 2]
        if tip.y > pip.y:  
            fingers_curled += 1

    return fingers_curled >= 3

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

def are_hands_crossed(hand1, hand2):
    wrist1 = hand1.landmark[mp_hands.HandLandmark.WRIST]
    wrist2 = hand2.landmark[mp_hands.HandLandmark.WRIST]

    # wrists must be near each other horizontally
    return abs(wrist1.x - wrist2.x) < 0.2

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

def are_fingers_together(hand, threshold=0.05):
    """
    Returns True if the fingers are together (not splayed).
    """

    index_tip = hand.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP
    ]

    middle_tip = hand.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP
    ]

    ring_tip = hand.landmark[
        mp_hands.HandLandmark.RING_FINGER_TIP
    ]

    pinky_tip = hand.landmark[
        mp_hands.HandLandmark.PINKY_TIP
    ]

    return (
        distance(index_tip, middle_tip) < threshold and
        distance(middle_tip, ring_tip) < threshold and
        distance(ring_tip, pinky_tip) < threshold
    )


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

# def overlay_transparent(background, overlay, x, y):

#     bg_h, bg_w = background.shape[:2]
#     h, w = overlay.shape[:2]

#     # completely outside frame
#     if x >= bg_w or y >= bg_h:
#         return

#     if x + w <= 0 or y + h <= 0:
#         return

#     # clip coordinates
#     x1 = max(x, 0)
#     y1 = max(y, 0)

#     x2 = min(x + w, bg_w)
#     y2 = min(y + h, bg_h)

#     # corresponding overlay coordinates
#     overlay_x1 = x1 - x
#     overlay_y1 = y1 - y

#     overlay_x2 = overlay_x1 + (x2 - x1)
#     overlay_y2 = overlay_y1 + (y2 - y1)

#     # cropped regions
#     background_region = background[y1:y2, x1:x2]

#     overlay_region = overlay[
#         overlay_y1:overlay_y2,
#         overlay_x1:overlay_x2
#     ]

#     # split channels
#     overlay_rgb = overlay_region[:, :, :3]

#     alpha = overlay_region[:, :, 3:] / 255.0

#     # blend
#     blended = (
#         background_region * (1 - alpha) +
#         overlay_rgb * alpha
#     ).astype("uint8")

#     background[y1:y2, x1:x2] = blended


#####################
# Mudras
######################

def detect_mushti(hand):
    """
    Mushti Mudra:
    - all fingers curled in (fist)
    - thumb knuckle visible
    """

    if not is_hand_fist(hand):
        return False

    return True

def gyana_mudra(hand):
    """
    Gyana Mudra:
    - index finger and thumb tips touching
    - other fingers extended
    """

    index_tip = hand.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP
    ]

    thumb_tip = hand.landmark[
        mp_hands.HandLandmark.THUMB_TIP
    ]

    if distance(index_tip, thumb_tip) > 0.1:
        return False

    # Check if other fingers are extended
    for finger in [
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]:
        tip = hand.landmark[finger]
        pip = hand.landmark[finger - 2]  # PIP joint is 2 landmarks before the tip
        if tip.y > pip.y:  # If the tip is below the PIP joint, it's not extended
            return False

    return True

def alapadma_mudra(hand):
    # fingers spread APART (NOT together)
    if are_fingers_together(hand, threshold=0.1):
        return False

    # all fingers extended upward
    for finger in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]:
        tip = hand.landmark[finger]
        pip = hand.landmark[finger - 2]
        if tip.y > pip.y:
            return False

    return True

def bhramara_mudra(hand):
    # index finger curled in, all other fingersout
    # other fingers extended
    index_tip = hand.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP
    ]
    if index_tip.y < hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        return False
    for finger in [
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.THUMB_TIP
        ]:
        tip = hand.landmark[finger]
        pip = hand.landmark[finger - 2]
        if tip.y > pip.y:
            return False
    return True


##########################
# Pose Detection Functions
##########################

# Earth Pose Detection
def detect_earth_pose(hand1, hand2):
    """
    Earth Pose:
    - both hands in mushti mudra (fist with thumb knuckle visible)
    - hands close together (threshold can be adjusted based on testing)
    """
    return (
        detect_mushti(hand1) and
        detect_mushti(hand2) and
        are_hands_close(hand1, hand2, threshold=0.08)
    )

# Water Pose Detection
def detect_water_pose(hand1, hand2):
    """
    Water Pose:
    - both hands in gyana mudra (index finger and thumb tips touching)
    - hands close together (threshold can be adjusted based on testing)
    """
    return (
        gyana_mudra(hand1) and
        gyana_mudra(hand2) and
        are_hands_close(hand1, hand2, threshold=0.08)
        )
def detect_fire_pose(hand1, hand2):
    """
    Fire Pose:
    - both hands in alapadma mudra (all fingers extended and spread apart)
    - hands crossed over each other (threshold can be adjusted based on testing)
    - hands will be vertically oriented (index fingers pointing up)
    """
  
    return (
        alapadma_mudra(hand1) and
        alapadma_mudra(hand2) and
        are_hands_crossed(hand1, hand2) and
        is_index_up(hand1) and
        is_index_up(hand2)
    )

def detect_air_pose(hand1, hand2):
    """
    Air Pose:
    - both hands in bhramara mudra (index finger curled in, all other fingers out)
    - hands crossed over each other (threshold can be adjusted based on testing)
    """
    return (
        bhramara_mudra(hand1) and
        bhramara_mudra(hand2) and
        are_hands_crossed(hand1, hand2)
    )

##########################
# Particle System
##########################
 
particles = []
 
def spawn_particles(x, y, element):
    """Spawn a burst of particles at (x, y) for the given element."""
 
    if element == "fire":
        for _ in range(8):
            particles.append({
                "x":     x + random.randint(-15, 15),
                "y":     y + random.randint(-5, 5),
                # fire shoots upward with a little sideways drift
                "vx":    random.uniform(-1.5, 1.5),
                "vy":    random.uniform(-6, -2),
                "life":  1.0,
                "decay": random.uniform(0.025, 0.05),
                "size":  random.randint(6, 16),
                # color shifts from white-yellow at birth to deep red at death
                # stored as (r_start, g_start, b_start) in BGR
                "color_hot":  (30,  200, 255),  # bright yellow-white (BGR)
                "color_cold": (0,   20,  180),  # deep red (BGR)
                "element": "fire",
            })
 
    elif element == "water":
        for _ in range(6):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            particles.append({
                "x":     x + random.randint(-10, 10),
                "y":     y + random.randint(-10, 10),
                # water flows outward and then drops under gravity
                "vx":    math.cos(angle) * speed,
                "vy":    math.sin(angle) * speed * 0.5 + random.uniform(-1, 0.5),
                "gravity": 0.12,
                "life":  1.0,
                "decay": random.uniform(0.008, 0.02),
                "size":  random.randint(3, 8),
                "color_hot":  (255, 220, 180),  # light blue-white (BGR)
                "color_cold": (180, 80,  20),   # deep blue (BGR)
                "element": "water",
            })
 
    elif element == "earth":
        for _ in range(6):
            particles.append({
                "x":     x + random.randint(-20, 20),
                "y":     y + random.randint(-5, 5),
                # earth chunks fly out and fall with strong gravity
                "vx":    random.uniform(-3, 3),
                "vy":    random.uniform(-4, -1),
                "gravity": 0.3,
                "life":  1.0,
                "decay": random.uniform(0.01, 0.025),
                # large chunky squares
                "size":  random.randint(8, 20),
                "color_hot":  (30,  120, 80),   # light earthy green (BGR)
                "color_cold": (10,  50,  100),  # dark brown (BGR)
                "element": "earth",
                # earth particles are square, not circular
                "shape": random.choice(["rect", "rect", "circle"]),
                # slight rotation per frame
                "angle": random.uniform(0, 360),
                "spin":  random.uniform(-5, 5),
            })
 
    elif element == "air":
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 7)
            particles.append({
                "x":    x,
                "y":    y,
                # air blasts outward in all directions fast
                "vx":   math.cos(angle) * speed,
                "vy":   math.sin(angle) * speed,
                "life": 1.0,
                "decay": random.uniform(0.04, 0.08),
                "size":  random.randint(2, 6),
                "color_hot":  (255, 255, 255),  # white (BGR)
                "color_cold": (180, 180, 180),  # light gray (BGR)
                "element": "air",
                # air particles get a turbulence nudge each frame
                "turbulence": random.uniform(0.3, 0.8),
            })
 
 
def lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )
 
 
def draw_circle_alpha(frame, cx, cy, radius, color, alpha):
    """Draw a filled circle with transparency onto frame."""
    if radius < 1 or cx < 0 or cy < 0:
        return
    h, w = frame.shape[:2]
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, w)
    y2 = min(cy + radius, h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.circle(overlay, (cx - x1, cy - y1), radius, color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi
 
 
def draw_rect_alpha(frame, cx, cy, size, angle, color, alpha):
    """Draw a rotated rectangle with transparency onto frame."""
    if size < 1:
        return
    pts = cv2.boxPoints(((cx, cy), (size, size), angle))
    pts = np.int32(pts)
    h, w = frame.shape[:2]
    x1 = max(np.min(pts[:, 0]) - 2, 0)
    y1 = max(np.min(pts[:, 1]) - 2, 0)
    x2 = min(np.max(pts[:, 0]) + 2, w)
    y2 = min(np.max(pts[:, 1]) + 2, h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    shifted = pts - np.array([x1, y1])
    cv2.fillPoly(overlay, [shifted], color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi
 
 
def update_and_draw_particles(frame):
    """Move, age, draw, and remove all particles."""
    dead = []
 
    for p in particles:
        # age
        p["life"] -= p["decay"]
        if p["life"] <= 0:
            dead.append(p)
            continue
 
        # move
        p["x"] += p["vx"]
        p["y"] += p["vy"]
 
        # gravity (water, earth)
        if "gravity" in p:
            p["vy"] += p["gravity"]
 
        # turbulence (air)
        if "turbulence" in p:
            p["vx"] += random.uniform(-p["turbulence"], p["turbulence"])
            p["vy"] += random.uniform(-p["turbulence"], p["turbulence"])
 
        # fire rises: accelerate upward slightly
        if p["element"] == "fire":
            p["vy"] -= 0.1
 
        # earth spins
        if p["element"] == "earth":
            p["angle"] += p["spin"]
 
        # color: interpolate from hot to cold as life drops
        t = 1.0 - p["life"]   # 0 at birth, 1 at death
        color = lerp_color(p["color_hot"], p["color_cold"], t)
 
        # size shrinks with life (fire and air shrink faster)
        current_size = int(p["size"] * p["life"])
 
        alpha = p["life"]  # fully opaque at birth, transparent at death
 
        cx, cy = int(p["x"]), int(p["y"])
 
        # draw
        if p["element"] == "earth" and p.get("shape") == "rect":
            draw_rect_alpha(frame, cx, cy, current_size, p["angle"], color, alpha)
        else:
            draw_circle_alpha(frame, cx, cy, max(current_size, 1), color, alpha)
 
    for p in dead:
        particles.remove(p)
# # Animation Overlay Functions

# # Load the effect images with alpha channel (transparency)
# fire_effect = cv2.imread(
#     "fire.png",
#     cv2.IMREAD_UNCHANGED
# )
# if fire_effect is None:
#     print("Failed to load fire.png")
#     exit()

# water_effect = cv2.imread(
#     "water.png",
#     cv2.IMREAD_UNCHANGED
# )
# if water_effect is None:
#     print("Failed to load water.png")
#     exit()


# earth_effect = cv2.imread(
#     "earth.png",
#     cv2.IMREAD_UNCHANGED
# )
# if earth_effect is None:
#     print("Failed to load earth.png")
#     exit()

# air_effect = cv2.imread(
#     "air.png",
#     cv2.IMREAD_UNCHANGED
# )
# if air_effect is None:
#     print("Failed to load air.png")
#     exit()

# # Resize the effect images to a standard size (e.g., 200x200 pixels)
# fire_effect = cv2.resize(fire_effect, (200, 200))
# water_effect = cv2.resize(water_effect, (200, 200))
# earth_effect = cv2.resize(earth_effect, (200, 200))
# air_effect = cv2.resize(air_effect, (200, 200))

# def render_effect(frame, hands, effect_image):

#     h, w, _ = frame.shape

#     for hand in hands:

#         index_tip = hand.landmark[
#             mp_hands.HandLandmark.INDEX_FINGER_TIP
#         ]

#         x = int(index_tip.x * w)
#         y = int(index_tip.y * h)

#         overlay_transparent(
#             frame,
#             effect_image,
#             x - 100,
#             y - 100
#         )
# def render_fire(frame, hands):
#     render_effect(frame, hands, fire_effect)


# def render_water(frame, hands):
#     render_effect(frame, hands, water_effect)


# def render_earth(frame, hands):
#     render_effect(frame, hands, earth_effect)


# def render_air(frame, hands):
#     render_effect(frame, hands, air_effect)

 
##########################
# Hand landmark positions
##########################
 
def get_hand_points(hand, frame_w, frame_h, hand_id):
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.WRIST,
    ]

    points = []

    for landmark_id in tips:

        lm = hand.landmark[landmark_id]

        raw_x = int(lm.x * frame_w)
        raw_y = int(lm.y * frame_h)

        key = (hand_id, landmark_id)

        if key not in smoothed_points:
            smoothed_points[key] = (raw_x, raw_y)

        old_x, old_y = smoothed_points[key]

        smooth_x = int(old_x * 0.8 + raw_x * 0.2)
        smooth_y = int(old_y * 0.8 + raw_y * 0.2)

        smoothed_points[key] = (smooth_x, smooth_y)

        points.append((smooth_x, smooth_y))

    return points
 
 
##########################
# State Variables
##########################
current_element = None
effect_start_time = 0
effect_duration = 5

smoothed_points = {}



##########################
# Main Loop
##########################
 
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
 
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)
 
while True:
    success, frame = cap.read()
    if not success:
        break
 
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
 
    h, w, _ = frame.shape
 
    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
    # Pose detection
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
 
        if detect_earth_pose(hand1, hand2):
            if current_element is None:
                current_element = "earth"
                effect_start_time = time.time()
                print("EARTH ACTIVATED")
 
        elif detect_water_pose(hand1, hand2):
            if current_element is None:
                current_element = "water"
                effect_start_time = time.time()
                print("WATER ACTIVATED")
 
        elif detect_fire_pose(hand1, hand2):
            if current_element is None:
                current_element = "fire"
                effect_start_time = time.time()
                print("FIRE ACTIVATED")
 
        elif detect_air_pose(hand1, hand2):
            if current_element is None:
                current_element = "air"
                effect_start_time = time.time()
                print("AIR ACTIVATED")
 
    # Spawn + render particles while effect is active
    if current_element is not None:
        elapsed = time.time() - effect_start_time
 
        if elapsed < effect_duration:
            # Spawn from all fingertips of both hands
            if results.multi_hand_landmarks:
                for i, hand in enumerate(results.multi_hand_landmarks):
                    points = get_hand_points(hand, w, h, i)
                    for (px, py) in points:
                        spawn_particles(px, py, current_element)
        else:
            current_element = None
            print("Effect ended")
 
    # Draw all live particles on top of the frame
    update_and_draw_particles(frame)
 
    # HUD — show active element name
    if current_element:
        colors = {
            "fire":  (0,  100, 255),
            "water": (200, 100, 0),
            "earth": (30,  120, 30),
            "air":   (200, 200, 200),
        }
        cv2.putText(
            frame,
            current_element.upper(),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            colors[current_element],
            3,
            cv2.LINE_AA
        )
 
    cv2.imshow("Bending with Bnat", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()