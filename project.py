import cv2
import mediapipe as mp
import pyautogui
import time

# Function to count the number of fingers based on landmark positions
def count_fingers(lst):
    count = 0
    threshold = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > threshold:
        count += 1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > threshold:
        count += 1
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > threshold:
        count += 1
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > threshold:
        count += 1
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        count += 1

    return count

# Function to detect if eyes are closed
def eyes_closed(landmarks):
    left_eye_ratio = (landmarks[159].y - landmarks[145].y) / (landmarks[33].x - landmarks[133].x)
    right_eye_ratio = (landmarks[386].y - landmarks[374].y) / (landmarks[362].x - landmarks[263].x)
    return left_eye_ratio < 0.15 and right_eye_ratio < 0.15

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize mediapipe hands and face modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=2)
face_mesh = mp_face_mesh.FaceMesh()

start_init = False
prev = -1
task_text = ""  # Initialize task text

player_minimized = False

while True:
    end_time = time.time()
    ret, frm = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        continue

    frm = cv2.flip(frm, 1)

    # Reduce frame size for performance improvement
    small_frame = cv2.resize(frm, (320, 240))

    # Process the frame with mediapipe hands
    res_hands = hands.process(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
    res_face = face_mesh.process(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))

    if res_face.multi_face_landmarks:
        face_landmarks = res_face.multi_face_landmarks[0]
        if eyes_closed(face_landmarks.landmark):
            if not player_minimized:
                pyautogui.hotkey('win', 'down')  # Minimize the current window
                player_minimized = True
                task_text = "Player Minimized"
        else:
            if player_minimized:
                pyautogui.hotkey('win', 'up')  # Restore the current window
                player_minimized = False
                task_text = "Player Restored"

    total_fingers = 0

    if res_hands.multi_hand_landmarks:
        for hand_landmarks in res_hands.multi_hand_landmarks:
            total_fingers += count_fingers(hand_landmarks)
            mp_drawing.draw_landmarks(frm, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if not(prev == total_fingers):
            if not(start_init):
                start_time = time.time()
                start_init = True
            elif (end_time - start_time) > 0.2:
                if total_fingers == 1:
                    pyautogui.press("right")  # Press right arrow key for forward skip
                    task_text = "Forward Skip"
                elif total_fingers == 2:
                    pyautogui.press("left")  # Press left arrow key for backward skip
                    task_text = "Backward Skip"
                elif total_fingers == 3:
                    pyautogui.press("volumeup")  # Press volume up key for increasing volume
                    task_text = "Volume Up"
                elif total_fingers == 4:
                    pyautogui.press("volumedown")  # Press volume down key for decreasing volume
                    task_text = "Volume Down"
                elif total_fingers == 5:
                    pyautogui.press("space")  # Press spacebar for pause/play
                    task_text = "Pause/Play"
                elif total_fingers == 6:
                    pyautogui.hotkey('win', 'down')  # Minimize the current window
                    task_text = "Brightness Up"
                elif total_fingers == 8:
                    pyautogui.hotkey('win', 'up')  # Restore the current window
                    task_text = "Brightness Down"
                elif total_fingers == 10:
                    if not player_minimized:
                        pyautogui.hotkey('win', 'down')  # Minimize the current window
                        player_minimized = True
                        task_text = "Player Minimized"
                    else:
                        pyautogui.hotkey('win', 'up')  # Restore the current window
                        player_minimized = False
                        task_text = "Player Restored"

                prev = total_fingers
                start_init = False

    # Draw heading and task text
    cv2.putText(frm, "Hand Gesture Media Player Controller", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frm, (10, 50), (500, 100), (0, 0, 0), -1)  # Draw a filled black rectangle for task text
    cv2.putText(frm, task_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        print("Exiting the program.")
        cv2.destroyAllWindows()
        cap.release()
        break
