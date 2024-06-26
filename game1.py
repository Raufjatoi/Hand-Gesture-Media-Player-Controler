import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize game state
board = np.zeros((3, 3), dtype=int)
player_turn = 1  # Player 1 starts with 'X'
winner = None
game_mode = None  # None for menu, 1 for multiplayer, 2 for single-player
play_again = False
chosen_game = None  # 1 for Tic-Tac-Toe, 2 for Rock-Paper-Scissors
countdown = 0
rps_player_move = None
rps_computer_move = None

# Function to draw the Tic-Tac-Toe board
def draw_board(img):
    for i in range(1, 3):
        cv2.line(img, (i*300, 0), (i*300, 900), (255, 255, 255), 6)
        cv2.line(img, (0, i*300), (900, i*300), (255, 255, 255), 6)

# Function to draw X and O on the board
def draw_XO(img, board):
    for i in range(3):
        for j in range(3):
            if board[i, j] == 1:
                cv2.putText(img, 'X', (j*300+90, i*300+210), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 15)
            elif board[i, j] == 2:
                cv2.putText(img, 'O', (j*300+90, i*300+210), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 15)

# Function to check for a winner
def check_winner(board):
    for i in range(3):
        if np.all(board[i, :] == board[i, 0]) and board[i, 0] != 0:
            return board[i, 0]
        if np.all(board[:, i] == board[0, i]) and board[0, i] != 0:
            return board[0, i]
    if board[0, 0] == board[1, 1] == board[2, 2] != 0:
        return board[0, 0]
    if board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return board[0, 2]
    if np.all(board != 0):
        return -1  # Draw
    return None

# Minimax algorithm for the computer player
def minimax(board, depth, is_maximizing):
    scores = {1: -1, 2: 1, -1: 0}  # Scores for X (player 1), O (computer), and draw
    winner = check_winner(board)
    if winner is not None:
        return scores[winner]

    if is_maximizing:
        best_score = -np.inf
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board[i, j] = 2
                    score = minimax(board, depth + 1, False)
                    board[i, j] = 0
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = np.inf
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board[i, j] = 1
                    score = minimax(board, depth + 1, True)
                    board[i, j] = 0
                    best_score = min(score, best_score)
        return best_score

# Function to get the best move for the computer
def get_best_move(board):
    best_score = -np.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = 2
                score = minimax(board, 0, False)
                board[i, j] = 0
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

# Function to count fingers
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

# Function to get the gesture position
def get_position(landmarks):
    x = int(landmarks.landmark[8].x * 900)
    y = int(landmarks.landmark[8].y * 900)
    return x // 300, y // 300

# Function to determine the move in Rock-Paper-Scissors
def get_rps_move(landmarks):
    finger_count = count_fingers(landmarks)
    if finger_count == 0:
        return 'Rock'
    elif finger_count == 2 or finger_count == 3:
        return 'Scissors'
    elif finger_count == 5:
        return 'Paper'
    return None

# Function to determine the winner of Rock-Paper-Scissors
def get_rps_winner(player_move, computer_move):
    if player_move == computer_move:
        return 'Draw'
    elif (player_move == 'Rock' and computer_move == 'Scissors') or \
         (player_move == 'Scissors' and computer_move == 'Paper') or \
         (player_move == 'Paper' and computer_move == 'Rock'):
        return 'Player'
    else:
        return 'Computer'

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (900, 900))

    if chosen_game is None:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_count = count_fingers(hand_landmarks)
                if finger_count == 1:
                    chosen_game = 1  # Tic-Tac-Toe
                elif finger_count == 2:
                    chosen_game = 2  # Rock-Paper-Scissors
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, 'Show 1 finger for Tic-Tac-Toe', (60, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        cv2.putText(frame, 'Show 2 fingers for Rock-Paper-Scissors', (60, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        cv2.imshow("Choose Game", frame)
        if cv2.waitKey(1) == 27:  # Esc key to exit
            break
    elif game_mode is None:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_count = count_fingers(hand_landmarks)
                if finger_count == 1:
                    game_mode = 2  # Single-player
                elif finger_count == 2:
                    game_mode = 1  # Multiplayer
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, 'Show 1 finger to play alone', (60, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        cv2.putText(frame, 'Show 2 fingers to play with a friend', (60, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        cv2.imshow("Choose Mode", frame)
        if cv2.waitKey(1) == 27:  # Esc key to exit
            break
    elif play_again:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_count = count_fingers(hand_landmarks)
                if finger_count == 3:
                    board = np.zeros((3, 3), dtype=int)
                    player_turn = 1
                    winner = None
                    play_again = False
                    countdown = 0
                    rps_player_move = None
                    rps_computer_move = None
                elif finger_count == 4:
                    break
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, 'Show 3 fingers to play again', (60, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        cv2.putText(frame, 'Show 4 fingers to quit', (60, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        cv2.imshow("Play Again?", frame)
        if cv2.waitKey(1) == 27:  # Esc key to exit
            break
    elif chosen_game == 1:
        # Tic-Tac-Toe game logic
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_count = count_fingers(hand_landmarks)
                if finger_count == 0:
                    x, y = get_position(hand_landmarks)
                    if x < 3 and y < 3 and board[y, x] == 0:
                        if game_mode == 1 or (game_mode == 2 and player_turn == 1):
                            board[y, x] = player_turn
                            player_turn = 3 - player_turn  # Switch turns
                            winner = check_winner(board)
                            if game_mode == 2 and winner is None:
                                move = get_best_move(board)
                                if move:
                                    board[move[0], move[1]] = player_turn
                                    player_turn = 3 - player_turn  # Switch turns
                                    winner = check_winner(board)
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        draw_board(frame)
        draw_XO(frame, board)

        if winner is not None:
            if winner == -1:
                cv2.putText(frame, 'Draw!', (250, 750), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, f'Player {winner} wins!', (150, 750), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            play_again = True
        else:
            cv2.putText(frame, f'Player {player_turn} turn', (230, 750), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)

        cv2.imshow("Tic-Tac-Toe", frame)
        if cv2.waitKey(1) == 27:  # Esc key to exit
            break
    elif chosen_game == 2:
        # Rock-Paper-Scissors game logic
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if countdown > 0:
                    cv2.putText(frame, str(countdown), (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 3)
                else:
                    rps_player_move = get_rps_move(hand_landmarks)
                    if rps_player_move:
                        if game_mode == 2:
                            rps_computer_move = random.choice(['Rock', 'Paper', 'Scissors'])
                            winner = get_rps_winner(rps_player_move, rps_computer_move)
                        else:
                            # Multiplayer mode: wait for both players to show their move
                            pass
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if countdown > 0:
            countdown -= 1
        elif winner:
            if winner == 'Draw':
                cv2.putText(frame, "It's a draw!", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, f'{winner} wins!', (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            play_again = True
        else:
            cv2.putText(frame, 'Rock-Paper-Scissors', (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
            if countdown == 0:
                cv2.putText(frame, '3', (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 3)
                countdown = 50

        cv2.imshow("Rock-Paper-Scissors", frame)
        if cv2.waitKey(1) == 27:  # Esc key to exit
            break

cap.release()
cv2.destroyAllWindows()
