import mediapipe as mp
import cv2 as cv


def process(image_path):
    use_static_image_mage = True
    min_detection_confidence = 0.5

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mage,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    )

    # Read an image from the file
    image = cv.imread(image_path)

    processed_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Process the image and draw landmarks.
    results = hands.process(processed_image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            print(f'Hand {hand_index + 1}:')
            for id, landmark in enumerate(hand_landmarks.landmark):
                print(f'\tNormalized landmkars {id}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})')
                # Landmark coordinates are normalized to [0,1]. Thus, you might want to convert them back to the image coordinates.
                landmark_x = int(landmark.x * image.shape[1])
                landmark_y = int(landmark.y * image.shape[0])
                print(f'\tLandmark {id}: (x: {landmark_x}, y: {landmark_y}, z: {landmark.z})')


    # Display the image.
    cv.imshow('Hand Landmarks', image)
    cv.waitKey(0)

    # Clean up
    cv.destroyAllWindows()
    hands.close()
