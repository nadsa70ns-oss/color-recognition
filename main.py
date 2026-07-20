import cv2
import numpy as np


# Detect colors in a frame
def detect_colors(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colors = [
        ("Red", np.array([0, 120, 70]), np.array([10, 255, 255]), (0, 0, 255)),
        ("Green", np.array([35, 50, 50]), np.array([85, 255, 255]), (0, 255, 0)),
        ("Blue", np.array([100, 150, 50]), np.array([140, 255, 255]), (255, 0, 0)),
        ("White", np.array([0, 0, 200]), np.array([180, 30, 255]), (255, 255, 255))
    ]

    for name, lower, upper, color in colors:

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            if cv2.contourArea(contour) > 800:

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                cv2.putText(
                    frame,
                    name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

    return frame


# Select mode
mode = input("Choose mode (camera/image): ").strip().lower()

if mode == "camera":

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detect_colors(frame)

        cv2.imshow("Color Recognition", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

elif mode == "image":

    image_path = input("Enter image path: ")

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found.")

    else:
        result = detect_colors(image)

        cv2.imshow("Color Recognition", result)

        cv2.waitKey(0)

else:
    print("Invalid option.")

cv2.destroyAllWindows()