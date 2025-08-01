import cv2

def main():
    # Open the default webcam (0) if you have external camera change the index to (1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the image horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Display the resulting frame
        cv2.imshow("Webcam Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
