import cv2
from ultralytics import YOLO

def main():
    # Load the segmentation model
    model = YOLO("yolo11m-seg.pt")  # replace with full path if needed

    # Open the default webcam (0) if you have external camera change the index to (1)
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print(" Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for natural view
        frame = cv2.flip(frame, 1)

        # Run YOLO segmentation prediction
        results = model.predict(source=frame, task='segment', verbose=False)

        # Visualize results
        annotated_frame = results[0].plot()
        print(results[0])
        # Show in window
        cv2.imshow("YOLOv11 Segmentation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
