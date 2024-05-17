import cv2
from ultralytics import YOLO

def train_model(yolo_model):
    yolo_model.train(data='data.yaml', epochs=3)

    return yolo_model

def detect_objects_in_photo(yolo_model, image_path):
    image_orig = cv2.imread(image_path)
    
    # yolo_model = YOLO('/home/pc16/Weapons-and-Knives-Detector-with-YOLOv8-main./Weapons-and-Knives-Detector-with-YOLOv8-main/runs/detect/Normal/weights/best.pt')
    # yolo_model.train(data='data.yaml', epochs=5)
    
    #yolo_model.train(data='data.yaml', epochs=3)
    
    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    result_path = r"D:\GIKI\6th semester\CE339L\lab10\final_detection\Weapons-and-Knives-Detector-with-YOLOv8-main\Results\testing2.jpg"
    cv2.imwrite(result_path, image_orig)
    return result_path

def detect_objects_in_video(video_path):
    yolo_model = YOLO('/home/pc16/Weapons-and-Knives-Detector-with-YOLOv8-main./Weapons-and-Knives-Detector-with-YOLOv8-main/runs/detect/Normal/weights/best.pt')
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_video_path = "/home/pc16/Weapons-and-Knives-Detector-with-YOLOv8-main./Weapons-and-Knives-Detector-with-YOLOv8-main/Results/detected_objects_video2.avi"
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
    video_capture.release()
    out.release()

    return result_video_path

def detect_objects_and_plot(path_orig):
    image_orig = cv2.imread(path_orig)
    
    yolo_model = YOLO('/home/pc16/Weapons-and-Knives-Detector-with-YOLOv8-main./Weapons-and-Knives-Detector-with-YOLOv8-main/runs/detect/Normal/weights/best.pt')
    
    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    cv2.imshow("Teste", image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()