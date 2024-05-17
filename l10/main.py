import cv2
from ultralytics import YOLO
from detectingimages import train_model, detect_objects_in_photo, detect_objects_in_video, detect_objects_and_plot

def main():
    # Test detecting objects in a single image
    #image_path = "Dataset_2/test/images/WhatsApp-Video-2023-11-22-at-19_46_37_mp4-54_jpg.rf.37c622c6a79b700b0b2a707896b63ff1.jpg"

    #model = YOLO('/home/pc16/Weapons-and-Knives-Detector-with-YOLOv8-main./Weapons-and-Knives-Detector-with-YOLOv8-main/runs/detect/Normal/weights/best.pt')
    model = YOLO(r"D:\GIKI\6th semester\CE339L\lab10\final_detection\Weapons-and-Knives-Detector-with-YOLOv8-main\runs\detect\Normal\weights\best.pt")
    #model.export('my_yolo.pt')
    #model.export()
    model.save('my_yolo.pt')
    #model = YOLO(model='my_yolo.pt', task='data.yaml')

    # Comment out the below 2 lines if you don't wish to train the model again
    #train_model(model)
    #model.save('my_yolo.pt')

    image_path = r"D:\GIKI\6th semester\CE339L\lab10\final_detection\Knife_vs_Pistol\eval_pistol\011.jpg"
    result_image_path = detect_objects_in_photo(model, image_path)
    
    print("Object detection result saved at:", result_image_path)

    # Test detecting objects in a video
    #video_path = "test_video.mp4"
    #result_video_path = detect_objects_in_video(video_path)
    #print("Object detection result video saved at:", result_video_path)

    # Test detecting objects in an image and displaying it
    #path_orig = "test_image.jpg"
    #detect_objects_and_plot(path_orig)

if __name__ == "__main__":
    main()
