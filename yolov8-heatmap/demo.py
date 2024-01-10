from ultralytics import YOLO
import heatmap
import cv2
import argparse

model = YOLO("yolov8s.pt") # YOLOv8 custom/pretrained model
cap = cv2.VideoCapture("testing.mp4")

# Video writer
video_writer = cv2.VideoWriter("heatmap_output.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               int(cap.get(5)),
                               (int(cap.get(3)), int(cap.get(4))))

# Heatmap init
heatmap_obj = heatmap.Heatmap()

heatmap_obj.set_args(colormap=cv2.COLORMAP_TURBO,
                     imw=cap.get(4),  # should same as im0 width
                     imh=cap.get(3),  # should same as im0 height
                     view_img=False)

while True: 
    ret, frame = cap.read() 
    if(ret): 
        results = model.track(frame, persist=True)
        frame = heatmap_obj.generate_heatmap(frame, tracks=results)
        video_writer.write(frame)
    else:
        break

video_writer.release()
cap.release()
