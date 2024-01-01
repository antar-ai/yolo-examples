from ultralytics import YOLO
import heatmap
import cv2

model = YOLO("yolov8s.pt") # YOLOv8 custom/pretrained model
cap = cv2.VideoCapture("testing.mp4")
w,h = cap.get(3), cap.get(4)

# Bbox for counting: Horizontal, Vertical, Custom
count_reg_type = "horizontal"

if count_reg_type == "horizontal":
    count_reg_pts = [[0, int(h/2)],[int(w), int(h/2)], [int(w), int((h/2)+int(h*0.05))], [0, int((h/2)+int(h*0.05))]]
elif count_reg_type == "vertical":
    count_reg_pts = [[int(w/2), 0],[int(w/2), int(h)], [int((w/2)+int(w*0.05)), int(h)], [int((w/2)+int(w*0.05)), 0]]
else:
    count_reg_pts = []

# Video writer
video_writer = cv2.VideoWriter("heatmap_output.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               int(cap.get(5)),
                               (int(cap.get(3)), int(cap.get(4))))

# Heatmap init
heatmap_obj = heatmap.Heatmap()

heatmap_obj.set_args(colormap=cv2.COLORMAP_JET,
                     imw=cap.get(4),  # should same as im0 width
                     imh=cap.get(3),  # should same as im0 height
                     view_img=False,
                     count_reg_pts=count_reg_pts)

while cap.isOpened(): 
    ret, frame = cap.read() 
    if(ret): 
        results = model.track(frame, persist=True)
        frame = heatmap_obj.generate_heatmap(frame, tracks=results)
        video_writer.write(frame)
         
video_writer.release()
cap.release()
