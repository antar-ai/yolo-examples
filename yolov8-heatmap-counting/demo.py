from ultralytics import YOLO
import heatmap
import cv2
import argparse

def process(opt):
    weights = opt.weights
    model = YOLO(weights) # YOLOv8 custom/pretrained model
    cap = cv2.VideoCapture(opt.source)
    w,h = cap.get(3), cap.get(4)

    # Bbox for counting: Horizontal, Vertical, Custom
    if opt.count_reg_type:
        count_reg_type = opt.count_reg_type
    else:    
        count_reg_type = "horizontal"
    

    if count_reg_type == "horizontal":
        count_reg_pts = [[0, int(h/2)],[int(w), int(h/2)], [int(w), int((h/2)+int(h*0.05))], [0, int((h/2)+int(h*0.05))]]
    elif count_reg_type == "vertical":
        count_reg_pts = [[int(w/2), 0],[int(w/2), int(h)], [int((w/2)+int(w*0.05)), int(h)], [int((w/2)+int(w*0.05)), 0]]
    else:
        count_reg_pts = []

    # Video writer
    video_writer = cv2.VideoWriter(opt.output_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                int(cap.get(5)),
                                (int(cap.get(3)), int(cap.get(4))))

    # Heatmap init
    heatmap_obj = heatmap.Heatmap()

    if opt.colormap_theme:
        theme = opt.colormap_theme.upper()
    else:
        theme = "TURBO"

    colormaps =  { "AUTUMN" : 0,
    "BONE" : 1,
    "JET" : 2,
    "WINTER" : 3,
    "RAINBOW" : 4,
    "OCEAN" : 5,
    "SUMMER" : 6,
    "SPRING" : 7,
    "COOL" : 8,
    "HSV" : 9,
    "PINK" : 10,
    "HOT" : 11,
    "PARULA" : 12,
    "MAGMA" : 13,
    "INFERNO" : 14,
    "PLASMA" : 15,
    "VIRIDIS" : 16,
    "CIVIDIS" : 17,
    "TWILIGHT" : 18,
    "TWILIGHT_SHIFTED" : 19,
    "TURBO" : 20,
    "DEEPGREEN" : 21
    }

    heatmap_obj.set_args(colormap=colormaps[theme],
                        imw=cap.get(4),  # should same as im0 width
                        imh=cap.get(3),  # should same as im0 height
                        view_img=False,
                        count_reg_pts=count_reg_pts)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8s.pt', help='model.pt path(s)')
    parser.add_argument('--source', required=str, help='video or stream')  # file/folder, 0 for webcam
    parser.add_argument('--output_path', required=str, help='output file path')
    parser.add_argument('--count_reg_type', type=str, help='object confidence threshold')
    parser.add_argument('--colormap_theme', type=str, help='Colormap theme refer to colormaps list')

    opt = parser.parse_args()

    process(opt)