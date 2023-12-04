##  Blurred Head and Faces to Protect Privacy 

### Download model trained on crowd human using yolov5(m) architeture
Download Link:  [YOLOv5m-crowd-human](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) 


<br/>

**Output (Crowd Human Model)**

[![image](assets/demo.png)](assets/blur_face.mp4)

<br/>


## Demo 

```bash
python3 detect.py --weights crowdhuman_yolov5m.pt --source _test/ --view-img  --heads
```
