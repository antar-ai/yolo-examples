##  Stock Keeping Unit Maintainance using YOLOv5 

### Download model trained on SKU-110k dataset using yolov5(m) architeture
Download Link:  [YOLOv5m-SKU-110k](https://drive.google.com/file/d/1KlEYqkNpK9JlpnBDndIjeGpZGreVQiFr/view?usp=sharing) 


<br/>

**Output (Stock Keeping Units)**

![image](assets/result.png)

<br/>


## Demo 

```bash
python3 detect.py --weights yolov5m-sku110k.pt --data data/SKU-110K.yaml --source output1.mp4 --line-thickness 1
```


## Future Works:
1. Conducting shelf-specific product counts.
2. Identifying and filling shelf gaps promptly.
3. Adjusting product positions for optimal visibility and accessibility.