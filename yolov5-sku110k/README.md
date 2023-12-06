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
1. Counting Products shelf-wise
2. Checking for voids in shelf for quick refilling
3. Positions where Products needs to be pulled forward
