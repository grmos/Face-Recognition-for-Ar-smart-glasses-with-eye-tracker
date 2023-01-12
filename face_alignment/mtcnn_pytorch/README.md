# MTCNN

`pytorch` implementation of **inference stage** of face detection algorithm described in  
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

## How to use it
```python
from src import detect_faces
from PIL import Image

image = Image.open('image.jpg')
bounding_boxes, landmarks = detect_faces(image)
```

## Requirements
* pytorch 0.2
* Pillow, numpy
