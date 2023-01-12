# AdaFace

`pytorch` implementation of Afaface Neural Network for feature extraction  
[AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964).
[code](https://github.com/mk-minchul/AdaFace).


## How to use it
```python
import torch
from Feature_Extractor import net
from face_alignment import align
import numpy as np
import cv2

adaface_models = {
    'ir_101':"Feature_Extractor/pretrained/adaface_ir101_ms1mv2.ckpt",
}
def load_pretrained_model(architecture='ir_101'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture],map_location=torch.device('cpu'))['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

image = cv2.imread(imagePath)
aligned_rgb_img,box = align.get_aligned_face(image)
if(aligned_rgb_img!=None):
    bgr_tensor_input = to_input(aligned_rgb_img)
    feature, norms = model(bgr_tensor_input)
```

## Requirements
* pytorch 0.2
* Pillow, numpy
* opencv-python