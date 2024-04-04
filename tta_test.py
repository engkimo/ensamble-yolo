import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
from TTA.cococls import get_cls
import TTA.odach as oda

cococlass = get_cls() # for viz
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

def loadimg(path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img = cv2.imread(path)
    img = cv2.resize(img, (1024,1024))
    img = img.transpose([2,0,1]) / 255 # 0-1 float!
    return torch.from_numpy(img).unsqueeze(0).to(device).float()

# Declare TTA variations
tta = [oda.HorizontalFlip()]#, oda.VerticalFlip()] #oda.Multiply(0.9), oda.Multiply(1.1)]
scale = [0.8, 0.9, 1, 1.1, 1.2]
# load image
impath = "imgs/cars3.jpg"
img = loadimg(impath)
images=torch.concat((img,img))

#original faster_rcnn input
images=list(img.to(device) for img in images)
# wrap model and tta
model.eval()
tta_model = oda.TTAWrapper(model, tta, scale)
# Execute TTA!
outputs = tta_model(images)

#boxes_list, scores_list, labels_list = [], [], []
for j, (image,prediction) in enumerate(zip(images,outputs)):
    image=(image.cpu().detach().numpy()*255).astype(np.uint8)
    img_channel_last = np.moveaxis(image, 0, -1).copy()

    boxes=prediction['boxes'].cpu().detach().numpy()
    labels=prediction['labels'].cpu().detach().numpy()
    scores=prediction['scores'].cpu().detach().numpy()
#    boxes_list.append(boxes)
 #   scores_list.append(scores)
  #  labels_list.append(labels)
    for i, box in enumerate(boxes):
        cv2.rectangle(img_channel_last,
                  (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])),
                  220, 2)
        cv2.putText(img_channel_last, cococlass[labels[i]] + " {:.2f}".format(scores[i]), (int(box[0]), int(box[3])),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1) # Write the prediction
        
    img_channel_last = cv2.cvtColor(img_channel_last, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'imgs/output{j}.jpg', img_channel_last)

