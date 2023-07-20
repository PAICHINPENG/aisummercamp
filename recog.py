import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision import utils
import json
import urllib.request
model = models.resnet50(pretrained=True)

from google.colab import drive
drive.mount('/content/drive')

LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
labels_path = 'imagenet_labels.json'
urllib.request.urlretrieve(LABELS_URL, labels_path)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('/content/drive/MyDrive/SummerCamp/789.jpg')
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)
model.eval()
with torch.no_grad():
    output = model(input_batch)
with open(labels_path) as f:
    labels = json.load(f)

# 讀取預測結果
_, predicted_idx = torch.max(output, 1)
predicted_label = labels[predicted_idx.item()]
print('預測物品名稱：', predicted_label)
