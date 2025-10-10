from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

Image_Folder = r"Birds/test"
for filename in os.listdir(Image_Folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        image_path = os.path.join(Image_Folder, filename)
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        outputs = model.generate(**inputs)
        captions = processor.decode(outputs[0], skip_special_tokens = True)
        print(f"{filename} : {captions}")
