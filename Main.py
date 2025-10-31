from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
from PIL import Image
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

Image_Folder = r"Birds/test" # Retrieve the image folder
placeholder = st.empty()
for filename in os.listdir(Image_Folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        image_path = os.path.join(Image_Folder, filename)
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        outputs = model.generate(**inputs)
        captions = processor.decode(outputs[0], skip_special_tokens = True)
        print(f"{filename} : {captions}")
        placeholder.image(image_path, caption=captions)
        
                

