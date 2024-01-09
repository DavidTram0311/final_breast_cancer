import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import os

# import style

st.title('PyTorch Breast Cancer Prediction')
path = os.getcwd()

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "png")

model_fineTuning = torch.load(f"{path}/Resnet_fineTuning.pth", map_location=torch.device('cpu'))

def predict_img_class(img, model):
    class_names = ['benign', 'malignant','normal',]
    device = torch.device('cpu')
    model = model.to(device)
    image = Image.open(img)

    #convert image to tensor
    img = transforms.Resize(256)(image)
    # img = transforms.CenterCrop(224)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = img.to(device)

    # Add an extra dimension and move the tensor back to the original device
    img = img.unsqueeze(0)

    print(img.shape)

    with torch.no_grad():
      model.eval()
      outputs = model(img)
      _, preds = torch.max(outputs, 1)

    return class_names[preds.item()]

if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict_img_class(file_up, model_fineTuning)

    # print out the top 5 prediction labels with scores
    st.write("Prediction: ", labels)
