import streamlit as st
from PIL import Image
import torch
from torchvision import transforms as T
from inference import get_prediction, draw_boxes

# Load your trained model
model = torch.load("model.pt", map_location=torch.device('cpu'))
model.eval()

# Define your class names based on your training
class_names = ["Helmet", "No Helmet"]

st.title("🛡️ Helmet Detection App")
st.write("Upload an image to check for helmet detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess
    transform = T.Compose([
        T.Resize((600, 600)),
        T.ToTensor()
    ])
    image_tensor = transform(image)

    # Inference
    prediction = get_prediction(model, image_tensor, threshold=0.8, device=torch.device('cpu'))

    # Draw boxes
    if prediction['boxes'].nelement() != 0:
        img_with_boxes = draw_boxes(image_tensor, prediction, class_names)
        st.image(img_with_boxes, caption='Detection Result', use_column_width=True)
    else:
        st.write("No confident detections found!")
