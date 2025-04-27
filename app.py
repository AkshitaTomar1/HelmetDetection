import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
from inference import get_prediction, draw_boxes

# Load your trained model
model = torch.load("model.pt", map_location=torch.device('cpu'))
model.eval()



# Recreate the model with correct number of classes
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=3)

# Load the trained weights
#model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
#model.eval()



# Define class names properly (3 classes)
class_names = [
    "rider-helmet-bike",  # ID 0
    "With Helmet",        # ID 1
    "Without Helmet"      # ID 2
]

st.title("🛡️ Helmet Detection App")
st.write("Upload an image to check for helmet detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess
    transform = T.Compose([
        T.Resize((600, 600)),  # Optionally: use Resize + CenterCrop to preserve aspect ratio better
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
