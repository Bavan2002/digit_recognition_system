"""
Streamlit app for digit recognition.
Upload handwritten digit images to get predictions.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from model import load_model, get_device


@st.cache_resource
def load_digit_model():
    """Load trained model."""
    device = get_device()
    model = load_model("models/mnist_cnn_model.pt", device=str(device))
    return model, device


def preprocess_image(image):
    """Preprocess uploaded image for model."""
    # Convert to grayscale
    img = image.convert("L")

    # Invert if needed (white background -> black background)
    img_array = np.array(img)
    if img_array.mean() > 127:
        img = ImageOps.invert(img)

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to tensor and normalize
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)

    return img_tensor


def main():
    st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="centered")

    st.title("🔢 Handwritten Digit Recognition")
    st.markdown("Upload an image of a handwritten digit (0-9) for recognition.")

    # Load model
    try:
        model, device = load_digit_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Please ensure the model file exists at models/mnist_cnn_model.pt")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Digit Image", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file:
        # Load and display image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess and predict
        img_tensor = preprocess_image(image).to(device)

        with st.spinner("Recognizing digit..."):
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                predicted_digit = output.argmax(1).item()
                confidence = probabilities[predicted_digit].item()

        with col2:
            st.markdown("### Prediction")
            st.success(f"**Digit: {predicted_digit}**")
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        st.markdown("### Probabilities")
        for digit in range(10):
            prob = probabilities[digit].item()
            st.progress(prob, text=f"Digit {digit}: {prob * 100:.1f}%")

    else:
        st.info("📤 Upload an image to begin recognition")

    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - Use clear handwritten digits
        - Single digit per image
        - Black digit on white background (or vice versa)
        - Centered digit
        - Reasonable size (will be resized to 28x28)
        """)


if __name__ == "__main__":
    main()
