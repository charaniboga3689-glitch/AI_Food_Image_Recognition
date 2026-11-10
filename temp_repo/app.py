import streamlit as st
from PIL import Image
import os
from predict import predict_image
from nutrition_data import nutrition_db

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Food Recognition", page_icon="🍽️")

st.title("🍽️ AI Food Recognition & Nutrition Analyzer")
st.write("Upload a food image to detect **food name + calories + protein + sugar**")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload a food image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Show image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", width=300)

    # Save temp image
    os.makedirs("temp", exist_ok=True)
    temp_path = "temp/temp.jpg"
    image_pil.save(temp_path)

    # Prediction
    class_name, confidence = predict_image(temp_path)

    # ===============================
    # RESULT CARD
    # ===============================
    st.markdown("## 🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🍕 Food: **{class_name.upper()}**")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence:.2f}**")

    # ===============================
    # NUTRITION INFO
    # ===============================
    with col2:
        if class_name in nutrition_db:
            data = nutrition_db[class_name]

            st.markdown("### 🥗 Nutrition Info (Per Piece)")

            st.metric("🔥 Calories", f"{data['calories']} kcal")
            st.metric("💪 Protein", f"{data['protein']} g")
            st.metric("🍬 Sugar", f"{data['sugar']} g")
        else:
            st.warning("Nutrition data not available for this food.")

    st.markdown("---")
    st.info("⚡ Built using MobileNetV2 + Deep Learning + Streamlit")
