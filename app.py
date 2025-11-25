import os
import tempfile
import streamlit as st
from reid_pipeline import run_reid

st.set_page_config(page_title="Person Re-ID Demo", layout="centered")

st.title("🎥 Person Re-Identification Demo")
st.write("Upload a video, and the app will run your 3-way fused Re-ID pipeline on it.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.subheader("Original Input")
    st.video(uploaded_file)

    if st.button("Run Person Re-ID"):
        with st.spinner("Processing video... this may take a while ⏳"):
            # Save uploaded file to a temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(uploaded_file.read())
                input_path = tmp_in.name

            # Output path
            os.makedirs("outputs", exist_ok=True)
            output_path = os.path.join("outputs", "reid_output.mp4")

            # Run your pipeline
            try:
                result_path = run_reid(input_path, output_path)

                st.success("Processing complete! ✅")
                st.subheader("Re-ID Output")
                with open(result_path, "rb") as f:
                    st.video(f.read())

                with open(result_path, "rb") as f:
                    st.download_button(
                        label="Download processed video",
                        data=f,
                        file_name="reid_output.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"Error during processing: {e}")
