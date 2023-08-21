import streamlit as st
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from torchvision.models import detection
import torch
from torchvision import models
from io import BytesIO

st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="icon.png",
    layout="centered",
    initial_sidebar_state="expanded",
)

'''@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def instantiate_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path = "model/last.pt", force_reload=True)
    model.eval()
    model.conf = 0.5
    model.iou = 0.45
    return model

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def download_success():
    st.balloons()
    st.success('✅ Download Successful !!')

upload_path = "uploads/"
download_path = "downloads/"
model = instantiate_model()'''

st.title(' Automatic Vehicle Type Recognition 🚘🚙')
st.sidebar.header('Input')
selected_type = st.sidebar.selectbox('Please select any Vedio', ["Upload Vedio", "Live Video Feed"])
text_box= st.sidebar.text_input("Paste link here:")
checkbox_state = st.sidebar.checkbox("Detect Vehicle")
checkbox_state = st.sidebar.checkbox("Read Number Plate")
button=st.sidebar.button("Generate Report")
button=st.sidebar.button("Edit Vedio")
button=st.sidebar.button("Open Gallery")



if selected_type == "Upload Vedio":
    st.info('Supports all popular vedio formats 📷 - MP4, MOV, WEBM and HTML5')
    uploaded_file = st.file_uploader("Upload Vedio", type=["mp4","mov","webm","html5"])

    if uploaded_file is not None:
        with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())
        with st.spinner(f"Working..."):
            uploaded_vedio = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
            downloaded_vedio = os.path.abspath(os.path.join(download_path,str("output_"+uploaded_file.name)))

            with open(uploaded_vedio,'rb') as imge:
                img_bytes = imge.read()

            img = Image.open(io.BytesIO(img_bytes))
            results = model(img, size=640)
            results.render()
            for img in results.imgs:
                img_base64 = Image.fromarray(img)
                img_base64.save(downloaded_vedio, format="MP4")

            final_image = Image.open(downloaded_vedio)
            print("Opening ",final_image
                  )
            st.markdown("---")
            st.image(final_image, caption='This is how your final results looks like')
            with open(downloaded_vedio, "rb") as file:
                if uploaded_file.name.endswith('.mp4') or uploaded_file.name.endswith('.MP4'):
                    if st.download_button(
                                            label="Download Output",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='vedio/mp4'
                                         ):
                        download_success()
                if uploaded_file.name.endswith('.mov') or uploaded_file.name.endswith('.MOV'):
                    if st.download_button(
                                            label="Download Output",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='vedio/mov'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.webm') or uploaded_file.name.endswith('.WEBM'):
                    if st.download_button(
                                            label="Download Output",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='vedio/webm'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.html5') or uploaded_file.name.endswith('.HTML5'):
                    if st.download_button(
                                            label="Download Output",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='vedio/html5'
                                         ):
                        download_success()
    else:
        st.warning('⚠ Please upload your Vedio')


else:
    st.info('The Live Feed from Web-Camera will take some time to load')
    live_feed = st.checkbox('Start Web-Camera ✅')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if live_feed:
        while(cap.isOpened()):
            success, frame = cap.read()
            if success == True:
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                img = Image.open(io.BytesIO(frame))
                model = instantiate_model()
                results = model(img, size=640)
                results.print()
                img = np.squeeze(results.render())
                img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                break
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            FRAME_WINDOW.image(frame)
    else:
        cap.release()
        cv2.destroyAllWindows()
        st.warning('⚠ The Web-Camera is currently disabled.')


