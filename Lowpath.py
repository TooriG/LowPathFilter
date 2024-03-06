import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_low_pass_filter(image, threshold):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # FFTを計算
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # ローパスフィルタの適用
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    # 中心からしきい値の距離以外を0にする
    fshift[crow-threshold:crow+threshold, ccol-threshold:ccol+threshold] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

st.title("ローパスフィルタアプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['png', 'jpg', 'jpeg'])

threshold = st.slider("フィルタのしきい値", min_value=1, max_value=100, value=50)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    filtered_image = apply_low_pass_filter(image, threshold)
    st.image(filtered_image, caption='フィルタ適用後の画像', use_column_width=True)
