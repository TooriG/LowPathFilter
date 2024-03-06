import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_low_pass_filter(image, threshold):
    # BGRからRGBへ変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # FFTを計算
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # ローパスフィルタのマスク作成
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-threshold:crow+threshold, ccol-threshold:ccol+threshold] = 1

    # マスクを適用して逆FFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # スケールを調整
    img_back = (img_back / np.max(img_back) * 255).astype(np.uint8)
    return img_back

st.title("ローパスフィルタアプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['png', 'jpg', 'jpeg'])

threshold = st.slider("フィルタのしきい値", min_value=1, max_value=100, value=50)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    if image.shape[2] == 4:  # アルファチャンネルが存在する場合は削除
        image = image[:, :, :3]

    filtered_image = apply_low_pass_filter(image, threshold)
    
    # PIL Imageオブジェクトに変換してから表示
    filtered_image_pil = Image.fromarray(filtered_image)
    st.image(filtered_image_pil, caption='フィルタ適用後の画像', use_column_width=True)
