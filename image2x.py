import streamlit as st
from PIL import Image, PngImagePlugin
import io
import zipfile

# 画像をリサイズする関数
def resize_image(image):
    original_image = Image.open(image)
    # PNG metadataのコピー
    metadata = original_image.info
    size = (original_image.width * 2, original_image.height * 2)
    resized_image = original_image.resize(size, Image.Resampling.LANCZOS)
    # Metadataを再割り当て
    resized_image.info = metadata
    return resized_image

# Streamlit UI
st.title('画像リサイズアプリ')

uploaded_files = st.file_uploader("画像をアップロードしてください", type=['jpg', 'png'], accept_multiple_files=True)

if uploaded_files:
    # Zipファイルの作成
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for uploaded_file in uploaded_files:
            # 画像をリサイズ
            image = resize_image(uploaded_file)
            # メモリ内のバッファに画像を保存
            img_buffer = io.BytesIO()
            # PNG metadataを保持しながら保存
            image.save(img_buffer, format='PNG', pnginfo=image.info)
            img_buffer.seek(0)
            # Zipファイルに追加
            zip_file.writestr(uploaded_file.name, img_buffer.read())

    # ユーザーにダウンロード用のリンクを提供
    zip_buffer.seek(0)
    st.download_button(
        label="画像をダウンロード",
        data=zip_buffer,
        file_name="resized_images.zip",
        mime="application/zip"
    )