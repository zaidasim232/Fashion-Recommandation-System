import streamlit as st
import pickle as pkl
import pandas as pd
import os
import zipfile
import gdown

from PIL import Image
import os

# Sayfa Başlığı
st.title("Moda Öneri Sistemi")

#google driveden fotoğrafları çekme yükleme
# Dosyaları Drive'dan indiren güvenli yapı
@st.cache_resource
def download_pkl_files():
    if not os.path.exists('tfidf_matrix.pkl'):
        # tfidf_matrix.pkl Drive ID'si
        gdown.download(
            'https://drive.google.com/uc?id=DRIVE_FILE_ID_1',
            'tfidf_matrix.pkl',
            quiet=False,
        )
    if not os.path.exists('fashion_products.pkl'):
        gdown.download(
            'https://drive.google.com/uc?id=DRIVE_FILE_ID_2',
            'fashion_products.pkl',
            quiet=False,
        )
    if not os.path.exists('fashion_recommender_model.pkl'):
        gdown.download(
            'https://drive.google.com/uc?id=DRIVE_FILE_ID_3',
            'fashion_recommender_model.pkl',
            quiet=False,
        )


download_pkl_files()

#Dosyaları yükleme
@st.cache_resource
def load_files():
    df = pkl.load(open('fashion_products.pkl', 'rb'))
    tfidf_matrix = pkl.load(open('tfidf_matrix.pkl', 'rb'))
    model = pkl.load(open('fashion_recommender_model.pkl', 'rb'))
    df = df.reset_index(drop=True)
    return df, tfidf_matrix, model

df, tfidf_matrix, model = load_files()

#ürün önerme fonksiyonu
def recommend_products(item_index, model, tfidf_matrix):
    query = tfidf_matrix[item_index]
    distances, indices = model.kneighbors(query, n_neighbors=6)
    return indices[0], distances[0]

# Ürünü gösteren fonksiyon
def recommend_and_display(item_index, model, tfidf_matrix):
    indices, distances = recommend_products(item_index, model, tfidf_matrix)

    cols = st.columns(6)

    # indices doğrudan 1D dizi olduğu için enumerate ile sütun indeksini (col_idx) ve ürün indeksini (product_idx) alıyoruz
    for col_idx, product_idx in enumerate(indices):
        with cols[col_idx]:

            img_path = df.loc[product_idx, 'image_path']

            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            else:
                st.error("Görsel Yok")

            # Ürün Adı ve Mesafe Değeri
            st.text(df.loc[product_idx, 'productDisplayName'])


# STREAMLIT ARAYÜZÜ
search_query = st.text_input(
    "Aramak istediğiniz ürün adını yazın (ör: T-shirt, Watch, Jeans, Blue):"
)

if search_query:
    # Arama terimine uyan ilk 30 ürünü filtrele
    filtered_df = df[
        df['productDisplayName'].str.contains(
            search_query, case=False, na=False
        )
    ].head(30)

    if not filtered_df.empty:
        selected_idx = st.selectbox(
            "Bulunan ürünlerden birini seçin:",
            options=filtered_df.index,
            format_func=lambda idx: f"{idx} - {df.loc[idx, 'productDisplayName']}",
            index=None,
            placeholder="Ürün seçiniz...",
        )

        if selected_idx is not None:
            st.subheader("Seçilen Ürün")
            col_main1, col_main2 = st.columns([1, 3])

            with col_main1:
                img_path = df.loc[selected_idx, 'image_path']
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, width=200)
                else:
                    st.warning("Fotoğraf bulunamadı")

            with col_main2:
                st.write(
                    f"**Ürün Adı:** {df.loc[selected_idx, 'productDisplayName']}"
                )

            st.write("---")
            st.subheader("Önerilen Benzer Ürünler")
            recommend_and_display(selected_idx, model, tfidf_matrix)
    else:
        st.warning("Aranan kritere uygun ürün bulunamadı.")
