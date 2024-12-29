import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os
import base64
import zipfile
import numpy as np
from PIL import Image

# URL вашего backend сервера
BACKEND_URL = "http://api:8000"

if "list_models" not in st.session_state:
    st.session_state.list_models = []

if "list_models_inference" not in st.session_state:
    st.session_state.list_models_inference = []

st.title("Машинное Обучение Сервис")

# 1. Загрузка датасета
# 1. Upload the ZIP file
st.header("1. Загрузка Данных")
uploaded_zip = st.file_uploader("""Загрузите ZIP файл с фото формата jpg
                                и файлом меток классов этих фото в формате
                                csv с колонками pic_name и label.""", type="zip")

if uploaded_zip is not None:
    # Step 2: Extract ZIP content
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall("uploaded_data")
        extracted_files = zip_ref.namelist()

    st.write("Ваши данные:", extracted_files)

    # Step 3: Find images and the labels file
    image_files = sorted([f for f in extracted_files if f.endswith('.jpg')])
    labels_file = [f for f in extracted_files if f.endswith('.csv')]

    if not labels_file:
        st.error("Файл с метками классов не найден.")
    else:
        st.success(f"Файл меток классов: {labels_file[0]}")

        # Step 4: Read and Parse Labels
        labels_path = os.path.join("uploaded_data", labels_file[0])
        labels_df = pd.read_csv(labels_path)
        if 'pic_name' not in labels_df.columns or 'label' not in labels_df.columns:
            st.error("CSV файл должен содержать колонки 'pic_name' и 'label'.")
        else:
            labels_df.sort_values(by='pic_name', inplace=True)
            labels = labels_df['label'].to_list()

            if len(labels) != len(image_files):
                st.error("Количество фото не совпадает с количеством меток!")
            else:
                st.success("Количество фото совпадает с количеством меток.")

                sorted_images = sorted(image_files)

                # Step 5: Encode images to Base64
                encoded_images = []
                for img_file in sorted_images:
                    img_path = os.path.join("uploaded_data", img_file)
                    with open(img_path, 'rb') as image_file:
                        encoded_bytes = base64.b64encode(image_file.read())
                        encoded_string = encoded_bytes.decode('utf-8')
                        encoded_images.append(encoded_string)

                st.write(f"Закодировано {len(encoded_images)} изображений.")

                payload = {
                    "X": encoded_images,
                    "y": labels
                }

                if st.button("Загрузить на сервер"):
                    response = requests.post(f"{BACKEND_URL}/upload", json=payload)
                    if response.status_code == 201:
                        st.success(json.loads(response.text)['message'])
                    else:
                        st.error(f"Ошибка при загрузке данных. Код ошибки: {response.status_code}")

# 2. Визуалиация EDA
st.header("2. EDA")

if "eda_btn_clicked" not in st.session_state:
    st.session_state['eda_btn_clicked'] = False


def callback():
    # change state value
    st.session_state['eda_btn_clicked'] = True


if st.button('Показать EDA', on_click=callback) or st.session_state['eda_btn_clicked']:
    list_eda_categories = ['Метаданные', 'Изображения']
    eda_category = st.selectbox("Выберите интересующий раздел EDA: ", list_eda_categories)
    if eda_category == 'Метаданные':
        st.image("uploaded_data/eda_graphs/eda_cat_vars_dist.png", caption="")
        st.image("uploaded_data/eda_graphs/eda_age_dist.png", caption="")
        st.image("uploaded_data/eda_graphs/eda_cat_vars_dist.png", caption="")
    else:
        st.image(
            "uploaded_data/eda_graphs/eda_pixel_intens_dist.png",
            caption="Распределение интенсивности ЧБ пикселей"
            )
        st.image(
            "uploaded_data/eda_graphs/eda_pixel_intens_dist_by_chan_by_class.png",
            caption="Распределение интенсивности RGB пикселей"
            )

# 3. Создание новой модели и выбор гиперпараметров
st.header("3. Создание новой модели")

model_id = st.text_input('Назовите модель')

hyperparameters = {}
lr = st.text_input("Learning rate", value=0.00003)
if lr:
    lr = float(lr)
hyperparameters["lr"] = lr
n_epochs = st.slider("Количество эпох", min_value=1, max_value=10, value=3)
hyperparameters["n_epochs"] = n_epochs

config = {'hyperparameters': hyperparameters, 'id': model_id}
if st.button("Создать и обучить модель"):
    payload = {
        "config": config
    }
    response = requests.post(f"{BACKEND_URL}/fit", json=payload)
    if response.status_code == 201:
        st.success(json.loads(response.text)['message'])
    else:
        st.error(f"Ошибка при создании модели. Код ошибки: {response.status_code}")

# 4. Просмотр информации о модели и кривых обучения
st.header("4. Информация о модели и кривые обучения")

# Initialize list_models to empty
list_models = []

# Button to fetch the model list
if st.button("Получить информацию о моделях"):
    response = requests.get(f"{BACKEND_URL}/list_models")
    if response.status_code == 200:
        response_json = response.json()
        message = response_json.get("message", "")
        list_models = message[49:].split(", ")
        st.session_state.list_models = list_models

        # Optional: Handle case when list_models might be `['']`
        # if the message is empty or incorrectly formatted
        if len(list_models) == 1 and not list_models[0]:
            list_models = []
    else:
        st.error(f"Ошибка при получении списка моделей. Код: {response.status_code}")

# Only show multiselect if we have a non-empty list of models
if st.session_state.list_models:
    selected_models = st.multiselect("Выберите модели для просмотра", st.session_state.list_models)

    # Only proceed if the user has selected at least one model
    if selected_models:
        # Request metrics for the selected models
        metrics_response = requests.post(
            f"{BACKEND_URL}/get_metrics",
            json={"models": selected_models},
        )

        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()['responses']
            all_loss_data = []

            for i, model in enumerate(metrics_data):
                st.write(f"Информация о модели {selected_models[i]}:")
                st.write(f"Train Accuracy: {model['Train_Accuracy']}")
                st.write(f"Learning Rate: {model['lr']}")
                st.write(f"N epochs: {model['n_epochs']}\n")

                df_curve = pd.DataFrame({
                            "epoch": np.arange(1, len(model["Loss"]) + 1),
                            "train_loss": model["Loss"],
                            "model": selected_models[i]
                        })
                all_loss_data.append(df_curve)

            if all_loss_data:
                combined_df = pd.concat(all_loss_data, ignore_index=True)

                # Plot all loss curves on one graph
                fig = px.line(
                    combined_df,
                    x="epoch",
                    y="train_loss",
                    color="model",
                    markers=True,
                    title="Кривые обучения для всех выбранных моделей",
                    labels={
                        "train_loss": "Train Loss",
                        "epoch": "Эпоха",
                        "model": "Модель"
                    }
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Не удалось получить метрики модели.")
    else:
        st.info("Пожалуйста, выберите хотя бы одну модель для просмотра метрик.")
else:
    st.info("Нажмите 'Получить информацию о моделях', чтобы загрузить список моделей.")

# 5. Инференс по обученной модели
st.header("5. Инференс по обученной модели")

list_models_inference = []

# Button to fetch the model list
if st.button("Начать Инференс"):
    response = requests.get(f"{BACKEND_URL}/list_models")
    if response.status_code == 200:
        response_json = response.json()
        message = response_json.get("message", "")
        list_models_inference = message[40:].split(", ")
        st.session_state.list_models_inference = list_models_inference
    else:
        st.error(f"Ошибка при получении списка моделей. Код: {response.status_code}")

# Only show multiselect if we have a non-empty list of models
if st.session_state.list_models_inference:
    selected_model = st.selectbox("Выберите модель для предсказания", st.session_state.list_models_inference)

    # Only proceed if the user has selected at least one model
    if selected_model:
        if selected_model in st.session_state.list_models_inference:
            uploaded_jpg = st.file_uploader("""Загрузите jpg файл.""", type="jpg")
            if uploaded_jpg:

                upload_dir = "uploaded_data_inference"

                # Create the directory if it doesn't exist
                os.makedirs(upload_dir, exist_ok=True)

                img_path = os.path.join("uploaded_data_inference", uploaded_jpg.name)

                with open(img_path, 'wb') as image_file:
                    image_file.write(uploaded_jpg.getbuffer())

                img = Image.open(img_path)

                # Convert the image to RGB (in case it's not)
                img = img.convert("RGB")

                # Create a Plotly Express figure
                fig = px.imshow(img)

                # Update layout to remove axes and margins for better visualization
                fig.update_layout(
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    xaxis_visible=False,
                    yaxis_visible=False,
                    margin=dict(l=0, r=0, t=0, b=0)
                )

                # Display the figure in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                with open(img_path, 'rb') as image_file:
                    encoded_bytes = base64.b64encode(image_file.read())
                    encoded_string = encoded_bytes.decode('utf-8')

                payload = {
                            "id": selected_model,
                            "X": encoded_string
                        }

                predict_response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json=payload,
                )
                mapping_dict = {
                    0: "Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)",
                    1: "Basal cell carcinoma",
                    2: "Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)",
                    3: "Dermatofibroma",
                    4: "Melanoma",
                    5: "Nevus",
                    6: "Vascular lesion"
                }
                st.write(f"""The selected model predicts that it is
                        {mapping_dict[predict_response.json()['y']]} on your photo.""")
        else:
            st.error("Модель не найдена.")
# 6. Удалить все созданные модели
st.header("6. Удалить все созданные модели")
if st.button("Удалить все Ваши модели"):
    predict_response = requests.delete(
                    f"{BACKEND_URL}/remove_all"
                )
    st.session_state.list_models = []
    st.session_state.list_models_inference = []
    st.success("Все ваши модели успешно удалены!")
