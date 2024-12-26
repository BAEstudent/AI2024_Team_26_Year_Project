import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os
import base64
import zipfile
from PIL import Image
import numpy as np

# URL вашего backend сервера
BACKEND_URL = "http://127.0.0.1:8000"

if "list_models" not in st.session_state:
    st.session_state.list_models = []

st.title("Машинное Обучение Сервис")

# 1. Загрузка датасета
# 1. Upload the ZIP file 
st.header("1. Загрузка Данных")
uploaded_zip = st.file_uploader("Загрузите ZIP файл с фото формата jpg и файлом меток классов этих фото в формате csv с колонками pic_name и label.", type="zip")

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

# 2. Создание новой модели и выбор гиперпараметров
st.header("2. Создание новой модели")

model_id = st.text_input('Назовите модель')

hyperparameters = {}
lr = st.text_input("Learning rate", value=0.00003)
if lr:
    lr = float(lr)
hyperparameters["lr"] = lr
n_epochs = st.slider("Количество эпох", min_value=1, max_value=10, value=3)
hyperparameters["n_epochs"] = n_epochs

config = {'hyperparameters': hyperparameters, 'id': model_id}
if st.button("Создать и Обучить Модель"):
    payload = {
        "config": config
    }
    response = requests.post(f"{BACKEND_URL}/fit", json=payload)
    if response.status_code == 201:
        st.success(json.loads(response.text)['message'])
    else:
        st.error(f"Ошибка при создании модели. Код ошибки: {response.status_code}")

# 3. Просмотр информации о модели и кривых обучения
st.header("3. Информация о Модели и Кривые Обучения")

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
                    title="Кривые Обучения для всех выбранных моделей",
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

