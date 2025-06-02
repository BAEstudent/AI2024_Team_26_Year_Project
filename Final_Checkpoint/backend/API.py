from io import BytesIO
import base64
import copy
import asyncio
from http import HTTPStatus
from typing import List, Tuple
import logging
from logging.handlers import TimedRotatingFileHandler
import json
import uvicorn
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI, Request
from pydantic import BaseModel
from PIL import Image
from tqdm import tqdm
import timm
from starlette.middleware.base import BaseHTTPMiddleware


app = FastAPI(
    docs_url="/docs",
    json_url="/docs.json",
)

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")  # Используем uvicorn логгер, чтобы соблюдать консистенцию\

handler = TimedRotatingFileHandler(
    'logs/logs.log', when="midnight", interval=1, backupCount=7
)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)


class LogRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        '''
        Прокидывает логгер в FastAPI
        '''
        # Захватываем тело запроса (если нужно)
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Логирование запроса
        logger.info("Request: %s  %s  - Body: %s", request.method, request.url, json.dumps(body))

        # Получаем и возвращаем ответ
        response = await call_next(request)

        # Логирование ответа
        logger.info("Response status: %s", response.status_code)

        return response


app.add_middleware(LogRequestsMiddleware)

MODEL_PATH = "best_model_n_0_cosine_s.pt"
NUM_CLASSES = 7


class UploadRequest(BaseModel):
    X: List[str]
    y: List[int]


class Hyperparameters(BaseModel):
    lr: float
    n_epochs: int


class Config(BaseModel):
    hyperparameters: Hyperparameters
    id: str


class FitRequest(BaseModel):
    config: Config


class MetricsRequest(BaseModel):
    models: List[str]


class RemoveRequest(BaseModel):
    models: List[str]


class MessageResponse(BaseModel):
    message: str


class MetricsResponse(BaseModel):
    Train_Accuracy: float
    Loss: List[float]
    lr: float
    n_epochs: int


class MetricsResponses(BaseModel):
    responses: List[MetricsResponse]


class PredictRequest(BaseModel):
    id: str
    X: str


class PredictResponse(BaseModel):
    y: int
    message: str


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

base_model = timm.create_model(
    "tf_efficientnetv2_s.in21k_ft_in1k",
    pretrained=False,
    num_classes=NUM_CLASSES,
)
in_features = base_model.classifier.in_features
base_model.classifier = nn.Sequential(
    nn.Linear(in_features, NUM_CLASSES)
)
base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
base_model.to(device)
base_model.eval()

data_config = timm.data.resolve_model_data_config(base_model)
transform = timm.data.create_transform(**data_config, is_training=False)


class ImageDataset(Dataset):
    """Dataset that keeps raw numpy images and their labels."""

    def __init__(self, images_np: np.ndarray, labels_np: np.ndarray, transform=None):
        self.images_np = images_np
        self.labels_np = labels_np
        self.transform = transform

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images_np[idx].astype("uint8")
        img_pil = Image.fromarray(img)
        if self.transform is not None:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
        label = int(self.labels_np[idx])
        return img_tensor, label


models = {'default': (base_model, )}

X_user = None
y_user = None


# API endpoints
@app.post("/upload", response_model=MessageResponse, status_code=HTTPStatus.CREATED)
async def upload(request: UploadRequest):
    '''
    Ручка загрузки датасета
    '''
    global X_user, y_user
    request = request.model_dump()

    decoded_bytes = base64.b64decode(request['X'][0])
    image = Image.open(BytesIO(decoded_bytes))
    images_np = np.asarray([np.asarray(image)])

    # Decode the rest of the images
    if len(request['X']) > 1:
        for bytes_image in request['X'][1:]:
            decoded_bytes = base64.b64decode(bytes_image)
            image = np.asarray([Image.open(BytesIO(decoded_bytes))])
            images_np = np.append(images_np, image, axis=0)

    X_user, y_user = images_np, np.array(request['y'])
    return MessageResponse(message="Your data has been successfully uploaded.")


@app.post("/fit", response_model=MessageResponse, status_code=HTTPStatus.CREATED)
async def fit(request: FitRequest):
    '''
    Ручка обучения модели
    '''

    request = request.model_dump()

    new_model = copy.deepcopy(base_model)

    metrics = {
        'Train_Accuracy': None,
        'Loss': []
        }

    lr = request['config']['hyperparameters']['lr']
    n_epochs = request['config']['hyperparameters']['n_epochs']

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)

    train_dataset = ImageDataset(X_user, y_user, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(n_epochs):
        new_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in tqdm(train_loader):
            # Move data to device
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = new_model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Statistics
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        metrics['Loss'].append(epoch_loss)

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    metrics['Train_Accuracy'] = epoch_acc
    metrics['lr'] = lr
    metrics['n_epochs'] = n_epochs
    models.update({request['config']['id']: (new_model, metrics)})
    return MessageResponse(message=f"""Model {request['config']['id']} trained and saved""")


@app.post("/get_metrics", response_model=MetricsResponses)
async def get_metrics(request: MetricsRequest):
    '''
    Ручка для получения информации о модели
    '''
    request = request.model_dump()
    metrics_list = []
    for model_id in request['models']:
        metrics_list.append(models[model_id][1])
    return MetricsResponses(responses=metrics_list)


@app.post("/remove", response_model=MessageResponse)
async def remove(request: RemoveRequest):
    '''
    Удаление модели по имени
    '''
    request = request.model_dump()
    for model_id in request['models']:
        if model_id != 'default':
            models.pop(model_id)
    return MessageResponse(message=f"""Models {", ".join(request['models'])} have been deleted.""")


@app.delete("/remove_all", response_model=MessageResponse)
async def remove_all():
    '''
    Удаление всех моделей
    '''
    for model_id in list(models.keys()):
        if model_id != 'default':
            models.pop(model_id)
    return MessageResponse(message='All models have been deleted.')


@app.get("/list_models", response_model=MessageResponse)
async def list_models():
    '''
    Получение списка моделей
    '''
    return MessageResponse(message=f"""We currently have the following models: {", ".join(list(models.keys()))}""")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    req = request.model_dump()
    model_id = req["id"]
    if model_id not in models:
        return PredictResponse(y=-1, message=f"Model '{model_id}' not found.")

    decoded_bytes = base64.b64decode(req["X"])
    img = Image.open(BytesIO(decoded_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model = models[model_id][0]
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    label = int(predicted.item())
    return PredictResponse(y=label, message=f"Model '{model_id}' predicts label {label}.")


async def main():
    '''
    Поднимаем сервер
    '''
    uvicorn.run("API:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    asyncio.run(main())
