from io import BytesIO
import base64
import copy
import asyncio
from http import HTTPStatus
from typing import List, Tuple
import uvicorn
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from tqdm import tqdm
import timm


app = FastAPI(
    docs_url="/docs",
    json_url="/docs.json",
)

MODEL_PATH = "trained_model_state_L_480_white.pt"


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

model = timm.create_model(
    'tf_efficientnetv2_l.in21k_ft_in1k',
    pretrained=True,
    features_only=True,
)
model = model.to(device)
model.eval()

# Create transforms based on the model data config
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)


# Define a custom dataset that loads images and labels from NumPy arrays
class NumpyDataset(Dataset):
    def __init__(self, images_np: np.ndarray, labels_np: np.ndarray, transform=None):
        self.images_np = images_np
        self.labels_np = labels_np
        self.transform = transform

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Convert one image from NumPy to a PIL Image
        # Make sure the dtype and shape are compatible (e.g., uint8, shape [H, W, C])
        img = self.images_np[idx]

        # Convert image to PIL; if your array is float, you might need to cast to uint8
        # or scale as needed. Adjust accordingly.
        img_pil = Image.fromarray(img.astype('uint8'))

        # Apply transforms if provided
        if self.transform is not None:
            img_pil = self.transform(img_pil)

        label = self.labels_np[idx]
        return img_pil, label


@torch.no_grad()
async def extract_features(images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    last_feature_map = outputs[-1]
    features = last_feature_map.view(last_feature_map.size(0), -1)
    return features


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return out


async def transform_data(images_np: np.ndarray, labels_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = NumpyDataset(images_np, labels_np, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    sample_images, _ = next(iter(dataloader))
    sample_images = sample_images.to(device)
    sample_features = await extract_features(sample_images)
    feature_size = sample_features.size(1)

    num_samples = len(dataset)
    X = torch.zeros(num_samples, feature_size)
    y = torch.zeros(num_samples, dtype=torch.long)

    start_idx = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        features = await extract_features(images)
        batch_size = features.size(0)

        # Move features and labels to CPU and store them
        X[start_idx:start_idx+batch_size] = features.cpu()
        y[start_idx:start_idx+batch_size] = labels.cpu()

        start_idx += batch_size

    return X, y


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.features[idx], self.labels[idx]


default_model = LogisticRegressionModel(92160, 7)
default_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
default_model.to(device)
models = {'default': (default_model, )}

X_user = None
y_user = None

X_user_processed = None
y_user_processed = None


# API endpoints
@app.post("/upload", response_model=MessageResponse, status_code=HTTPStatus.CREATED)
async def upload(request: UploadRequest):
    global X_user, y_user, X_user_processed, y_user_processed
    request = request.model_dump()

    # Decode 1 image
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
    X_user_processed, y_user_processed = await transform_data(X_user, y_user)
    return MessageResponse(message="Your data has been successfully uploaded.")


@app.post("/fit", response_model=MessageResponse, status_code=HTTPStatus.CREATED)
async def fit(request: FitRequest):
    global X_user_processed, y_user_processed

    request = request.model_dump()

    new_model = copy.deepcopy(default_model)

    metrics = {
        'Train_Accuracy': None,
        'Loss': []
        }

    lr = request['config']['hyperparameters']['lr']
    n_epochs = request['config']['hyperparameters']['n_epochs']

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)

    train_dataset = FeatureDataset(X_user_processed, y_user_processed)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(n_epochs):
        new_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
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
    request = request.model_dump()
    metrics_list = []
    for model in request['models']:
        metrics_list.append(models[model][1])
    return MetricsResponses(responses=metrics_list)


@app.post("/remove", response_model=MessageResponse)
async def remove(request: RemoveRequest):
    request = request.model_dump()
    for model in request['models']:
        if model != 'default':
            models.pop(model)
    return MessageResponse(message=f"""Models {", ".join(request['models'])} have been deleted.""")


@app.delete("/remove_all", response_model=MessageResponse)
async def remove_all():
    for model in list(models.keys()):
        if model != 'default':
            models.pop(model)
    return MessageResponse(message='All models have been deleted.')


@app.get("/list_models", response_model=MessageResponse)
async def list_models():
    return MessageResponse(message=f"""We currently have the following models: {", ".join(list(models.keys()))}""")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # Реализуйте инференс загруженной модели
    request = request.model_dump()
    model_id = request['id']

    decoded_bytes = base64.b64decode(request['X'])
    image = Image.open(BytesIO(decoded_bytes))
    images_np = np.asarray([np.asarray(image)])

    X_user_inference, y_user_inference = images_np, np.zeros(1)
    X_user_inference_processed, y_user_inference_processed = await transform_data(X_user_inference, y_user_inference)

    inference_dataset = FeatureDataset(X_user_inference_processed, y_user_inference_processed)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    for features, labels in inference_loader:
        features = features.to(device)
        outputs = models[model_id][0](features)
        _, predicted = torch.max(outputs.data, 1)

    label = predicted[0]
    return PredictResponse(y=label, message=f"""Model {request["id"]} predicts label {label}""")


async def main():
    uvicorn.run("API:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    asyncio.run(main())
