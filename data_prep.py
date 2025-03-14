import numpy as np
from tqdm import tqdm
import torch
import timm
from torchvision.datasets import ImageFolder
from torch.utils.data import  DataLoader

imagefolder = "C:/Users/User/images/480_white/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = timm.create_model(
    'tf_efficientnetv2_l.in21k_ft_in1k',
    pretrained=True,
    num_classes=0,       # => no classifier head
    global_pool='avg',   # => global average pooling
    features_only=False  # => return a single (B, 1280) output
)
model = model.to(device)
model.eval()

data_config = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**data_config, is_training=False)

def extract_features(images):
    """Run images through the model and return a (batch_size, 1280) tensor."""
    with torch.no_grad():
        # For features_only=False, the model returns a single tensor of shape (B, 1280)
        outputs = model(images)  # shape: (B, 1280)
        # No need to index with [-1] or do any flattening, it is already (B, 1280).
        return outputs

dataset = ImageFolder(imagefolder, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check feature dimension
sample_images, _ = next(iter(dataloader))
sample_images = sample_images.to(device)
sample_features = extract_features(sample_images)
feature_size = sample_features.size(1)
print(f"Sample feature size: {sample_features.shape}")  # Should be (B, 1280)

# Prepare storage for all features/labels
num_samples = len(dataset)
X = torch.zeros(num_samples, feature_size)
y = torch.zeros(num_samples, dtype=torch.long)

start_idx = 0
for images, labels in tqdm(dataloader):
    images = images.to(device)
    labels = labels.to(device)

    features = extract_features(images)  # shape: (batch_size, 1280)
    batch_size = features.size(0)

    # Move features and labels to CPU and store them
    X[start_idx:start_idx+batch_size] = features.cpu()
    y[start_idx:start_idx+batch_size] = labels.cpu()

    start_idx += batch_size

train_test_ratio = 0.8
train_size = round(train_test_ratio * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Train size is {train_size}")
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Экспорт фичей
X_train_numpy, X_test_numpy = X_train.numpy(), X_test.numpy()
y_train_numpy, y_test_numpy = y_train.numpy(), y_test.numpy()

np.savetxt("X_train_V2_L_480.csv", X_train_numpy, delimiter=",")
np.savetxt("X_test_V2_L_480.csv", X_test_numpy, delimiter=",")
np.savetxt("y_train_V2_L_480.csv", y_train_numpy, delimiter=",")
np.savetxt("y_test_V2_L_480.csv", y_test_numpy, delimiter=",")

print("Done!")
