#!/usr/bin/env python
# coding: utf-8

# In[31]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import os
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.autograd import Function


# In[32]:


import random
import numpy as np

def set_random_seeds(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # Numpy module.
    random.seed(seed_value)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seeds()


# In[33]:


class PACSDataset(Dataset):
    def __init__(self, root_dir, domains, transform=None):
        self.root_dir = root_dir
        self.domains = domains
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_images_labels()

    def _load_images_labels(self):
        for domain in self.domains:
            domain_dir = os.path.join(self.root_dir, domain)
            classes = sorted(
                [
                    d
                    for d in os.listdir(domain_dir)
                    if os.path.isdir(os.path.join(domain_dir, d))
                ]
            )

            for label, class_name in enumerate(classes):
                class_dir = os.path.join(domain_dir, class_name)
                for image_name in os.listdir(class_dir):
                    if image_name.endswith((".png", ".jpg", ".jpeg")):
                        self.images.append(os.path.join(class_dir, image_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Kiểm tra xem hình ảnh có giá trị NaN hay không
        if torch.isnan(image).any():
            raise ValueError(f"Image at index {idx} contains NaN values.")

        return image, label

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])


# In[34]:


def get_dataloader(root_dir, train_domains, test_domain, batch_size=16):
    train_dataset = PACSDataset(root_dir, train_domains, transform=get_transform())
    val_dataset = PACSDataset(root_dir, train_domains, transform=get_transform())
    test_dataset = PACSDataset(root_dir, [test_domain], transform=get_transform())
    
    # Chia train và validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# In[35]:


from torchvision.models import efficientnet_b7

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # Khởi tạo mô hình EfficientNet-B1 mà không sử dụng pretrained weights
        # self.efficientnet = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.efficientnet = efficientnet_b7(weights=None)

        # Lấy số features từ lớp cuối cùng của EfficientNet-B1
        in_features = self.efficientnet.classifier[1].in_features

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid(),
        )

        # Mean (mu) and log-variance (logvar) layers
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

        self.dropout = nn.Dropout(0.5)  # Add dropout

    def forward(self, x):
        # Pass input through EfficientNet feature extractor
        features = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(features)
        x = torch.flatten(x, 1)

        x = self.dropout(x)  # Apply dropout

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Compute mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# In[36]:


class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, z):
        z = self.dropout(z)
        return self.fc(z)


# In[37]:


class Decoder(nn.Module):
    def __init__(self, latent_dim, num_domains):
        super(Decoder, self).__init__()
        self.domain_embedding = nn.Embedding(num_domains, latent_dim)
        
        self.init_conv = nn.Conv2d(latent_dim, 512, 3, padding=1)
        
        self.up1 = UNetUpBlock(512, 256)
        self.up2 = UNetUpBlock(256, 128)
        self.up3 = UNetUpBlock(128, 64)
        self.up4 = UNetUpBlock(64, 32)
        self.up5 = UNetUpBlock(32, 16)
        
        self.final_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.attention = nn.Sequential(nn.Conv2d(3, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, z, domain_label):
        domain_embed = self.domain_embedding(domain_label)
        z = z + domain_embed
        
        x = z.view(-1, z.size(1), 1, 1)
        x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        x = self.init_conv(x)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        
        x = self.final_conv(x)
        
        attention_map = self.attention(x)
        x = x * attention_map
        
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


# In[38]:


class DomainClassifier(nn.Module):
    def __init__(self, latent_dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradientReversalLayer.apply(x, alpha)


# In[39]:


class LabelSmoothingLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets: torch.Tensor, n_classes: int, smoothing=0.0):
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def reduce_loss(self, loss):
        return (
            loss.mean()
            if self.reduction == "mean"
            else loss.sum() if self.reduction == "sum" else loss
        )

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))

class DynamicWeightBalancer:
    def __init__(self, init_alpha=1.0, init_beta=1.0, init_gamma=1.0, init_delta=1.0, patience=5, scaling_factor=0.7):
        self.alpha = init_alpha  # Reconstruction loss weight
        self.beta = init_beta    # Classification loss weight
        self.gamma = init_gamma  # KL divergence weight
        self.delta = init_delta  # Domain loss weight
        
        self.patience = patience
        self.scaling_factor = scaling_factor
        self.best_loss = float('inf')
        self.counter = 0

    def update(self, current_loss, recon_loss, clf_loss, kl_loss, domain_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.counter = 0
            # Increase classification weight and decrease others
            self.beta /= self.scaling_factor
            self.alpha *= self.scaling_factor
            self.gamma *= self.scaling_factor
            self.delta *= self.scaling_factor

        # Ensure classification loss weight is always significantly larger
        total_weight = self.alpha + self.beta + self.gamma + self.delta
        self.alpha = max(0.1, min(0.3, self.alpha / total_weight))
        self.beta = max(0.6, min(0.8, self.beta / total_weight))
        self.gamma = max(0.05, min(0.15, self.gamma / total_weight))
        self.delta = 1 - self.alpha - self.beta - self.gamma

        return self.alpha, self.beta, self.gamma, self.delta


# In[40]:


def reparameterize(mu, logvar, dropout_rate=0.5):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    z = F.dropout(z, p=dropout_rate, training=True)  # Apply dropout
    return z

def compute_loss(
    reconstructed_imgs_list,
    original_imgs,
    mu,
    logvar,
    predicted_labels,
    true_labels,
    domain_predictions,
    domain_labels,
    clf_loss_fn,
    domain_loss_fn,
    epoch,
    total_epochs,
    balancer,
):
    recon_loss = sum(
        F.mse_loss(recon, original_imgs, reduction="mean")
        for recon in reconstructed_imgs_list
    ) / len(reconstructed_imgs_list)

    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    clf_loss = clf_loss_fn(predicted_labels, true_labels)
    domain_loss = domain_loss_fn(domain_predictions, domain_labels)

    #     alpha, beta, gamma, delta = balancer.update(
    #         recon_loss + clf_loss + kld_loss + domain_loss,
    #         recon_loss,
    #         clf_loss,
    #         kld_loss,
    #         domain_loss,
    #     )
    alpha = 0.1
    beta = 1
    gamma = 0.1
    delta = 0.2
    
    total_loss = (
        alpha * recon_loss + beta * clf_loss + gamma * kld_loss - delta * domain_loss
    )
    return (
        total_loss,
        recon_loss.item(),
        clf_loss.item(),
        kld_loss.item(),
        domain_loss.item(),
        alpha,
        beta,
        gamma,
        delta,
    )


# In[41]:


def mixup_data(x, y, alpha=1.0, device="cuda"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# In[42]:


import copy


def train_model_progressive(
    encoder,
    decoders,
    classifier,
    domain_classifier,
    train_domains,
    test_domain,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    num_epochs=100,
    device="cuda",
    patience=10,
):
    print("Training model with progressive domain adaptation")
    print(f"Number of epochs: {num_epochs}")
    print(f"Patience: {patience}")
    print(f"Train domains: {train_domains}")
    print(f"Test domain: {test_domain}")
    print(f"Device: {device}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    clf_loss_fn = LabelSmoothingLoss(smoothing=0.1)
    domain_to_idx = {
        domain: idx for idx, domain in enumerate(train_domains + [test_domain])
    }
    domain_loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_test_accuracy = 0.0
    patience_counter = 0
    balancer = DynamicWeightBalancer()

    # Để lưu mô hình tốt nhất
    best_model = {"encoder": None, "decoders": None, "classifier": None, "domain_classifier": None}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        encoder.train()
        classifier.train()
        for domain in train_domains:
            decoders[domain].train()

        running_loss = 0.0
        running_recon_loss = 0.0
        running_clf_loss = 0.0
        running_kl_loss = 0.0
        total_samples = 0

        # Training loop on train dataset
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            domain_labels = torch.zeros(inputs.size(0), 1).to(
                device
            )  # 0 for source domain
            inputs, labels_a, labels_b, lam = mixup_data(
                inputs, labels, alpha=0.2, device=device
            )

            mu, logvar = encoder(inputs)
            z = reparameterize(mu, logvar)

            # Forward pass through domain classifier with gradient reversal
            p = float(epoch) / num_epochs
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            domain_predictions = domain_classifier(grad_reverse(z, alpha))

            reconstructed_imgs_list = []
            for domain in train_domains:
                domain_label = torch.tensor(
                    [domain_to_idx[domain]] * inputs.size(0), device=device
                )
                reconstructed_imgs = decoders[domain](z, domain_label)
                reconstructed_imgs_list.append(reconstructed_imgs)

            predicted_labels = classifier(z)

            (
                loss,
                recon_loss,
                clf_loss,
                kl_loss,
                domain_loss,
                alpha,
                beta,
                gamma,
                delta,
            ) = compute_loss(
                reconstructed_imgs_list,
                inputs,
                mu,
                logvar,
                predicted_labels,
                labels,
                domain_predictions,
                domain_labels,
                lambda pred, target: mixup_criterion(
                    clf_loss_fn, pred, labels_a, labels_b, lam
                ),
                domain_loss_fn,
                epoch,
                num_epochs,
                balancer,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            running_loss += loss.item() * inputs.size(0)
            running_recon_loss += recon_loss * inputs.size(0)
            running_clf_loss += clf_loss * inputs.size(0)
            running_kl_loss += kl_loss * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = running_loss / total_samples
        avg_recon_loss = running_recon_loss / total_samples
        avg_clf_loss = running_clf_loss / total_samples
        avg_kl_loss = running_kl_loss / total_samples

        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, Clf: {avg_clf_loss:.4f}, KL: {avg_kl_loss:.4f} Domain: {domain_loss:.4f}"
        )
        print(
            f"Weights - Alpha: {alpha:.4f}, Beta: {beta:.4f}, Gamma: {gamma:.4f}, Delta: {delta:.4f}"
        )

        # Validation
        encoder.eval()
        classifier.eval()
        for domain in train_domains:
            decoders[domain].eval()

        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)

                mu, logvar = encoder(inputs)
                z = reparameterize(mu, logvar)

                reconstructed_imgs_list = []
                for domain in train_domains:
                    domain_label = torch.tensor(
                        [domain_to_idx[domain]] * inputs.size(0), device=device
                    )
                    reconstructed_imgs = decoders[domain](z, domain_label)
                    reconstructed_imgs_list.append(reconstructed_imgs)

                predicted_labels = classifier(z)

                val_loss, _, _, _, _, _, _, _, _ = compute_loss(
                    reconstructed_imgs_list,
                    inputs,
                    mu,
                    logvar,
                    predicted_labels,
                    labels,
                    domain_predictions,
                    domain_labels,
                    clf_loss_fn,
                    domain_loss_fn,
                    epoch,
                    num_epochs,
                    balancer,
                )
                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Đánh giá trên tập test
        if (epoch + 1) % 3 == 0:
            print(
                f"--- Evaluating on Test Domain ({test_domain}) at Epoch {epoch + 1} ---"
            )
            test_accuracy, test_loss, domain_accuracy = evaluate_model(
                encoder, classifier, domain_classifier, test_loader, device
            )
            print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Domain Accuracy: {domain_accuracy:.2f}%")

            # Save best model based on test accuracy
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model["encoder"] = copy.deepcopy(encoder.state_dict())
                best_model["decoders"] = {
                    domain: copy.deepcopy(decoder.state_dict())
                    for domain, decoder in decoders.items()
                }
                best_model["classifier"] = copy.deepcopy(classifier.state_dict())
                best_model["domain_classifier"] = copy.deepcopy(domain_classifier.state_dict())
                print(
                    f"New best model saved with test accuracy: {best_test_accuracy:.2f}%"
                )

        # Early stopping based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        scheduler.step(avg_val_loss)

    # Load best model
    encoder.load_state_dict(best_model["encoder"])
    for domain, state_dict in best_model["decoders"].items():
        decoders[domain].load_state_dict(state_dict)
    classifier.load_state_dict(best_model["classifier"])
    domain_classifier.load_state_dict(best_model["domain_classifier"])

    print(f"Training completed. Best test accuracy: {best_test_accuracy:.2f}%")

    return encoder, decoders, classifier


# In[43]:


def evaluate_model(encoder, classifier, domain_classifier, dataloader, device):
    encoder.eval()
    classifier.eval()
    domain_classifier.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    domain_correct = 0
    clf_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            domain_labels = torch.ones(inputs.size(0), 1).to(
                device
            )  # 1 for target domain

            mu, logvar = encoder(inputs)
            z = reparameterize(mu, logvar)
            outputs = classifier(z)
            domain_outputs = domain_classifier(z)

            loss = clf_loss_fn(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            domain_pred = (domain_outputs > 0.5).float()
            domain_correct += (domain_pred == domain_labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / total
    domain_accuracy = 100 * domain_correct / total
    return accuracy, avg_loss, domain_accuracy


# In[44]:


# Main training and evaluation script
DATA_PATH = "/kaggle/input/pacs-dataset/kfold"
latent_dim = 256
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_domains = ["art_painting", "cartoon", "sketch"]
test_domain = "photo"
all_domains = train_domains + [test_domain]

# Initialize models
encoder = Encoder(latent_dim).to(device)
decoders = {domain: Decoder(latent_dim, len(train_domains)).to(device) for domain in train_domains}
classifier = Classifier(latent_dim, num_classes).to(device)
domain_classifier = DomainClassifier(latent_dim).to(device)

# Optimizer and Scheduler
params = list(encoder.parameters()) + list(classifier.parameters())
for decoder in decoders.values():
    params += list(decoder.parameters())

optimizer = optim.AdamW(params, lr=5e-4, weight_decay=1e-3) 
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
num_epochs = 150  # Tăng số epoch

# Get dataloaders
train_loader, val_loader, test_loader = get_dataloader(DATA_PATH, train_domains, test_domain)

# Train model
encoder, decoders, classifier = train_model_progressive(
    encoder,
    decoders,
    classifier,
    domain_classifier,
    train_domains,
    test_domain,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    num_epochs,
    device=device,
    patience=10,
)

print(f"Final evaluation on test domain: {test_domain}")
test_accuracy, test_loss, domain_accuracy = evaluate_model(
    encoder, classifier, domain_classifier, test_loader, device
)
print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Domain Accuracy: {domain_accuracy:.2f}%")
# Final evaluation on the test domain
print(f"Final evaluation on test domain: {test_domain}")
test_accuracy, test_loss, domain_accuracy = evaluate_model(
    encoder, classifier, domain_classifier, test_loader, device
)
print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f} Domain Accuracy: {domain_accuracy:.2f}%")

