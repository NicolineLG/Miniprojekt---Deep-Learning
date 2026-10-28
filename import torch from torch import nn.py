from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# Mappen med de forbehandlede billeder
data_dir = Path(
    r"C:\Users\Bruger\Desktop\Skole\Skole\Semester_3"
    r"\Deeplearning\Mini projekt\Minecraft Mob Detection classification"
)
train_dir = data_dir / "train"

# AlexNet bruger 224x224 RGB-billeder og ImageNet-normalisering
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

print("Klasser:", train_data.classes)
print("Antal træningsbilleder:", len(train_data))

# Brug AlexNet med fortrænede vægte
weights = models.AlexNet_Weights.DEFAULT
model = models.alexnet(weights=weights)

# Frys de fortrænede lag
for parameter in model.parameters():
    parameter.requires_grad = False

# Udskift sidste lag, så det forudsiger animals eller monsters
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    len(train_data.classes),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Vægt klasserne, fordi der er flere dyr end monstre i træningsdataene
class_counts = torch.bincount(
    torch.tensor(train_data.targets),
    minlength=len(train_data.classes),
).float()

class_weights = class_counts.sum() / (len(class_counts) * class_counts)
loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=0.001)

# Træn modellen
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(train_loader)
    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- loss: {average_loss:.4f} "
        f"- træfsikkerhed: {accuracy:.1%}"
    )

# Gem den trænede model
save_path = data_dir.parent / "alexnet_animal_monster.pth"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": train_data.classes,
    },
    save_path,
)

print("Modellen er gemt her:", save_path)