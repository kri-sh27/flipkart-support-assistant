# from pathlib import Path
# import torch
# from torch import nn
# from torchvision import transforms, models
# from PIL import Image
# ROOT=Path(__file__).resolve().parents[1]
# CLASSES=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
# TRANSFORM=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
# def load_model():
#     ckpt=torch.load(ROOT/'models/product_classifier.pt',map_location=DEVICE)
#     model=models.resnet18(weights=None)
#     if ckpt['finetuned']:
#         model.fc=nn.Linear(model.fc.in_features,10)
#         model.load_state_dict(ckpt['state_dict'])
#     else:
#         model.fc=nn.Linear(model.fc.in_features,10)
#         model.fc.load_state_dict(ckpt['state_dict'])
#     return model.to(DEVICE).eval()
# def predict(image_path:str)->dict:
#     model=load_model(); image=Image.open(image_path).convert('L'); x=TRANSFORM(image).unsqueeze(0).to(DEVICE)
#     with torch.no_grad(): p=torch.softmax(model(x),dim=1)[0]; conf,idx=p.max(0)
#     return {'category':CLASSES[int(idx)],'confidence':float(conf)}
# if __name__=='__main__':
#     import sys; print(predict(sys.argv[1]))
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Same preprocessing used for ResNet-18
TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


def load_model():

    checkpoint = torch.load(
        ROOT / "models" / "product_classifier.pt",
        map_location=DEVICE,
    )

    # ------------------------------------------------------
    # Important:
    # Use the SAME pretrained ResNet-18 backbone used
    # during Part 2 feature extraction.
    # ------------------------------------------------------

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        10,
    )

    if checkpoint["finetuned"]:

        # Fine-tuned checkpoint contains the
        # complete ResNet state dictionary.

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        # Feature-extraction checkpoint contains
        # only the trained classifier head.

        model.fc.load_state_dict(
            checkpoint["state_dict"]
        )

    return model.to(DEVICE).eval()


def predict(
    image_path: str,
) -> dict:

    model = load_model()

    image = (
        Image.open(image_path)
        .convert("L")
    )

    x = (
        TRANSFORM(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        probabilities = torch.softmax(
            model(x),
            dim=1,
        )[0]

        confidence, index = (
            probabilities.max(0)
        )

    return {
        "category": CLASSES[int(index)],
        "confidence": float(confidence),
    }


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m "
            "product_image_query.predict_product "
            "<image_path>"
        )
        raise SystemExit(1)

    print(
        predict(sys.argv[1])
    )