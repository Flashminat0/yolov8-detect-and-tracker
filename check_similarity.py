import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights


def image_similarity(img1_path, img2_path):
    # Set up the model and transformations
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.avgpool.register_forward_hook(get_activation("avgpool"))

    # Extract features from an image
    def extract_features(img_path):
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        with torch.no_grad():
            out = model(img[None, ...])
        return activation["avgpool"].numpy().squeeze()

    # Extract features from both images
    vec1 = extract_features(img1_path)
    vec2 = extract_features(img2_path)

    # Compute cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return similarity

# Example usage:
# similarity_score = image_similarity("task/0.jpg", "task/it20014940.jpg")
# print("Similarity Score:", similarity_score)
