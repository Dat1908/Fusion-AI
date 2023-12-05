import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from timeit import default_timer

device = 'cpu'
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

def load_model(weight_path="D:\modelvytec\shape_face.pth"):
    weights = torch.load(weight_path)
    model = torchvision.models.efficientnet_b4()
    model.classifier = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(weights)
    return model.cpu()

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names,
    image_path: str,
    image_size = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device):

    img = Image.open(image_path)
    img = img.convert("RGB")
    if transform is not None:
        image_transform = transform
    else:
        image_transform = T.Compose([T.Resize(image_size),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    model.eval()
    start_time = default_timer()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    end_time = default_timer()
    if target_image_pred_probs.max() > 0.8:
        print(f"Predict time: {end_time-start_time:.2f} seconds")
        plt.figure()
        plt.imshow(img)
        plt.title(
            f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
        )
        plt.axis(False)
        plt.show()
        print(class_names[target_image_pred_label] + 'Probability: ' + target_image_pred_probs.max())
    else:
        print('Please upload other image.')
    
def main():
    img_path = "D:\skincolor.jpg"
    model = load_model()
    print('Load success')
    pred_and_plot_image(model, class_names=classes, image_path=img_path)
    
if __name__ == '__main__':
    main()