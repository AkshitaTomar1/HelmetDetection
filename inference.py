import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage

def get_prediction(model, image, threshold=0.8, device=torch.device("cpu")):
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        prediction = model([image])[0]

    keep = prediction['scores'] > threshold

    boxes = prediction['boxes'][keep]
    labels = prediction['labels'][keep]
    scores = prediction['scores'][keep]

    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        if width * height > 500:
            filtered_boxes.append(box)
            filtered_labels.append(labels[i])
            filtered_scores.append(scores[i])

    return {
        'boxes': torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4)),
        'labels': torch.stack(filtered_labels) if filtered_labels else torch.empty((0,), dtype=torch.int64),
        'scores': torch.stack(filtered_scores) if filtered_scores else torch.empty((0,))
    }

def draw_boxes(image, prediction, class_names):
    if image.max() <= 1.0:
        image = (image * 255).byte()
    else:
        image = image.byte()

    labels = []
    for lbl in prediction['labels']:
        label_idx = lbl.item()
        if label_idx >= len(class_names):
            labels.append("Unknown")
        else:
            labels.append(class_names[label_idx])

    img_with_boxes = draw_bounding_boxes(
        image,
        prediction['boxes'],
        labels=labels,
        width=3,
        colors="red"
    )

    to_pil = ToPILImage()
    return to_pil(img_with_boxes)
