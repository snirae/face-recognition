import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import tensor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from data import LFWDataset
from convolutional_encoder import ConvAutoEncoder
from resnet_encoder import ResNetEncoder
from clip_encoder import CLIPEncoder
from face_resnet import FaceResNet
from similarity_net import SimilarityNet
from facenet import FaceNet
from siamese_net import SiameseNet

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = '../LFW/lfw_funneled'
lfw_dataset = LFWDataset(data_path, transform=transform)
train_size = int(0.9 * len(lfw_dataset))
test_size = len(lfw_dataset) - train_size
train_dataset, test_dataset = random_split(lfw_dataset, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def crop_face(image):
    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # crop the first face recognized
    try:
        x1, y1, x2, y2 = faces[0]
        return image[y1:y1 + y2, x1:x1 + x2]
    except Exception as e:
        print(e.with_traceback())
        return None


def crop_resize(image, x=300, y=300):
    res = crop_face(image)
    return cv2.resize(res, (x, y))


siamese = SiameseNet()
siamese.load_state_dict(torch.load('../trained-models/SiameseNet.pt',
                                   map_location=torch.device('cpu')))


def encode(encoder, image):
    image = tensor(image).float()
    image = image.reshape((3, 224, 224))
    image = image.unsqueeze(0)
    with torch.no_grad():
        return encoder.encode(image)


def encode_image_face(encoder, image):
    image = crop_resize(image, 224, 224)
    return encode(encoder, image)


################################################################################
model = siamese
model.eval()

# jm = encode_image_face(model, read_image('faces/jason-momoa/jm.jpeg'))
# jm2 = encode_image_face(model, read_image('faces/jason-momoa/2.jpeg'))
# jm3 = encode_image_face(model, read_image('faces/jason-momoa/3.jpeg'))
#
# a = encode_image_face(model, read_image('faces/adele/adele.jpeg'))
# a2 = encode_image_face(model, read_image('faces/adele/1.jpeg'))
# a3 = encode_image_face(model, read_image('faces/adele/2.jpeg'))
#
# print(f'jm & jm2: {cosine_similarity(jm, jm2), euclidean_distances(jm, jm2)}')
# print(f'jm & jm3: {cosine_similarity(jm, jm3), euclidean_distances(jm, jm3)}')
# print(f'jm2 & jm3: {cosine_similarity(jm2, jm3), euclidean_distances(jm2, jm3)}')
#
# print('\n')
#
# print(f'a & a2: {cosine_similarity(a, a2), euclidean_distances(a, a2)}')
# print(f'a & a3: {cosine_similarity(a, a3), euclidean_distances(a, a3)}')
# print(f'a2 & a3: {cosine_similarity(a2, a3), euclidean_distances(a2, a3)}')
#
# print('\n')
#
# print(f'jm & a: {cosine_similarity(jm, a), euclidean_distances(jm, a)}')
# print(f'jm2 & a: {cosine_similarity(a, jm2), euclidean_distances(a, jm2)}')
# print(f'jm3 & a: {cosine_similarity(jm3, a), euclidean_distances(jm3, a)}')
#
# print('\n')
#
# print(f'jm & a2: {cosine_similarity(jm, a2), euclidean_distances(jm, a2)}')
# print(f'jm2 & a2: {cosine_similarity(a2, jm2), euclidean_distances(a2, jm2)}')
# print(f'jm3 & a2: {cosine_similarity(jm3, a2), euclidean_distances(jm3, a2)}')
#
# print('\n')
#
# print(f'jm & a3: {cosine_similarity(jm, a3), euclidean_distances(jm, a3)}')
# print(f'jm2 & a3: {cosine_similarity(a3, jm2), euclidean_distances(a3, jm2)}')
# print(f'jm3 & a3: {cosine_similarity(jm3, a3), euclidean_distances(jm3, a3)}')

with torch.no_grad():
    res_dist = []
    res_sim = []

    y_pred = []
    y_test = []

    threshold = lambda x: 1 if x <= 0.5 else 0
    for idx, batch in enumerate(test_loader):
        # if idx % 5 == 0:
        #     print(f'{idx}/{len(test_loader)}')
        anchor, positive, negative = batch
        # a = model(anchor)
        # p = model(positive)
        # n = model(negative)

        pos = model(anchor, positive).item()
        neg = model(anchor, negative).item()

        y_test += [1, 0]
        y_pred += [threshold(pos), threshold(neg)]

        print([pos, neg])

print()


def stats(res):
    arr = np.array(res)
    arr = np.concatenate((arr[:, 0][arr[:, 0] <= 5], arr[:, 1][arr[:, 1] <= 5]))
    print('medians:', np.median(arr[:, 0]), np.median(arr[:, 1]))
    print('means:', arr[:, 0].mean(), arr[:, 1].mean())
    print('max:', arr[:, 0].max(), arr[:, 1].max())
    print('min:', arr[:, 0].min(), arr[:, 1].min(), end='\n\n')


stats(y_pred)
