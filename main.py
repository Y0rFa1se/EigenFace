import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

eigenfaces = np.load("basis.npy")
with open("vars.pkl", "rb") as f:
    M, scaler = pickle.load(f)

def decompose(img, number_of_basis = 50):
    original_shape = img.shape
    img = img.reshape(-1)
    img -= M
    img /= scaler
    coeffecients = np.array([])
    for ef in eigenfaces[:number_of_basis]:
        coeffecients = np.append(coeffecients, np.dot(img, ef))
        
    comp = np.zeros(img.shape)
    for i in range(number_of_basis):
        comp += coeffecients[i] * eigenfaces[i]
        
    comp *= scaler
    comp += M

    return comp.reshape(original_shape), coeffecients

img_path = [
    "samples/11.jpg",
    "samples/12.jpg",
    "samples/21.jpeg",
    "samples/22.jpeg"
]
images = []
for path in img_path:
    img = cv2.imread(path)
    img = cv2.resize(img, (47, 62))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img)

images = np.array(images)
images = images.astype(np.float64)
images /= 255/2
images -= 1

# 1
number_of_basis = [1, 5, 10, 50, 100, 300, 500, 1000, 2000]
for n in number_of_basis:
    comp, _ = decompose(images[0], n)
    plt.imsave(f"results/1_{n}.jpg", comp, cmap="gray")

# 2
coeffitients = []
for img in images:
    _, coef = decompose(img)
    coeffitients.append(coef)
    
for i in range(images.shape[0]):
    for j in range(images.shape[0]):
        if i == j:
            continue
        
        cos_sim = np.dot(coeffitients[i], coeffitients[j])
        cos_sim /= (np.linalg.norm(coeffitients[i]) * np.linalg.norm(coeffitients[j]))
        print(cos_sim)
        
    print()