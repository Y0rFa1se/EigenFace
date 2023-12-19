# import cv2
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

    return comp.reshape(original_shape)


from sklearn.datasets import fetch_lfw_people

lfw_people = fetch_lfw_people(min_faces_per_person=70).images
sample = lfw_people[10]
number_of_basis = int(input())

plt.imsave("photos/original.png", sample, cmap="gray")
plt.imsave("photos/decomposed.png", decompose(sample, number_of_basis), cmap="gray")