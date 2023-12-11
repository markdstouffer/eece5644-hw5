import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score

img = io.imread('image.jpeg')

rows, cols, _ = img.shape

# image pre-processing
features = np.zeros((rows * cols, 5))
for i in range(rows):
    for j in range(cols):
        features[i * cols + j, :] = [i, j, img[i, j, 0], img[i, j, 1], img[i, j, 2]]

features = preprocessing.minmax_scale(features)

maxNumComponents = 10
bestNumComponents = 0
bestAvgLogLikelihood = -np.inf

# cross-validation
for k in range(1, maxNumComponents + 1):
    print(k)
    gm = GaussianMixture(n_components=k, random_state=0)
    avgLogLikelihood = np.mean(cross_val_score(gm, features, cv=10))
    if avgLogLikelihood > bestAvgLogLikelihood:
        bestAvgLogLikelihood = avgLogLikelihood
        bestNumComponents = k

# fit GMM with best number of components
gm = GaussianMixture(n_components=bestNumComponents, random_state=0).fit(features)

# predictions
labels = gm.predict(features)

labelsImg = labels.reshape((rows, cols))

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(labelsImg, cmap='viridis')
ax[1].set_title('Segmentation Image')
ax[1].axis('off')
plt.show()
