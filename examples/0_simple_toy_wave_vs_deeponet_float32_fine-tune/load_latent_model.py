from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

train_data = np.load('./data/train.npz')['data']  # t,x,u
input_t = train_data[:,0]


latent_model_100 = keras.models.load_model('./latent_model_100')
latent_rep_100 = latent_model_100.predict(input_t)

latent_model_500 = keras.models.load_model('./latent_model_500')
latent_rep_500 = latent_model_500.predict(input_t)

plt.scatter(latent_rep_100, latent_rep_500)
plt.axis('equal')
plt.show()