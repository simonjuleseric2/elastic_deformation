import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from moviepy.editor import ImageSequenceClip
import cv2

def elastic_transf(img, alpha, sigma):

    random_state = np.random.RandomState(None)
    shape = img.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    if len(shape)>2:
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    else:
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    image = map_coordinates(img, indices, order=1, mode='reflect')

    return image.reshape(img.shape)



ims = []
sigma=10

imgx = cv2.imread('photo.jpg')
imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
imgx=cv2.resize(imgx, (300, 300), interpolation = cv2.INTER_CUBIC)

for i in range(1, 3000, 50):
    imgx=elastic_transf(imgx, i, sigma)
    ims.append(imgx)


clip = ImageSequenceClip(ims, fps=25)
#clip.write_gif('simulation.gif', fps=20)
clip.write_videofile('elast_transform.mp4', fps=25)
