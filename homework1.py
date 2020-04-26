import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve
from PIL import Image, ImageFilter 

#reading image
img = mpimg.imread('/home/onur/Desktop/YL/Computer_Vision/HW1/SunnyLake.bmp')

#taking average of channels of image to convert to gray scale
gray = np.dot(img[...,:3], [0.33, 0.33, 0.33])

#plotting gray scale image
plt.imshow(gray, cmap = plt.get_cmap('gray'))

#flatting numpy array to show in histogram
flat = gray.reshape(np.prod(gray.shape[:2]),-1)
flat = np.int64(flat)

#plotting histogram
plt.hist(flat, bins = 256)
plt.show()



#thresholding with different values
threshold1 = (gray > 32) * 256
threshold2 = (gray > 64) * 256
threshold3 = (gray > 128) * 256
threshold4 = (gray > 156) * 256

#choosing one of them and plotting
plt.imshow(threshold4, cmap = plt.get_cmap('gray'))


#plotting all of images with different threshold values
plt.figure()

f, axarr = plt.subplots(1,4) 

axarr[0].imshow(threshold1, cmap = plt.get_cmap('gray'))
axarr[1].imshow(threshold2, cmap = plt.get_cmap('gray'))
axarr[2].imshow(threshold3, cmap = plt.get_cmap('gray'))
axarr[3].imshow(threshold4, cmap = plt.get_cmap('gray'))



mean = 0
var = 1
sigma = var ** 0.5
gaussian1 = np.random.normal(mean, sigma, (300, 400)) 
noisy_image1 = np.zeros(img.shape, np.int64)


noisy_image1[:, :, 0] = img[:, :, 0] + gaussian1
noisy_image1[:, :, 1] = img[:, :, 1] + gaussian1
noisy_image1[:, :, 2] = img[:, :, 2] + gaussian1


var = 5
sigma = var ** 0.5
gaussian2 = np.random.normal(mean, sigma, (300, 400)) 
noisy_image2 = np.zeros(img.shape, np.int64)


noisy_image2[:, :, 0] = img[:, :, 0] + gaussian2
noisy_image2[:, :, 1] = img[:, :, 1] + gaussian2
noisy_image2[:, :, 2] = img[:, :, 2] + gaussian2

var = 10
sigma = var ** 0.5
gaussian3 = np.random.normal(mean, sigma, (300, 400)) 
noisy_image3 = np.zeros(img.shape, np.int64)


noisy_image3[:, :, 0] = img[:, :, 0] + gaussian3
noisy_image3[:, :, 1] = img[:, :, 1] + gaussian3
noisy_image3[:, :, 2] = img[:, :, 2] + gaussian3

var = 20
sigma = var ** 0.5
gaussian4 = np.random.normal(mean, sigma, (300, 400)) 
noisy_image4 = np.zeros(img.shape, np.int64)


noisy_image4[:, :, 0] = img[:, :, 0] + gaussian4
noisy_image4[:, :, 1] = img[:, :, 1] + gaussian4
noisy_image4[:, :, 2] = img[:, :, 2] + gaussian4

plt.imshow(noisy_image1)
plt.imshow(noisy_image2)
plt.imshow(noisy_image3)
plt.imshow(noisy_image4)
plt.imshow(img)




#taking average of channels of image to convert to gray scale
I1 = np.dot(noisy_image1[...,:3], [0.33, 0.33, 0.33])

#plotting gray scale image
plt.imshow(I1, cmap = plt.get_cmap('gray'))

#taking average of channels of image to convert to gray scale
I5 = np.dot(noisy_image2[...,:3], [0.33, 0.33, 0.33])

#plotting gray scale image
plt.imshow(I5, cmap = plt.get_cmap('gray'))

#taking average of channels of image to convert to gray scale
I10 = np.dot(noisy_image3[...,:3], [0.33, 0.33, 0.33])

#plotting gray scale image
plt.imshow(I10, cmap = plt.get_cmap('gray'))

#taking average of channels of image to convert to gray scale
I20 = np.dot(noisy_image4[...,:3], [0.33, 0.33, 0.33])

#plotting gray scale image
plt.imshow(I20, cmap = plt.get_cmap('gray'))





#filtering operation with convolution

def filtering(image, kernel):
    res = convolve(image[:,:], kernel)
    return plt.imshow(res, cmap = plt.get_cmap('gray'))

#different kernels for different kind of filters
low_pass_kernel = np.array([ [1, 1, 1], [1, 1, 1], [1, 1, 1] ])
gaussian_kernel = np.array([ [1, 2, 1], [2, 4, 2], [1, 2, 1] ])
high_pass_kernel = np.array([ [-1, -1, -1], [-1, 8, -1], [-1, -1, -1] ])
high_boost_kernel = np.array([ [-1, -1, -1], [-1, 12, -1], [-1, -1, -1] ])

filtering(I1, low_pass_kernel)
filtering(I5, low_pass_kernel)
filtering(I10, low_pass_kernel)
filtering(I20, low_pass_kernel)

filtering(I1, gaussian_kernel)
filtering(I5, gaussian_kernel)
filtering(I10, gaussian_kernel)
filtering(I20, gaussian_kernel)

filtering(I1, high_pass_kernel)
filtering(I5, high_pass_kernel)
filtering(I10, high_pass_kernel)
filtering(I20, high_pass_kernel)

filtering(I1, high_boost_kernel)
filtering(I5, high_boost_kernel)
filtering(I10, high_boost_kernel)
filtering(I20, high_boost_kernel)



 
     
#creating a image object  
im1 = Image.open(r"/home/onur/Desktop/YL/Computer_Vision/HW1/Figure_1.png")  
     
#applying the median filter  
im2 = im1.filter(ImageFilter.MedianFilter(size = 3))  
     
im2.show()  
