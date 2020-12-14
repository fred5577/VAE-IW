from PIL import Image
import PIL, sys
import glob
import os

domain = "BeamRider"

data_path = "../Pictures/{}-v4".format(domain)

data_path = os.path.join(data_path)

image_paths = glob.glob(data_path + '/*.png')

if not os.path.exists("../Pictures/{}-Crop-v4".format(domain)):
    os.makedirs("../Pictures/{}-Crop-v4".format(domain))

imageObject = Image.open(image_paths[0])
width, height = imageObject.size 
top = 40
bottom = height
left = 0
right = width

for i in range(len(image_paths)):
    imageObject = Image.open(image_paths[i])
    imcrop = imageObject.crop((left,top,right,bottom))
    imcrop.save("../Pictures/{}-Crop-v4/image{}.png".format(domain,i))
    if i % 1000 == 0:
        print("Number of images processed: ", i)
