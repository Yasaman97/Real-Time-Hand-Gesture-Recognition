#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the useful libraries
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import os
import sys
import numpy as np
from PIL import Image
import io
import cv2 # OpenCV library
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import figure
from google.colab.patches import cv2_imshow
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils as np_utils


# In[ ]:


#connecting to Google Drive
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Create a real time video stream
def VideoCapture():
  js = Javascript('''
    async function create(){
      div = document.createElement('div');
      document.body.appendChild(div);

      video = document.createElement('video');
      video.setAttribute('playsinline', '');

      div.appendChild(video);
      stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "environment"}});
      video.srcObject = stream;

      await video.play();

      canvas =  document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);

      div_out = document.createElement('div');
      document.body.appendChild(div_out);
      img = document.createElement('img');
      div_out.appendChild(img);
    }

    async function capture(){
        return await new Promise(function(resolve, reject){
            pendingResolve = resolve;
            canvas.getContext('2d').drawImage(video, 0, 0);
            result = canvas.toDataURL('image/jpeg', 0.20);

            pendingResolve(result);
        })
    }

    function showimg(imgb64){
        img.src = "data:image/jpg;base64," + imgb64;
    }

  ''')
  display(js)


# In[ ]:


VideoCapture()         #start streaming the video
eval_js('create()')
while True:

  e1 = cv2.getTickCount()                         #to start calculating the clock cycles                          

  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)     #getting the grayscale of the image for faster computation (only one color channel instead of 3)     
  face = detect(gray, cascade = Cascade_face)     #detect the face in the grayscale image    
  draw_rects(im, face, (255, 0, 0))               #draw a recangle around the face    

  e2 = cv2.getTickCount()                          #to stop calculating the clock cycles     
  face_time = (e2-e1)/cv2.getTickFrequency()       #the amount time it takes, in seconds, to detect and draw the face the face                             
  print("the number of seconds to detect the face is :", face_time, "seconds")


  eval_js('showimg("{}")'.format(image2byte(im)))


# In[ ]:


def byte2image(byte):
  """Given the bytes of an image, this function will return an array representing the image"""  
  jpeg = b64decode(byte.split(',')[1])
  im = Image.open(io.BytesIO(jpeg))
  return np.array(im)


# In[ ]:


def image2byte(image):
  """Given the array representing an image, this function will return the bytes of the image"""  
  image = Image.fromarray(image)
  buffer = io.BytesIO()
  image.save(buffer, 'jpeg')
  buffer.seek(0)
  x = b64encode(buffer.read()).decode('utf-8')
  return x


# In[ ]:


def detect(img, cascade):
  """Given the image and a Haar cascade, this function detects the required object (the face in our case), 
  and return the top left and bottom right corners of the detected object,
  or an empty array if the object is not found. """                                                                                
  rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
  if len(rects) == 0: #return the empty list
    return []  
  rects[:,2:] += rects[:,:2] #return the left and bottom right corners of the detected object (x,y,w,h)

  return rects


# In[ ]:


def draw_rects(img, rects, color, size=2):
  """ Given the image, detected face, a color of our choosing and the thickness of the line, 
  this function draws a rectangle around the detected face. """  
  for x1, y1, x2, y2 in rects:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, size)


# In[ ]:


def window_detect(rects,margin,im):
  """Given the detected face, a margin (in pixels), and the image, 
  this function returns a bigger rectangle around the detected face, 
  with respect to the boundaries of the image, to make sure the bigger rectangle 
  is never out of bounds of the original image. """  
  boundary = im.shape
  x1 = max(0,rects[0]-margin) #if x-margin < 0, then choose zero
  y1 = max(0,rects[1]-margin) #if y-margin < 0, then choose zero
  x2 = min(rects[2]+margin,boundary[1]) #if x+margin > size of the image on the x axis , then choose the boundary
  y2 = min(rects[3]+margin,boundary[0]) #if y+margin > size of the image on the y axis , then choose the boundary

  window = [x1,y1,x2,y2]
  return window


# In[ ]:


def relativeAbsolute(window,face):
  """Given the detected face and the window, 
  this function finds and returns the absolute position of the bigger rectangle, 
  since the position of the window is relative to the face. """  
  absoluteX = window[0]+face[0]
  absoluteX2 = window[0]+face[2]
  absoluteY = window[1]+face[1]
  absoluteY2 = window[1]+face[3]
  return [[absoluteX, absoluteY, absoluteX2, absoluteY2]] #return it the same way as for the face detect


# In[ ]:


def restrict(rect,margin,im):
  """Given the detected face, a margin and the original image, 
  this function restricts the rectangle just around the face, so that the color of the hair is not detected. """  
  boundary = im.shape
  x1=min(640,rect[0]+margin)
  y1=min(480,rect[1]+margin)
  x2=max(rect[2]-margin,x1)
  y2=max(rect[3]-margin,y1)
  return [x1,y1,x2,y2]


# In[ ]:


def Picture_save(storage): 
  """ after all the images for a letter are stored in all_images, 
  they are passed to this function, changing the path by hand, in order to store all of them in a folder on the drive"""  
  # Sets the paths 
  path = "/content/drive/MyDrive/MLCV/Data/Dataset_3/" 
  Letter = input("Letter")
  #ID = range(1,100) # 
  Size= [16, 224] # To create an image in both 16 x16 and 224 x224
  for ID,image in enumerate(storage):
    for b in Size:
      file_name= Letter.upper()+"_" + str(ID) +"_"+  str(b)
      ### Changing the size of the picture
      im_size = cv2.resize(image, (b,b) , interpolation = cv2.INTER_AREA)
      cv2.imwrite(path + file_name + ".jpg", im_size)
      print(path+file_name)  


# In[ ]:


def Resize_N_Write(id, letter, dataset='dataset.txt'):
  """ After storing all the images, use this function to change the size of 16x16 images to 1x256, append them, 
  and save it in a text file. """  

  path = "path/Dataset/" 
  img = cv2.imread(path+letter+"_" + str(id) + "_16.jpg")             #reads the image of a certain letter and id number of the size 16x16
  res = cv2.resize(img, dsize=(1,256), interpolation=cv2.INTER_CUBIC) #resizing the image to 1x256
  flat_res = [str(a[0][0]) for a in res]                              #turning the res file values to strings
   
  is_empty = False
  try:
    with  open("path/Dataset/" + dataset, "r") as fileobj:   #open the dataset in read mode
      if len(fileobj.readlines())==0:                                                      #check if the dataset is empty
        is_empty = True
  except:
    is_empty = True                                                                        #if the dataset does not exist, set the empty variable to true
  with open("path/Dataset/" + dataset, "a+") as fileobj:     #open the dataset in append mode
    L_list=[letter, *flat_res]                                                             #create a list with the letter and all the values flat_res
    str_list = str(L_list)
    str_list = str_list.replace('\'', '')[1:-1]                                            #removing punctuation
    if is_empty:
      fileobj.write(str_list)                                                              #if the file doesn't exist, crreates it and if it does, appends the string list to it 
    else:
      fileobj.write("\n" + str_list)                                                       #if the file doesn't exist, crreates it and if it does, appends the string list to it


# In[ ]:


def load_dataset(dataset_file_path):
  """ Taking the path to the dataset that was a text file, 
  writes the first column of each row into lettets, 
  and the rest (256 numbers) into samples"""
  a = np.loadtxt(dataset_file_path, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
  samples, letters = a[:,1:], a[:,0]
  return samples, letters


# In[ ]:


#The Haar cascade is used to detect the face
Cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[ ]:


# Simple detection of the face
VideoCapture()          #start streaming the video
eval_js('create()')

while True:

  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)           #getting the grayscale of the image for faster computation (only one color channel instead of 3)                                                                   
  face = detect(gray, cascade = Cascade_face)          #detect the face in the grayscale image                        
  draw_rects(im, face, (255, 0, 0))                     #draw a recangle around the face                       
  
  if len(face)>0:                                      #if the first face is detected, start running the inner while loop
    isDetected = True
  else:                                                 #else, keep looking for the face in the outer while loop
    isDetected = False

  eval_js('showimg("{}")'.format(image2byte(im)))

  while isDetected:

    for rect in face:                                    #this is because face is inside []
      e1 = cv2.getTickCount()                            #to start calculating the clock cycles

      byte = eval_js('capture()')
      im = byte2image(byte)
      gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)                                                   
                                                                                
      window = window_detect(rect,30,im)                 #compute a bigger rectangle around the face
      draw_rects(im, [window], (0,255,0))

      roi_gray = gray[window[1]:window[3], window[0]:window[2]]                             
      roi_color = im[window[1]:window[3], window[0]:window[2]]                              

                                                          
                                                          
      new_face = detect(roi_gray, cascade = Cascade_face)   #detect the new face inside the bigger rectangle around the first detected face

      if len(new_face)>0:                                   #if the face is found inside the bigger rectangle, find the absolute position of the rectangle around it
        rects = relativeAbsolute(window,new_face[0])
        draw_rects(im, rects, (255, 0, 0))
      else:                                                 #else, the face is outside the bigger rectangle, we should get out of the inner loop and detect a first face again
        isDetected = False                        
                  
      e2 = cv2.getTickCount()                            #to stop calculating the clock cycles                        
      face_time = (e2-e1)/cv2.getTickFrequency()         #the amount time it takes, in seconds, to detect and draw the face the face                                                       
      print("the number of clock-cycles to detect the face is :", face_time, "seconds")

      eval_js('showimg("{}")'.format(image2byte(im)))


# Instead of searching through the whole image, find the face at first, then draw a slightly bigger rectangle around it, as the face usually will not move from this area a lot, and then go through this slightly bigger rectangle to find the face.

# In[ ]:


# Improved face detection by restricting the search area
VideoCapture()          #start streaming the video
eval_js('create()')

while True:

  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)           #getting the grayscale of the image for faster computation (only one color channel instead of 3)                                                                   
  face = detect(gray, cascade = Cascade_face)          #detect the face in the grayscale image                        
  draw_rects(im, face, (255, 0, 0))                     #draw a recangle around the face                       
  
  if len(face)>0:                                      #if the first face is detected, start running the inner while loop
    isDetected = True
  else:                                                 #else, keep looking for the face in the outer while loop
    isDetected = False

  eval_js('showimg("{}")'.format(image2byte(im)))

  while isDetected:

    for rect in face:                                    #this is because face is inside []
      e1 = cv2.getTickCount()                            #to start calculating the clock cycles

      byte = eval_js('capture()')
      im = byte2image(byte)
      gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)                                                   
                                                                                
      window = window_detect(rect,30,im)                 #compute a bigger rectangle around the face
      draw_rects(im, [window], (0,255,0))

      roi_gray = gray[window[1]:window[3], window[0]:window[2]]                             
      roi_color = im[window[1]:window[3], window[0]:window[2]]                              

                                                          
                                                          
      new_face = detect(roi_gray, cascade = Cascade_face)   #detect the new face inside the bigger rectangle around the first detected face

      if len(new_face)>0:                                   #if the face is found inside the bigger rectangle, find the absolute position of the rectangle around it
        rects = relativeAbsolute(window,new_face[0])
        draw_rects(im, rects, (255, 0, 0))
      else:                                                 #else, the face is outside the bigger rectangle, we should get out of the inner loop and detect a first face again
        isDetected = False                        
                  
      e2 = cv2.getTickCount()                            #to stop calculating the clock cycles                        
      face_time = (e2-e1)/cv2.getTickFrequency()         #the amount time it takes, in seconds, to detect and draw the face the face                                                       
      print("the number of clock-cycles to detect the face is :", face_time, "seconds")

      eval_js('showimg("{}")'.format(image2byte(im)))


# Using the HVS channels of the detected face, calculate the histogram. Then, calculate the backprojection of this histogram over the image, which gives a probability of the face being found in other parts of the image, and use the camshift algorithm to track the face.

# In[ ]:


# Tracking the face using camshift algorithm

VideoCapture()       #start streaming the video
eval_js('create()')

first_detect = False
while not first_detect:
  window = []
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)               #getting the grayscale of the image for faster computation (only one color channel instead of 3)
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV)                #getting the hsv channels of the image for better detection of the skin color for histogram
  face = detect(gray, cascade = Cascade_face)               #detect the face in the grayscale image, since it's faster
  draw_rects(im, face, (255, 0, 0))                 
  camdetect = False
  if len(face)==0:                                          #if the first face is not detected, keep the flag the same
    first_detect = False
  else:                                                     #else, change it to True
    first_detect = True  
   
  if len(window) == 0 and first_detect:                     #if the first face is found and we don't have the bigger windows, we can go to inner loop
    window.append(window_detect(face[0],30,im))
    camdetect = True
    
    gray_roi = gray[window[0][1]:window[0][3], window[0][0]:window[0][2]]
    hsv_roi = hsv[window[0][1]:window[0][3], window[0][0]:window[0][2]]
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))    #thresholding hsv of interested region to lower and upper bounds
    hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])                               #calculating the histogram of hsv over the bigger window, using the mask
    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)                                      #normalizing the histogram between 0 and 255
    

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )     # tells the algorithm that we want to terminate either after some number of iterations or when the convergence metric reaches some small value (respectively).

while camdetect:

   byte = eval_js('capture()')
   im = byte2image(byte)
   gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)          
   hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

   track_window = (window[0][0], window[0][1], window[0][2], window[0][3])   #getting the coordinates of the bigger window
   dst = cv2.calcBackProject([hsv], [0], hist, [0, 256], 1)                  #using the hsv image to calculate the backprojection
   
   ret, track_window = cv2.CamShift(dst, track_window, term_crit)            #applying camshift to get the new location
   
   im[:] = dst[...,np.newaxis]                                               #adding the probability matrix from backprojection to the image
   

   draw_rects(im, face, (255, 0, 0))
   draw_rects(im, window, (0, 255, 0))
   eval_js('showimg("{}")'.format(image2byte(im)))


# Set the probability of the face, given by the backprojection, to zero in order to cover the face (turn the pixels in the place the face is found to zero), so that the algorithm can find our hands, since our hands and face are the same color and have quite similar histogram distributions.

# In[ ]:


# Detecting the hand by setting the probability of the face to zero

VideoCapture()         #start streaming the video
eval_js('create()')

first_detect = False
while not first_detect:
  window = []
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)                   #getting the grayscale of the image for faster computation (only one color channel instead of 3)     
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV)                    #getting the hsv channels of the image for better detection of the skin color for histogram
  face = detect(gray, cascade = Cascade_face)                   #detect the face in the grayscale image, since it's faster
  draw_rects(im, face, (255, 0, 0))                 
  camdetect = False

  if len(face)==0:                                              #if the first face is not detected, keep the flag the same
    first_detect = False
  else:
    first_detect = True                                         #else, change it to True
   
  if len(window) == 0 and first_detect:                         #if the first face is found and we don't have the bigger windows, we can go to inner loop
    window.append(window_detect(face[0],30,im))
    camdetect = True
    
    gray_roi = gray[window[0][1]:window[0][3], window[0][0]:window[0][2]]
    hsv_roi = hsv[window[0][1]:window[0][3], window[0][0]:window[0][2]]
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))       #thresholding hsv of interested region to lower and upper bounds
    hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])                                  #calculating the histogram of hsv over the bigger window, using the mask
    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)                                         #normalizing the histogram between 0 and 255
    

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )     # tells the algorithm that we want to terminate either after some number of iterations or when the convergence metric reaches some small value (respectively).

while camdetect:

   byte = eval_js('capture()')
   im = byte2image(byte)
   gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)          
   hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

   track_window = (window[0][0], window[0][1], window[0][2], window[0][3])                #getting the coordinates of the bigger window
   dst = cv2.calcBackProject([hsv], [0], hist, [0, 256], 1)                               #using the hsv image to calculate the backprojection
   
   for i in range(window[0][0], window[0][2]):                                            #for all he pixels inside the bigger window around the face
     for j in range(window[0][1], window[0][3]):
       dst[j,i]=0                                                                         #set the probability to zero

   ret, track_window = cv2.CamShift(dst, track_window, term_crit)                         #applying camshift to get the new location
   
   im[:] = dst[...,np.newaxis]                                                            #adding the probability matrix from backprojection to the image
   

   draw_rects(im, face, (255, 0, 0))
   draw_rects(im, window, (0, 255, 0))

   roi=cv2.rectangle(im, track_window , (255,255,255), 1)

   eval_js('showimg("{}")'.format(image2byte(im)))


# In order to build a dataset, take pictures of the hands, while performing sign language for a couple of letters, and store the images in 2 sizes: 16x16 and 224x224

# In[ ]:


all_images = []           # in order to save all the images, from different runs of the next cell


# In[ ]:


VideoCapture()             #start streaming the video
eval_js('create()')

Picture_count=int(input("How many pictures do you want to take? "))                     #asks the user for the number of images they want to take
Time=int(input("How many seconds do you want to wait between each photo? "))            #asks the user for the amount of time they need between images
n_images=0                                                                              #counter for images approved by the user
saved_images =[]                                                                        #temporary storage for approved images


first_detect=False


while not first_detect:
  window=[] 
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)                                          #getting the grayscale of the image for faster computation (only one color channel instead of 3)
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV)                                           #getting the hsv channels of the image for better detection of the skin color for histogram
  face = detect(gray, cascade = Cascade_face)                                           #detect the face in the grayscale image, since it's faster

  if len(face)==0:                                                                     #if the first face is not detected, keep the flag the same
    first_detect=False
  else:                                                                                #else, change it to True
    first_detect=True
    
  if first_detect:                                                                     #if the first face is found, we can go to inner loop
    restricted_face = restrict(face[0],30,im)                                                 #the window that is restricted to the face
    hsv_roi=hsv[restricted_face[1]:restricted_face[3], restricted_face[0]:restricted_face[2]] #restricting the hsv to the region of interest
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))       #thresholding hsv of interested region to lower and upper bounds
    window.append(window_detect(face[0],30,im))                                               
  
    camdetect=True                                                                            #so we can use the camshift 
    hist=cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])                                   #calculating the histogram of hsv over the bigger window, using the mask
    hist=cv2.normalize(hist, hist,  alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)             #normalizing the histogram between 0 and 255

    track_window = (window[0][0], window[0][1], window[0][2], window[0][3])                   #getting the coordinates of the bigger window

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 

while camdetect:
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) 
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
  mask2 = cv2.inRange(hsv, np.array((0,64,32)), np.array((180,200,200))) 
  dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)                                    #using the hsv image to calculate the backprojection

  for i in range(window[0][0], window[0][2]):                                                 #for all he pixels inside the bigger window around the face
    for j in range(window[0][1], window[0][3]): 
      dst[j,i]=0                                                                              #set the probability to zero

  dst &= mask2

  ret, track_window = cv2.CamShift(dst, track_window, term_crit)                              #applying camshift to get the new location                 
  
  im[:] = dst[...,np.newaxis]                                                                 #adding the probability matrix from backprojection to the image

  draw_rects(im, [restricted_face], (255, 0, 255))
 
  eval_js('showimg("{}")'.format(image2byte(im)))
  if n_images==Picture_count:                                                                 #if the number of saved images is the same as the number the user requested, stop the algorithn
    break
  else:                                                                                       #else, wait for the amount of time requested by the user to take the next image
    time.sleep(Time)
  cropped_image = im[int(track_window[1]):int(track_window[1]+track_window[3]),               #store the image of the hand
                          int(track_window[0]):int(track_window[0]+track_window[2])]

  cv2_imshow(cropped_image)                                                                   #show the image to the user
  
  save_image=input("Do you want to save the image Y or N: ")                                  #ask if the image should be saved or not
  
  if str.lower(save_image) == 'y':                                                            #if the answer was yes, 

    saved_images.append(cropped_image)                                                        #add the image of hand to saved images
    print("Picture",n_images+1,"was stored")                                                  #print the number of saved images
    n_images+=1
    continue
  else:
    continue 


# In[ ]:


#Storing the approved images from each run in all_images
print("Number of images just saved: ",len(image_store))
all_images.extend(image_store)
print("Total number of images just saved: ",len(all_images))


# In[ ]:


#Call the function to write all the images of a letter in the desired destination
Picture_save(all_images)


# Resize the images in size 16x16 to an array of size 1x256, and save it as a text file in the same path as the images. One last step is to use the code provided to us to shuffle the lines of this text file.

# In[ ]:


#creating a list of the letters in my datasets, and the 3 text files for each dataset
letters = ["A","B","S"]
Dataset = "Dataset.txt"


# In[ ]:


# there are 100 images of each letter in the folder
for i in range(100):
  for letter in letters:
    Resize_N_Write(i, letter, Dataset, '1')


# In[ ]:


lines = open('path/Dataset.txt').readlines()
random.shuffle(lines)
open('path/Dataset.txt', 'w').writelines(lines)


# Using the Keras library to train a multi layer perceptron on each of the datasets.

# In[ ]:


dataset_file_path = "path/Dataset.txt"


# In[ ]:


num_classes = 26                                                                # 26 letters
epochs = 80                                                                     # number of epochs

train_ratio = 0.7                                                               # defining the ratio to split the dataset
samples, letters = load_dataset(dataset_file_path)                              # loading the dataset given by path
n_train_samples = int(len(samples) * train_ratio)                               # number of training samples is set to 0.7 of the whole dataset sample
x_train, y_train = samples[:n_train_samples], letters[:n_train_samples]         # first 70% of samples is put into training sample 
x_val, y_val = samples[n_train_samples:], letters[n_train_samples:]             # the rest (last 30%) is put into validation samples

x_train = x_train.astype('float32')                                             
x_val = x_val.astype('float32')
x_train /= 255                                                                  # normalizing the values
x_val /= 255
print(x_train.shape[0], 'train samples')                                        # printing the number of samples
print(x_val.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_val = keras.utils.np_utils.to_categorical(y_val, num_classes)

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(256,)))                    # input layer
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))                             # output layer

model.summary()

model.compile(loss='categorical_crossentropy',                                  # loss function for multi-class classification model
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,                                           # training the model on train set
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, verbose=0)                                 # evaluating the model on validation set
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])


# Plot the accuracy and loss of Train vs. Validation across different epochs in 2 seperate figures.

# In[ ]:


fig=plt.figure(figsize=(12,4), dpi= 100)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
fig=plt.figure(figsize=(12,4), dpi= 100)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Saving the model and the weights for later use.

# In[ ]:


model_json = model.to_json()
with open("path/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("path/model_weights.h5")
print("Saved model to disk")


# In[ ]:


from keras.models import model_from_json

# load json and create model
json_file = open("path/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("path/model_weights.h5")
loaded_model.summary()


# Real-time hand gesture recognition

# In[ ]:


VideoCapture()             #start streaming the video
eval_js('create()')

first_detect=False

while not first_detect:
  window=[] 
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)                                                 #getting the grayscale of the image for faster computation (only one color channel instead of 3)
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV)                                                  #getting the hsv channels of the image for better detection of the skin color for histogram
  face = detect(gray, cascade = Cascade_face)                                                 #detect the face in the grayscale image, since it's faster

  if len(face)==0:                                                                            #if the first face is not detected, keep the flag the same
    first_detect=False
  else:                                                                                       #else, change it to True
    first_detect=True
    
  if first_detect:                                                                            #if the first face is found, we can go to inner loop
    restricted_face = restrict(face[0],30,im)                                                 #the window that is restricted to the face
    hsv_roi=hsv[restricted_face[1]:restricted_face[3], restricted_face[0]:restricted_face[2]] #restricting the hsv to the region of interest
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))       #thresholding hsv of interested region to lower and upper bounds
    window.append(window_detect(face[0],30,im))                                               
  
    camdetect=True                                                                            #so we can use the camshift 
    hist=cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])                                   #calculating the histogram of hsv over the bigger window, using the mask
    hist=cv2.normalize(hist, hist,  alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)             #normalizing the histogram between 0 and 255

    track_window = (window[0][0], window[0][1], window[0][2], window[0][3])                   #getting the coordinates of the bigger window

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 

while camdetect:
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) 
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV) 
  mask2 = cv2.inRange(hsv, np.array((0,64,32)), np.array((180,200,200)))
  dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)                                    #using the hsv image to calculate the backprojection

  for i in range(window[0][0], window[0][2]):                                                 #for all he pixels inside the bigger window around the face
    for j in range(window[0][1], window[0][3]): 
      dst[j,i]=0                                                                              #set the probability to zero

  dst &= mask2
  
  ret, track_window = cv2.CamShift(dst, track_window, term_crit)                              #applying camshift to get the new location                 
  
  im[:] = dst[...,np.newaxis]                                                                 #adding the probability matrix from backprojection to the image

  draw_rects(im, [restricted_face], (255, 0, 255))
 
  eval_js('showimg("{}")'.format(image2byte(im)))

  cropped_image = im[int(track_window[1]):int(track_window[1]+track_window[3]),               #store the image of the hand
                          int(track_window[0]):int(track_window[0]+track_window[2])]

                                                                    #show the image to the user
  im_size = cv2.resize(cropped_image, (16,16) , interpolation = cv2.INTER_AREA)
  res = cv2.resize(im_size, dsize=(1,256), interpolation=cv2.INTER_CUBIC)                     #resizing the image to 1x256
  flat_res = np.array([str(a[0][0]) for a in res])                                            #turning the res file values to strings
  flat_res = flat_res.astype('float32')                                                       # making the image the same type as was used for training the model
  flat_res /= 255                                                                             # normalizing it

  prediction = loaded_model.predict(flat_res[np.newaxis, :])                                  
  prediction = prediction.argmax()
  predicted_letter = chr(ord('A') + prediction)

  
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(im, predicted_letter, (int(track_window[0]), int(track_window[1])), font, 1, (0, 255, 255), 2, cv2.LINE_4)   # showing the prediction of the model on the output image

  eval_js('showimg("{}")'.format(image2byte(im)))

