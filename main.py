

from fastapi import FastAPI, File, UploadFile

import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import skimage.transform as trans
from warnings import filterwarnings

app = FastAPI()


@app.post("/files")
async def UploadImage(file: bytes = File(...)):
    with open(f'as.jpg','wb') as image:
        image.write(file)
        image.close()
    




    filterwarnings("ignore",category=DeprecationWarning)
    filterwarnings("ignore", category=FutureWarning) 
    filterwarnings("ignore", category=UserWarning)
    ClassModel = load_model('BTcnnModel.h5')
    
    def Class_Pridict(path , color):

        r,g,b = color
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = image / 255
        image = trans.resize(image,(256,256,1))
        predicted = ClassModel.predict(np.reshape(image, (1, 256, 256, 1)))

        if predicted[0][0] > 0.5 :
            w=1
            return w
        else :
            w=0
            return w
            
    x=  Class_Pridict('as.jpg' , (0, 0 ,255))
    
    if x == 1:
        r= "yes"
    else :
        r= "no"

    

    
    return r

















