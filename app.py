from flask import Flask, render_template, request
from flask_restful import Api, Resource

import keras.utils as image
import os
import numpy as np
import pandas as pd
from keras.models import load_model
import shutil
import cv2 as cv2
 
app = Flask(__name__)
api = Api(app)

model_path = r'EfficientNetB3_Model_22.h5'
working_dir = r'./App/'

class_label_map = { 0: 'Eczema',
                    1: 'Warts',
                    2: 'Melanoma',
                    3: 'Atopic',
                    4: 'Basal',
                    5: 'Melanocytic',
                    6: 'Benign',
                    7: 'Psoriasis',
                    8: 'Seborrheic',
                    9: 'Tinea'}


def predictor(sdir,  model_path, crop_image = False):    
    img_size=(300, 300)
    scale=1
    try: 
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1]) 
        print (s1,s2)
    path_list=[]
    paths=os.listdir(sdir)
    for f in paths:
        path_list.append(os.path.join(sdir,f))
    print (' Model is being loaded - this will take a few seconds')
    model=load_model(model_path)
    image_count=len(path_list)    
    index_list=[] 
    prob_list=[]
    cropped_image_list=[]
    good_image_count=0
    for i in range (image_count):       
        img=cv2.imread(path_list[i])
        if crop_image == True:

            # status, img=crop(img)
            status=True
        else:
            status=True
        if status== True:
            good_image_count +=1
            img=cv2.resize(img, img_size)
            cropped_image_list.append(img)
            img=img*s2 - s1
            img=np.expand_dims(img, axis=0)
            p= np.squeeze (model.predict(img))
            index=np.argmax(p)
            prob=p[index]
            index_list.append(index)
            prob_list.append(prob)
            print("p: ", p)
    if good_image_count==1:
        # class_name= class_df['class'].iloc[index_list[0]]
        class_name = class_label_map[index_list[0]]
        probability= prob_list[0]
        img=cropped_image_list [0]
        return class_name, probability
    elif good_image_count == 0:
        return None, None
    most=0
    for i in range (len(index_list)-1):
        key= index_list[i]
        keycount=0
        for j in range (i+1, len(index_list)):
            nkey= index_list[j]
            if nkey == key:
                keycount +=1
        if keycount> most:
            most=keycount
            isave=i
    best_index=index_list[isave]    
    psum=0
    bestsum=0
    for i in range (len(index_list)):
        psum += prob_list[i]
        if index_list[i]==best_index:
            bestsum += prob_list[i]  
    img= cropped_image_list[isave]/255    
    # class_name=class_df['class'].iloc[best_index]
    class_name = class_label_map[best_index]
    return class_name, bestsum/image_count


def predict(image_path, store_path, img):
    # model was trained on rgb images so convert image to rgb
    file_name=os.path.split(image_path)[1]
    dst_path=os.path.join(store_path, file_name)
    cv2.imwrite(dst_path, img)
    # check if the directory was created and image stored
    print(os.listdir(store_path))

    class_name, probability = predictor(store_path,  model_path, crop_image = False) # run the classifier
    msg=f'Image is of class {class_name} with a probability of {probability * 100: 6.2f} %'
    print(msg)
    return [class_name, probability]


@app.route('/', methods=['GET'])
def init():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def hostingBased():
    imagefile = request.files['imageFile']
    image_path = "./App/images/" + imagefile.filename
    imagefile.save(image_path)

    store_path=os.path.join(working_dir, 'storage')
    if os.path.isdir(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)
    
    img=cv2.imread(image_path,  cv2.IMREAD_REDUCED_COLOR_2)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    class_name = predict(image_path, store_path, img)[0]
    prob = predict(image_path, store_path, img)[1] 

    if os.path.exists(image_path):
        os.remove(image_path)
        print("File deleted:", image_path)
    else:
        print("File does not exist:", image_path)

    return render_template('index.html', prediction = class_name, probability = f' {prob * 100: 6.2f} %')


class predictApi(Resource):
    def post(self):
        # data = request.get_json()
        # image_data = request.form.get('image')
        imagefile = request.files['inputImage']
        if imagefile:
            image_path = "./App/images/" + imagefile.filename
            imagefile.save(image_path)

            store_path=os.path.join(working_dir, 'storage')
            if os.path.isdir(store_path):
                shutil.rmtree(store_path)
            os.mkdir(store_path)
            
            img=cv2.imread(image_path,  cv2.IMREAD_REDUCED_COLOR_2)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            class_name = predict(image_path, store_path, img)[0]
            prob = predict(image_path, store_path, img)[1]

            if os.path.exists(image_path):
                os.remove(image_path)
                print("File deleted:", image_path)
            else:
                print("File does not exist:", image_path)

            return {'class': class_name, 'prob': prob*100}, 200
        else:
            return {'message': 'No data received'}, 400  # Bad request status code
        

api.add_resource(predictApi, '/api/predict')

if __name__ == '__main__':
    app.run(port=3000, debug=True)