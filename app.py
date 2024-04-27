from flask import Flask, render_template, request
from flask_restful import Api, Resource

import os
import numpy as np
from keras.models import load_model
import shutil
import cv2 as cv2
 
app = Flask(__name__)
api = Api(app)

# model_path = r'EfficientNetB3_Model_22.tf'
model = load_model('EfficientNetB2-5Classes.h5')
working_dir = r'./App/'

class_label_map = { 0: 'Melanoma',
                    1: 'Basal Cell Carcinoma',
                    2: 'Melanocytic Nevi',
                    3: 'Benign Keratosis-like Lesions',
                    4: 'Seborrheic Keratoses and other Benign Tumors'}

def predictor(sdir):    
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
    image_count=len(path_list)   
    index_list=[]
    prob_list=[]
    cropped_image_list=[]
    good_image_count=0
    for i in range (image_count):     
        print(3)   
        img=cv2.imread(path_list[i])
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
        
    class_name = class_label_map[index_list[0]]
    probability= prob_list[0]
    img=cropped_image_list [0]
    return class_name, probability


def predict(image_path, store_path, img):
    # model was trained on rgb images so convert image to rgb
    file_name=os.path.split(image_path)[1]
    dst_path=os.path.join(store_path, file_name)
    cv2.imwrite(dst_path, img)
    # check if the directory was created and image stored
    print(os.listdir(store_path))

    class_name, probability = predictor(store_path) # run the classifier
    msg=f'Image is of class {class_name} with a probability of {probability * 100: 6.2f} %'
    print(msg)
    return class_name, probability

@app.route('/', methods=['GET'])
def init():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def hostingBased():
    imagefile = request.files['imageFile']
    image_path = "./App/images/" + imagefile.filename
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imagefile.save(image_path)

    store_path=os.path.join(working_dir, 'storage')
    if os.path.isdir(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)
    
    try:
        img=cv2.imread(image_path,  cv2.IMREAD_REDUCED_COLOR_2)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return render_template('index.html', output = "Error")

    class_name, prob = predict(image_path, store_path, img)

    if os.path.exists(image_path):
        os.remove(image_path)
        print("File deleted:", image_path)
    else:
        print("File does not exist:", image_path)

    if prob > 0.7:
        output = "Image is of class " + class_name # + "with a probability of " + f' {prob * 100: 6.2f} %'
    else:
        output = "Not good enough to make accurate prediction for this case"

    return render_template('index.html', output = output)


# class predictApi(Resource):
#     def post(self):
#         # data = request.get_json()
#         # image_data = request.form.get('image')
#         imagefile = request.files['inputImage']
#         if imagefile:
#             image_path = "./App/images/" + imagefile.filename
#             os.makedirs(os.path.dirname(image_path), exist_ok=True)
#             imagefile.save(image_path)

#             store_path=os.path.join(working_dir, 'storage')
#             if os.path.isdir(store_path):
#                 shutil.rmtree(store_path)
#             os.mkdir(store_path)

#             try:
#                 img=cv2.imread(image_path,  cv2.IMREAD_REDUCED_COLOR_2)
#                 img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             except:
#                 return {'message': 'Error'}, 200

#             class_name, prob = predict(image_path, store_path, img)

#             if os.path.exists(image_path):
#                 os.remove(image_path)
#                 print("File deleted:", image_path)
#             else:
#                 print("File does not exist:", image_path)
                
#             return {'class': class_name, 'prob': prob*100}, 200
#         else:
#             return {'message': 'No data received'}, 400  # Bad request status code
        

# api.add_resource(predictApi, '/api/predict')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


# if __name__ == '__main__':
#     app.run(port=3000, debug=True)

    