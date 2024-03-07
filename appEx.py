from flask import Flask, render_template, request

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
import keras.utils as image
import os
import numpy as np
import pandas as pd
from keras.models import load_model
import shutil
import cv2 as cv2

app = Flask(__name__)
model_path = r'../EfficientNetB3-skin disease-85.23.h5'
working_dir = r'./'

def predictor(sdir, csv_path,  model_path, crop_image = False):    
    # read in the csv file
    class_df=pd.read_csv(csv_path)    
    img_height=int(class_df['height'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_height, img_width)
    scale=class_df['scale by'].iloc[0] 
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
    print("|||||||||||||||||||||", path_list)
    print (' Model is being loaded- this will take a few seconds')
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
    if good_image_count==1:
        # class_name= class_df['class'].iloc[index_list[0]]
        class_name = class_label_map[index_list[0]]
        probability= prob_list[0]
        img=cropped_image_list [0] 
        print("1|||||||||||||||||||||", class_name, index_list[0])
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
    print("2|||||||||||||||||||||", class_name, best_index)
    return class_name, bestsum/image_count

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

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

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    store_path=os.path.join(working_dir, 'storage')
    if os.path.isdir(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)
    
    img=cv2.imread(image_path,  cv2.IMREAD_REDUCED_COLOR_2)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # model was trained on rgb images so convert image to rgb
    file_name=os.path.split(image_path)[1]
    dst_path=os.path.join(store_path, file_name)
    cv2.imwrite(dst_path, img)
    # check if the directory was created and image stored
    print(os.listdir(store_path))

    csv_path=r'../class_dict.csv' # path to class_dict.csv
    class_name, probability = predictor(store_path, csv_path,  model_path, crop_image = False) # run the classifier
    msg=f' image is of class {class_name} with a probability of {probability * 100: 6.2f} %'
    print(msg)

    return render_template('index.html', prediction = class_name)

    # # Load the image
    # img = image.load_img(image_path, target_size=(75, 100))  # Resize to match model input size
    # img_array = image.img_to_array(img)
    # img_array /= 255.0  # Normalize pixel values (assuming your model expects values in [0, 1])

    # # Expand dimensions to match model input shape
    # img_array = np.expand_dims(img_array, axis=0)

    # # Now you can make predictions
    # predictions = model.predict(img_array)

    # # Get the index of the highest probability
    # predicted_class_index = np.argmax(predictions)

    # # Get the corresponding label
    # predicted_class_label = class_label_map[predicted_class_index]

    # # Print the predicted class label and its corresponding probability
    # print("Predicted class:", predicted_class_label)
    # print("Probability:", predictions[0][predicted_class_index])

    # # classification = '%s (%.2f%%)' % (label[1], label[2]*100)


    # return render_template('index.html', prediction=predicted_class_label)


if __name__ == '__main__':
    app.run(port=3000, debug=True)