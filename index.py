from time import sleep
from flask import Flask, render_template, request, redirect, send_file
from uuid import uuid4
import os
import shutil
from model_selector.frame_extractor import save_frames
import model_selector.ModelSelector as ms

app = Flask(__name__)
selector = None # global model selector

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file(): 
    '''
        endpoint for uploading all the files
        All the video will be extracted into images
        Output images will be stored in the images folder and the subfolders will be the classes and their respective images
    '''
    length = len(request.form) // 2
    clearfolder("images")
    clearfolder("temp_video")

    for i in range(1,length+1):
        dest = os.path.join("images", request.form["class_name-"+str(i)])
        video_temp_dest = "temp_video"
        os.mkdir(dest)
        if not os.path.exists(video_temp_dest):
            os.mkdir(video_temp_dest)
        files = request.files.getlist("file-"+str(i))

        # determining the file type from the radio data file_type-#
        filetype = request.form.get("file_type-"+str(i))
        if filetype == "image":
            for file in files:
                filename = str(uuid4()) + ".jpg"
                file.save(os.path.join(dest, filename))
        else:
            # video file
            for file in files:
                filename = file.filename
                file.save(os.path.join(video_temp_dest, filename))
                # extract the images from the video
                save_frames(os.path.join(video_temp_dest, filename), dest, max_frames=200)
        #clearfolder(video_temp_dest)

    return redirect("/model_build")

@app.route("/model_build")
def model_build():
    return render_template("model.html")

@app.route("/train",methods=['POST'])
def train():
    print(request.form)
    models_list = ["mobilenet_v3_large_100_224","efficientnet_b7","inception_resnet_v2","efficientnetv2_xl_21k_ft1k","efficientnetv2_b3_21k_ft1k"]
    for mdl in models_list.copy():
        if request.form.get(mdl) != 'on':
            models_list.remove(mdl)
    epochs = request.form['epochs']
    print("model list",models_list)
    
    # getting handles of each models
    base_models = {}
    print("models list ",models_list)
    for mdl in models_list:
        base_models[mdl] = ms.base_models[mdl]
    base_models_dir = os.path.join("base_models","base_models")

    # balancing the data in the images folder
    ms.CustomDirectoryIterator.balance_images("images")

    global selector
    selector = ms.ModelSelector("images",base_models, ms.input_shape, "saved_models", 'base_models')

    if not os.path.exists(base_models_dir):
        # download the base models
        print('downloading basemodels...')
        selector.download_basemodels()
    clearfolder("saved_models")
    selector.load_models(load_from_local=False, batch_size=10) 
    print("loaded")
    selector.train_models(epochs, tune_hyperparameters=True, max_trials=3)
    print("trained")
   
    return redirect("/predict/empty")

@app.route("/predict/<abc>", methods=["POST","GET"])
def predict_page(abc):
    if abc == "show":
        images = request.files.getlist("files")
        clearfolder(os.path.join("static","prediction","prediction"))
        names = {}
        for file in images:
            name = str(uuid4())+".jpg"
            dest = os.path.join("static","prediction","prediction",name)
            # names[name] = "Predicted:"+"a"+"actual:"+"b"
            file.save(dest)
        if selector is not None:
            y_true, y_pred = selector.predict(os.path.join("static","prediction"),os.path.join("static","predicted"))
            # the images in the static/prediction dir will be used for prediction and the corresponding images will be saved in the static/predicted with name 1 to n
            print(y_true, y_pred)
            count_name = 1
            for i in y_pred:
                name = str(count_name)+".jpeg"
                names[name] = i
                count_name += 1
        else:
            print("not loaded, server might be crashed")
            return render_template("predict.html")
            
        return render_template("predict.html", images=names)
    return render_template("predict.html")

@app.route("/export")
def export():
    # zipping the file
    shutil.make_archive("trained_models","zip","saved_models")
    return send_file("trained_models.zip", as_attachment=True)

def clearfolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

if __name__ == '__main__':
    app.run(debug=True)