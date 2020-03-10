**0.How to use?**

1. download these code: open your powershell and "cd" into the folder that you wanna store this code. 
2. use git command:"git clone https://github.com/lizeyujack/summerstudio2020" to download it.
3. open your IDE(pycharm or VisualStudio) and run the file named "Classifier.py".

During the period of running, your webcam(camera) will open and close 3 times, the first time is for generate your gender, the second one is for your ethnicity and the last time is used for generating your cloth. In the end, there will be some pictures (ads) shows as slides to you based on the information captured by this model.

If you are interested in how did I trained my model, you can click this link https://github.com/lizeyujack/summerstudio2020/tree/master/training

**1. training**
***

The training phase is the core of our entire project. It includes collecting, filtering  and labeling our datasets. Finally, I converted some these ".xml" file to ".record" file.  I detailed what I have done to train a detector using [Google Cloud Platform](https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=japac-AU-all-en-dr-bkws-all-super-trial-e-dr-1008074&utm_content=text-ad-none-none-DEV_c-CRE_248263937479-ADGP_Hybrid+%7C+AW+SEM+%7C+BKWS+~+T1+%7C+EXA+%7C+General+%7C+1:1+%7C+AU+%7C+en+%7C+google+cloud+platform-KWID_43700023244271242-kwd-296644789888&userloc_1000286&utm_term=KW_google%20cloud%20platform&gclid=EAIaIQobChMIiNXsqpn75wIVFoWPCh0EEwqzEAAYASAAEgJ_gfD_BwE),
and I have referenced from the object detection tutorial [here](https://github.com/tensorflow/models/tree/master/research/object_detection). I haved created a virtual environment with dorker to make training easier. I have set up everything in this  [repository](https://gitlab.com/ZeyuLi123/tensorflow2), so I do not need to make other changes after cloning it into my VM Engine .

![training](https://github.com/lizeyujack/summerstudio2020/blob/master/images/training.gif)

**2. " model file"**
***
there are three file end with _model  (Gender_model, clothing_model and ethnicity_model), and these files are used for story the model that we have trained.

he highlight of our project are our `slideshow()`. We have built 17 different classes and we gave these classes different advertisement based on their different Attributes.

I put some pictures on each folders you could find them in `slowshow/`.

**3. Classifier.py**
****
Run `Classifier.py` could achieve all the functions of our program, It will detect you gender, ethnicity and clothing and return the forecast result.

Then, based on prediction informaion, the project will fetch some images from `slowshow/` and put these images into `slideshow/Ad_images`.

Finally, Our project will automatically play recommended ads(images in `slideshow/Ad_images`).

*The following video is the result of running the `Classifier.py` file. This is also the program we showed to you on the display day.*


![training](https://github.com/lizeyujack/summerstudio2020/blob/master/images/classifier1.gif)

**4. wabcam.py**
****
it is used for opening the webcam of our laptop to visualize our predicrion. Users can get the prediction to test our detector's accuracy. Because of the faster r-cnn of the model we use, this model is characterized by high accuracy with serious delay. When our products are applied in practice, the requirement of accuracy rate is relatively high, while the delay phenomenon can be ignored, so we did not choose the smooth mobilenet model.

*The video below is the effect of the trained race model.*


![training](https://github.com/lizeyujack/summerstudio2020/blob/master/images/ethnicity.gif)

**5. test.py**
****
Some functions are customized in `test.py`, I referenced it in `Classifier.py`. The basic function of this file is to empty all of the images in `slideshow/Ad_images` which is easier for us to recollect pictures in corresponding folders.


