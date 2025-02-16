
## Experiments
[ ] Try contrastive loss
[ ] Try triplet loss
[ ] Experiment with weight decay
[ ] Test single conv layers vs back to back conv layers between downsamples
[ ] Add a wandb config file and sweep conv sizes
[ ] Compare my model to a LoRA of gemmapali or a multimodal llama





[ ] move the class embedding generation into prep train test data.py? Otherwise go back to using a class embedding argument in the frontend_app.py instead of searching for it in the dataset folder 
[ ] fix the error when uploading an image
[ ] add some processing to resize images coming in and make them single channel
[ ] remove the model saving part for the sweep. once we know which model trains best we can retrain it and reanble the saving.
[ ] cleanup the metrics 
- [ ] is the epoch time metric actually telling me how long an epoch takes?


[X] Figure out how to keep track of the model experiments - use weights and biases 
[X] Test my model on cifar dataset instead to sanity check that it is capable of learning anything  
[X] create a small test dataset for iterating on the model - font-dataset-npz_test  
[X] make it so that it only saves the class averages at the end of training instead of everytime a new best model gets saved  
[X] Establish simple conv baseline for the encoder
[X] add a LR warmup and cosine learning rate  
[ ] Test a resnet as the encoder
[ ] Test a resnet with supervised contrastive loss (does it need to be a pretrain)


# Training on Alienware
- ssh timholds@192.168.68.125
    - setup ssh keys 
    - Creating a simple shell script to automate the sync + train process
- pull github repo with the latest model file 
- use github DVC w/ free tier of S3?


# Deployment
clear up confusion around the deployment steps
create the deployment directory
do i need to create the deployment directory with the script locally?
what if i copy paste all the files needed to launch the server into a specific folder and then just copied the folder
- need to copy the model folder into ml

[X] get a digital ocean ubuntu 22.04 server, 1GB RAM
- `ssh -i ~/.ssh/digitalocean root@137.184.232.187` mac or just
- `ssh root@137.184.232.187` from linux
[X] run the `deploy-structure.sh` script locally  
`chmod +x deployment/scripts/deploy-structure.sh`  
`./deployment/scripts/deploy-structure.sh`  
[X] copy over the deployment
`sudo rsync -avz deployment/ root@137.184.232.187:/var/www/freefontfinder/deployment/`
[X] copy of ml.model, class_embeddings, frontend_app.py,and frontend_requirements.txt. 
```
rsync -avz \
    deployment/ \
    root@137.184.232.187:/var/www/freefontfinder/deployment/
```
``` 
rsync -avz \
    frontend_app.py \
    frontend_requirements.txt \
    root@137.184.232.187:/var/www/freefontfinder/
```  
```
rsync -avz \
    templates/frontend.html \ 
    root@137.184.232.187:/var/www/freefontfinder/templates/
```   
```
rsync -avz \
    data/font-dataset-npz/fontCNN_BS64-ED512-IC32.pt \
    data/font-dataset-npz/class_embeddings_512.npy \ 
    data/font-dataset-npz/label_mapping.npy \
    root@137.184.232.187:/var/www/freefontfinder/model/
```  
make sure the setup script is executable on the server
`chmod +x /var/www/freefontfinder/deployment/setup.sh`

[X] configure the setup script correctly install dependencies on server  
[ ] (optional) find a lightweight version of pytorch  
[X] get a domain and have it point to the server
[X] get SSL working
[ ] setup a github actions to deploy to the server on push to main  
    - .github/workflows/deploy.yml


## Deployment Improvements 
Frontend Enhancements
-Add proper styling (the CSS file is missing)
-Add error handling for failed predictions
-Add loading states during image processing


Monitoring & Maintenance
-Set up proper application monitoring
-Implement backup strategy
-Add more comprehensive error logging


Security Enhancements
- Consider rate limiting
-Add input validation
-Implement file upload size limits
-Consider adding request logging for security monitoring


Performance Optimization
-Add caching for static assets
-Optimize model loading
-Consider CDN for static files

## TODO 
[X] figure out a workflow for deployment 
    - need to copy over the 
    - gather model file, embeddings file, 
    - what files do i need to copy over for the deployment? just the frontend_app - maybe make a frontend_requirements.txt? need flask, pytorch (pytroch lite?)
[ ] add test_e2e.py to github actions when merging develop into main
- update the script to automatically upload an image to the server and check the response



## ML Steps
Simple baseline model
Get a model to overfit
Regularize
Try a more complex model
Try a more complex model with more data
Try a more complex model with more data and more regularization
Try a more complex model with more data and more regularization and more data augmentation
[ ] Visualize model predictions some small holdout batch every epoch


## Bugs
[x] critical:  there's an off by one error in the labels mapping. When I try to use a super overfit model on the frontend, it returns the wrong class with super high certainty! Checking this out closer, I notice it's predicting the class 1 off from the correct class. For example, on an image of the "yesseva" font, the model predicts "yesteryear" with 99.5% certainty. 


# ML TODO
[ ] model experimentation: data augmentation - vary positioning/layout, font size and style, color and background color, text itself  
[ ] try a clip model of same font different font?  
[ ] train a classifier model and use the average class features to find which classes are closer or more similar to each other and return the top 5  
[ ] do we get anything out of top eigenvectors of the data covariance matrix  
[ ] distance between their mean images 
[ X ] how can i make the model name legible / get returned from the train script? the filename itself has some of the hyperparams baked in - for example `fontCNN_BS64-ED512-IC16.pt`  
- solution: create a ml.utils file with get_model_path()

# Misc
- try uploading to huggingface again

Do I need to do any regularization?



figure out how to combine steps in one script for gathering data and training the model
1) generate the data, save the data, prep_train_test_data.py     
2) train, generate class embeddings, save class embeddings, launch flask app  


## Questions
[ ] does it matter whether you take the class average over the train set vs the validation or test set?  



## Questions
[ ] does it matter whether you take the class average over the train set vs the validation or test set?  

## Compression
How are we handling compression?
What does image compression actually mean? Explain why it would be nonsensical to 

**On disk** We want the images compressed on disk so they don't take up too much space. If we were using cloud storage like AWS S3, having the images compressed will save money, since they charge based on the size of outgoing data rather than the number of transfer events.

**At train time**, we want the data uncompressed on the GPU so that we have arrays 

What kind of color space do we need for the training images?


Imagine the space of image augmentations that would be helpful if we wanted to let people upload arbitrary photos of fonts?
- font size
- bold / italicized
- font color, background color
- background texture
- text placement / centering

.npz is for saving multiple numpy arrays, .npy is for saving single arrays

.npz is a good choice here since it's more memory-efficient than .pkl for large arrays, and since the dataset is significantly larger than CIFAR.


## Questions
How much does it matter if the model is trained on jpg images but someone inputs a png image? The png in theory is not lossy, but we will have to resize it to the same size as the training images, which will introduce some lossiness.

-TODO some explaining about high dimensional representations where all the datapoints are super far from each other so being close in one dimension ends up being close in 

## Machine learning 
### Inference



### ML Questions
What if I framed my problem also as first a character recognition detection problem, and then used the sum of these 
- generated the images of all the charcters in PIL and then do all the data augmentation to the images where each image has just one character in it

[ ] find a good open-source ocr character segmentation model and use it to generate 
-> does a segmentation model help at all here? forcing the model to learn exactly which pixels are and aren't part of the font? caveat is that the low resolution images probably won't work

[ ] Are there any ML strategies for doing CCE on a dataset with a large number of classes?

[ ] how many datapoints per class do I want if I have around 700 classes? cifar1000 archs probably a good place to start
[ ] Is it better just to keep the classifier and return the top 5 classes or to omit the classifier and just use the get_embeddings() part of the model to extract the features and then compare that to the average features of each class?


# TODO
[ ] figure out how to incorporate weight and width into the data augmentation and perhaps the model explicitly ala https://fonts.google.com/specimen/Roboto/tester  
-> might involve downloading all the font files and rendering them differently  
[ ]  update the data generation to something less crude than this original ugly hack - text = text * 10  # Repeat text to ensure enough content  
[ X ] get the script reading the text from the lorum_ipsom.txt file  
[ X ] get the fonts read in to the html page from fonts.txt  
[ X ] get the script to render the text on an html page in the correct font  
[ X ] figure out how much data I need - 100-10k images per class  
[ X ] get the screenshots saving with minimal overlap  
[ X ] collect a dataset of screenshots  
[ X ] get the full sized model to train in colab on an A100  
    - taking ~90/s per iteration  
Make sure that the metrics look right and wandb   
[  ] Get a sweep working on colab 
[ ] Put a little demo of each of the fonts on the frontend 

[ X ] get the flask app to launch for the frontend using the model and class embeddings
[ ] make a simple frontend that is capable of taking in images of fonts and returning out similar fonts  
[ ] make a pretty frontend - maybe pay someone or ask conor  
[ ] start generating data augmentations  
 
[ ] make the validation dataset smaller  
[ ] add back model saving during the epochs instead of just at the end. i think there is a bug rn such that it saves the network state after the last epoch regardless of whether that's the best model or not.  
[ ] write a script to run to generate class embeddings using the training data after generating and saving the best train model so i can have multiple models saved and each of them can generate their own class embeddings  
[ ] figure out what data format to store the class embeddings in so that they can be used inside a flask app   
[ ] figure out how a flask app is supposed to work  
[ ] once the website it live, would be nice to be able to save the font images people are uploading so we can get a better idea of what kind of data augmentations to do


# Training experiments
First idea is just to get the loss to go down for train and test over a 30 epoch run with a tiny 3M parameter network.

# ML steps and ideas
[ ] train a distance model on cifar to make sure the idea works
[ ] figure out how to get the mean image of a class
[ ] how many examples per class do we need

## Metrics TODO
[ ] remove or debug empty classes metric
[ ] remove acc std
[ ] do I need step and epoch time charts?

"Data Impressions: Mining Deep Models to
Extract Samples for Data-free Applications" (2021)
- probably makes sense to use this approach and existing vision model
- use Dirichlet distribution to model prob(softmax output | class, trained model)
- dir sample space is probability over the classes
- "compute a normalized class similarity matrix (C) using the weights W connecting the final (softmax) and the pre-final layers"
- "concentration parameter (α) of the Dirichlet distribution" is the main juice here

no access to training data, "synthesizes pseudo samples from the underlying data
distribution on which it is trained."

do contrastive losses mean anything for us? could we classify the fonts into families and use that as an (additional) label?
-"Metric learning: focuses on learning distance metrics between data points"

[ ] figure out how to get the images into a nice shape for ML
- what resolution should i use 
- should i be using PIL to create the images directly and use the google fonts api to download the fonts files locally?

Create a separate system for training the network once you have the data

try some different text sizes for data augmentation


[ ] get an input-independent baseline by zeroing out inputs and seeing how it performs   
[ ] overfit on one batch, launch it and make sure it works on the frontend too  

# Known issues
The labels in the npz file are off by one, so we need to subtract by one in the `FontDataset`: `self.targets = data['labels']-1`


# Project Notes
## Notes
## Experiments
## Challenges
## TODOs
[ ] get the results on the frontend to be rendered in the font themselves! this should help visually add a sanity check to the results

[X] TODO delete extra create_embeddings.py file inside ml (or figure out which one is useful)
[ X ] Write 5  tests that correspond to the 5 main files that create the data, train, and do the frontend
[ X ] Disable wandb when im running the unit tests, unless needed
[ ] TODO make sure the test command args in the readme correspond to the ones in teste2e
[ ] TODO add test time fixed dataset visualizations so we can concretely see how the model is predicting

[ X ] create some test scripts to create a few images per class, train 1 epoch and save the model, load the model and do inference
- need a way to pass around class embedding file names

----
Get an open source character detection and segmentation model - font model should be able to sum across all the softmax values characters in the image and return the most likely font



[ ] get the model to train on a small dataset of 10 classes and 100 images per class
[ ] get the

Put on the front page:
crop your image so that there is only one font present
Upload an image or screenshot
