The readme is a project notes section for now.

# files
- html files to render the text into fonts
- flask server to put the baked html onto a webpage
- training data generator that takes screenshots of those webpages
- model class with pytorch achitecture
- training class 
- some other file to do the similarity computation

Problem Statement: how do we get the most similar fonts to a given font?
[ ] data augmentation - vary positioning of the text
[ ] try a clip model of same font different font?
[ ] train a classifier model and use the average class features to find which classes are closer or more similar to each other and return the top 5
[ ] do we get anything out of top eigenvectors of the data covariance matrix
[ ] distance between their mean images.

# files needed

ML steps
[ ] train a distance model on cifar to make sure the idea works
[ ] figure out how to get the mean image of a class
[ ] how many examples per class do we need

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

[ ] figure out how to get the images into a nice shape for ML
- what resolution should i use 
- should i be using PIL to create the images directly and use the google fonts api to download the fonts files locally?

Update whole shebang into a data generating parameter
Create a separate system for training the network once you have the data
[ ] get it reading from the lorum ipson file