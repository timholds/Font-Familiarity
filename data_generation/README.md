**Goal** Generate image data of some text in different fonts, with the intention of using these images to train a machine learning model downstream.

from the project root, run python data_generation/create_font_images.py

control the number of images per font:
**--text_file** - the text to be rendered in each font
**--font_file** - txt file that holds a list of fonts to render, each on their own line
**--image_resolution** - 

It accomplishes this by putting text in different fonts onto a flask app and screenshotting them. 

Main files:
lorem-ipsom.md
By default, the text it will render is in lorem ipsum

TODO move all the files needed for data generation into the folder
fonts.txt
lorem.ipsom
TODO add arg for the html canvas size to create_font_images and for the number of images to generate per class

Rerun it with a different directory than fonts-images2 and use the fonts.txt instead of full-fonts-list.txt

TODO add argparsing to prep_train_test_data.py so that it grabs the right dataset