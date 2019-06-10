# Referenced github projects #
https://github.com/rshivansh/San-Pytorch <br>
https://github.com/Shivanshu-Gupta/Visual-Question-Answering <br>
https://github.com/Cyanogenoid/pytorch-vqa <br>
https://github.com/hengyuan-hu/bottom-up-attention-vqa <br>


# Requirements #
Python 3.6 <br>
NLTK <br>

# ECE285-MLIP-Project-VQA Description:
 * data : folder contains all intermediate output results(json files, h5py files) <br>
 * misc: misc.Image_embedding_new.py: generate embedded image vector from image features extracted from Vgg16/ResNet152 <br>
   * misc.data_loader.py: generate dataloader for train and test data. Add raw question ,multiple choices, and answer when testing on validation set. <br>
   * misc.mutan.py: Build Mutan model to generate output vector from embedded image vector and embedded question vectors. <br>
   * misc.san.py: Build Stacked Attention Model <br>
   * misc.uils.py:some basic operations about json <br>
 * prepro: prepro.vqa: filter on raw data and generate intermediate output results(json files, h5py files) after outputs of           preprocess.py. It provides 2 method for further question embedding.
   * prepro.image: extracting image files with pretraine ResNet152/Vgg16.
 * train_model: contains trained model(for demo of model we only upload the model we will use in demo)
 * preprocess.py: download data from DSMLP and do initial preprocess,extracting image,question,multiple choices and answer from given data.
 * train.py: train model on training data ,saving every model at every epoch and save the model with lowest loss.
 * test.py: test model on validation data.
        
# Usage 
## Initial preprocess
we run preprocess.py to get two raw json files in data folder.

## preprocess vqa and image
we run prepro_vqa.py with word2vector_method=1 and word2vector_method=2.
create file "./data/\{train, val\}_image_features_after_\{vgg, res152\}".
then we run prepro_image.py to extract image features.

## train the model
run train.py to train the model.

## test the model
run test.py to test the model.
