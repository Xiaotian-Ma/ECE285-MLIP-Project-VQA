# Referenced github projects #
https://github.com/rshivansh/San-Pytorch, 
https://github.com/Shivanshu-Gupta/Visual-Question-Answering
https://github.com/Cyanogenoid/pytorch-vqa
https://github.com/hengyuan-hu/bottom-up-attention-vqa


# Requirements #
Python 3.6
NLTK

# ECE285-MLIP-Project-VQA Description:
data: folder contains all intermediate output results(json files, h5py files)
misc: misc.Image_embedding_new.py: generate embedded image vector from image features extracted from Vgg16/ResNet152
      misc.data_loader.py: generate dataloader for train and test data. Add raw question ,multiple choices, and answer when testing on validation set.
      misc.mutan.py: Build Mutan model to generate output vector from embedded image vector and embedded question vectors.
      misc.san.py: Build Stacked Attention Model 
      misc.uils.py:some basic operations about json
prepro: prepro.vqa: filter on raw data and generate intermediate output results(json files, h5py files) after outputs of preprocess.py. It provides 2 method for further question embedding.
        prepro.image: extracting image files with pretraine ResNet152/Vgg16.
train_model: contains trained model(for demo of model we only upload the model we will use in demo)
preprocess.py: download data from DSMLP and do initial preprocess,extracting image,question,multiple choices and answer from given data.
train.py: train model on training data ,saving every model at every epoch and save the model with lowest loss.
test.py: test model on validation data.
        
# Usage 
#Initial preprocess
we run preprocess.py to get two raw json files in data folder.

# preprocess vqa and image
we run preprocess_vqa.py with word2vector_method=1 and word2vector_method=2.


