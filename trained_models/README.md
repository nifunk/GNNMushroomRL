This folder contains some of the trained models that have been used for the results presented in the paper. For the
corresponding commands to use the models, please checkout the arguments.txt file in each of them and replace 
PATH_TO_REPO with the actual path to the place where you cloned this repository. Note that the python script to run these
commands is to be found inside the examples folder.

Note: At the end of the arguments.txt file we also provide the commands how such a model might be obtained by training.

If you do not have torch available, please omit the argument --use-cuda from all the training commands. Unfortunately,
if you want to take a look at the pretrained models that were in our case obtained using cuda, please checkout the other
branch of this repo which has a slightly adapted loading procedure and is especially constructed for this case.