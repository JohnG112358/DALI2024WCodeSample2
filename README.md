# DALI 2024 Winter Application - Machine Learning Track
### John Guerrerio

This repository contains my second code sample for my 2024 Winter DALI application.  It consists of code
to fine-tune a pre-trained BERT model for the LitCovid named entity recognition (NER) task (for more details, see my code sample explanation)
and several requisite files for that task.

## Files:
- LitCovid_Model.py: The code to fine-tune a BERT model for the LitCovid NER task.
- LitCovid_preprocessed: The preprocessed and labeled dataset to train the model on
- LitCovid_combined: The combined preprocessed and labeled validation and test sets.  This allows us to evaluate the model
on the validation and test sets at each epoch.  While this is unconventional, this approach was suggested me to my mentor 
as a means of gathering more information to debug an issue relating to our model's performance.

## Notes:
- This code requires an NVIDIA GPU to run in a reasonable amount of time
- This code was written in 2022 and WILL NOT RUN WITH THE MODERN VERSIONS OF PYTHON, TORCH, TRANSFORMERS, etc.  Please use Python 3.10
and the dependency versions specified in requirements.txt or the code will not run.