"""
DALI 2024W Application - Machine Learning Track
John Guerrerio

This file trains a deep-learning based named entity recognition model for the LitCovid NER task (explained in the code
sample explanation).

Note hyperparameter tuning occurred in an external bash script that controlled model training within the NIH's GPU cluster
Thus, this file uses default values for the hyperparameters
"""


import transformers
import torch
import datasets
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

# records precision, recall, f1, and accuracy
metric = load_metric("seqeval")

# We use BIO tags for this named entity recognition task
# BIO labels are a good way to simplify the process of converting the NER task into a sequence prediction task
# B-entity_type indicates the first token in an entity string
# I-entity_type indicates internal tokens in an entity string
# O indicated tokens that are not part of an entity string
label_list = ['O', 'B-VAC', 'I-VAC', 'B-STR', 'I-STR', 'B-VACFUND', 'I-VACFUND']

# the BERT model we are going to fine tune
# I could not find the pre-trained model I used when I  originally performed this research on HuggingFace,
# so I am using this model in its stead
# With PubMedBERT (the original model I used) I got a f1 around 0.97 after hyper parameter tuning
# I can't make any guarantees about the performance of this model
model_checkpoint = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

# tokenizer for the model we want to fine-tune
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Random seed to ensure reproducibility
RANDOM_SEED = 42

# Checks that tokenizer is a fast tokenizer
# These tokenizers are written in Rust and are significantly faster than the "slow" tokenizers written in Python
# This code is written to work with fast tokenizers only
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def tokenize_and_align_labels(examples, bio=True):
    """
    During preprocessing, we tokenize abstracts by word.  However, the tokenizer for our BERT model divides words into
    sub-word tokens.  We need to adjust our labels to account for these additional sub-word tokens
    Args:
        examples: A dataset containing the documents to tokenize and labels to adjust
        bio = A boolean representing weather the BIO labels or IO (BIO labels without the "B" labels) labels should be
        used (default is BIO)
    Returns:
        tokenized_inputs: A dictionary containing the tokens and adjusted labels for a document
    Raises:
        None
    """
    # run abstracts through the tokenizer
    tokenized_inputs = tokenizer(examples["sentences"], truncation=True, max_length=512, is_split_into_words=True)

    labels = []

    tags = examples["ner_tags"]
    if bio:
        tags = examples["bio_ner_tags"]

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # -100 padding/beginning/ending tokens
            elif word_idx != previous_word_idx:  # check if the current token corresponds to the last one
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels  # updated labels are added to the tokenized inputs
    return tokenized_inputs


def compute_metrics(p):
    """
    Function to calculate the model's performance on a single dataset after it makes predictions

    Args:
        p: Model output after it has made predictions (predictions and labels)
    Returns:
        A dictionary containing the model's precision, recall, f1, and accuracy
    Raises:
        None
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  # gets model predictions from BERT output tensors

    # Ignores -100 padding/beginning/ending tokens
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]  # gets model's predictions for a sequence

    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]  # gets ground truth for the same sequence

    # computes precision, recall, and f1
    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def compute_combined_metrics(p):
    """
    This function calculates the model's performance on the validation and test sets independently.

    Trainer does not support evaluation on multiple datasets, so to evaluate on both the validation and test set
    they need to be combined into one dataset initially and then the predictions are split into their respective
    datasets using the known length of each dataset

    In this case, the validation dataset has 50 documents and the test dataset has 100 documents
    Args:
        p: Model output after it has made predictions (labels and predictions)
    Returns:
        A dictionary containing the model's precision, recall, f1, and accuracy for both the validation set
        and the test set
    Raises:
        None
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  # gets model predictions from BERT output tensors

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]  # Ignores -100 padding/beginning/ending tokens

    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]

    # split out predictions and labels for the validation set
    validation_predictions = true_predictions[:50]
    validation_labels = true_labels[:50]

    # split out predictions and labels for the test set
    test_predictions = true_predictions[50:]
    test_labels = true_labels[50:]

    # compute metrics for validation and test set
    validation_results = metric.compute(predictions=validation_predictions, references=validation_labels)
    test_results = metric.compute(predictions=test_predictions, references=test_labels)

    return {
        "validation precision": validation_results["overall_precision"],
        "validation recall": validation_results["overall_recall"],
        "validation f1": validation_results["overall_f1"],
        "validation accuracy": validation_results["overall_accuracy"],
        "test precision": test_results["overall_precision"],
        "test recall": test_results["overall_recall"],
        "test f1": test_results["overall_f1"],
        "test accuracy": test_results["overall_accuracy"],
    }


def train_LitCovid_model(batch_size, data_file, learning_rate, num_epochs, weight_decay, save_directory, verbose=True,
                         gpu=True, combined_metrics=True, combined_dataset=None):
    """
    Args:
        batch_size: The batch size to be used when training the model
        data_file: A path to the full preprocessed dataset to train the model
        learning_rate: The learning rate to be used when training the model
        num_epochs: The number of epochs to train the model for
        weight_decay: The weight decay to use when training the model
        save_directory: The directory to save the pretrained model (will be created if it doesn't exist)
        verbose: A boolean representing weather extra information about the model should be printed (defaults to true)
        gpu: A boolean representing weather information about the gpu the model is training on
        should be printed (defaults to true)
        combined_metrics: Weather to compute evaluation metrics on the validation and test datasets at each epoch or
        on the validation set at each epoch and the test set only at the end (calculating metrics on both helped
        with debugging when performing the research)
        combined_dataset: A path to the preprocessed file containing the test and validation data sets
        (they must already have been split and shuffled; the validation set should come before the test set)
    Returns:
        Saves a pre-trained deep learning-based LitCovid NER model to the directory save_directory
    Raises:
        AssertionError if gpu is True and no gpu is available
        JsonDecodeError if combined_metrics is true and combined_dataset is not specified
    """
    # loads the preprocessed dataset
    dataset = load_dataset('json', data_files=data_file, field="records")

    # splits the dataset into train, validation and test sets
    # 70-10-20 split; change as needed
    train_testvalid = dataset['train'].train_test_split(test_size=0.3, seed=RANDOM_SEED)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.33, seed=RANDOM_SEED)

    if not combined_metrics:
        split_dataset = datasets.DatasetDict(
            {"train": train_testvalid['train'], "test": test_valid['train'], 'validation': test_valid["test"]})
    else:
        combined = load_dataset('json', data_files=combined_dataset, field="records")
        # "validation" dataset should be both the test set and the validation set in this case
        # trainer will only calculate metrics on the validation set at each epoch, so we need to pass in both sets here
        # and separate them when calculating the metrics
        split_dataset = datasets.DatasetDict(
            {"train": train_testvalid['train'], "test": test_valid['train'], 'validation': combined["train"]})

    if verbose:
        print("MODEL CHECKPOINT: " + model_checkpoint)
        print("BATCH SIZE: " + str(batch_size))
        print("LEARNING RATE: " + str(learning_rate))
        print("NUM EPOCHS: " + str(num_epochs))
        print("WEIGHT DECAY: " + str(weight_decay))
        print("------------------------------------------------------------------- \n \n \n")

    if gpu:
        available = torch.cuda.is_available()
        print("GPU is available: " + str(available))
        if available:
            print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # Runs tokenize_and_align_labels across all documents in train, validation, and test set
    tokenized_dataset = split_dataset.map(tokenize_and_align_labels, batched=True)

    # creates the model to predict NER tags
    # label2id and id2label match the numerical labels to their corresponding strings
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
                                                            num_labels=len(label_list),
                                                            label2id={"Other": 0, "B-Vaccine": 1, "I-Vaccine": 2,
                                                                      "B-Strain": 3, "I-Strain": 4,
                                                                      "B-Vaccine Funder": 5, "I-Vaccine Funder": 6},
                                                            id2label={0: "Other", 1: "B-Vaccine", 2: "I-Vaccine",
                                                                      3: "B-Strain", 4: "I-Strain",
                                                                      5: "B-Vaccine Funder",
                                                                      6: "I-Vaccine Funder"})

    # sets the hyperparameters for training
    args = TrainingArguments("LitCOVID_NER_Model_Finetuned", evaluation_strategy="epoch", learning_rate=learning_rate,
                             per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                             num_train_epochs=num_epochs, weight_decay=weight_decay)

    # handles forming batches from the datasets
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # training arguments
    if not combined_metrics:
        trainer = Trainer(model, args, train_dataset=tokenized_dataset["train"],
                          eval_dataset=tokenized_dataset["validation"],
                          data_collator=data_collator, tokenizer=tokenizer, compute_metrics=compute_metrics)
    else:
        trainer = Trainer(model, args, train_dataset=tokenized_dataset["train"],
                          eval_dataset=tokenized_dataset["validation"],
                          data_collator=data_collator, tokenizer=tokenizer, compute_metrics=compute_combined_metrics)

    # train the model
    trainer.train()

    # calculate the performance on the test set at the end if combined_metrics is false
    if not combined_metrics:
        # Evaluates model on test set
        predictions, labels, metrics = trainer.predict(test_dataset=tokenized_dataset['test'])
        print("Performance on test set:")
        print(metrics)  # prints the model performance on the test set

    # save the model
    model.save_pretrained(save_directory)


if __name__ == "__main__":
    train_LitCovid_model(8, "LitCovid_preprocessed.json", 2e-5, 8, 0.01, "LitCovid_Pretrained_Model",
                         combined_dataset="LitCovid_combined.json")
