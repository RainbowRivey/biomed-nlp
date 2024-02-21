import argparse

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("dataset", type=str)
parser.add_argument("sweeps", type=int, default="5")
args = parser.parse_args()


# #%% To start in interactive mode without argparse
# class args:
#     ...
# args.dataset = "SETH"
# args.epochs = 5

from seqeval.metrics import classification_report
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,  DataCollatorForTokenClassification
import evaluate
import time
import itertools
import datetime
import json
from tqdm.auto import tqdm
from numpyencoder import NumpyEncoder
import wandb


with open('wandb.key', 'r') as keyFile:
    WANDB_API_KEY = keyFile.readline().rstrip()
wandb.login(key=WANDB_API_KEY)

modelCheckpoint = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
dataset = args.dataset
path = f"./_{dataset}"
tokenizer = AutoTokenizer.from_pretrained(modelCheckpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")

def convertToCorpus(inputString):
        documents = []
        document = None
        subdoc=0
        for line in inputString:
            if line.startswith("#"):
                subdoc = 0
                doc_id = line
                if document:
                    documents.append(document)
                document = {}
                document["id"] = doc_id
                document["tokens"] = []
                document["str_tags"] = []
            else:
                iob = line.rsplit(",", 1)
                if len(iob) == 2:
                    if iob[0] in ".!?" and len(document['tokens'])>200:
                        document["tokens"].append(iob[0])
                        document["str_tags"].append(iob[1])
                        documents.append(document)
                        subdoc+=1
                        document = {}
                        document["id"] = f"{doc_id}-{subdoc}"
                        document["tokens"] = []
                        document["str_tags"] = []
                        continue

                    document["tokens"].append(iob[0])
                    document["str_tags"].append(iob[1])
                else:
                    print(line)
        return documents

#### Load datasets
import requests
f = requests.get(f"https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/{dataset}-train.iob")
trainFile = f.text.split("\n")
trainFile.pop(0) #Remove first element
trainCorpus = convertToCorpus(trainFile)
f = requests.get(f"https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/{dataset}-test.iob")
testFile = f.text.split("\n")
testFile.pop(0) #Remove first element
testCorpus = convertToCorpus(testFile)
del(testFile, trainFile)


label_list = sorted(list(
    set(list(itertools.chain(*list(map(lambda x: x["str_tags"], trainCorpus)))))))
label2id = dict(zip(label_list, range(len(label_list))))
id2label = {v: k for k, v in label2id.items()}

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions, references=true_labels)
    return results

def align_labels_with_tokens(labels, word_ids, id2label=id2label, label2id=label2id):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if (id2label[label].startswith("B-")):
                    label = label2id[id2label[label].replace("B-", "I-")]
                new_labels.append(label)

        return new_labels



    




for document in trainCorpus:
    document["ner_tags"] = list(
        map(lambda x: label2id[x], document["str_tags"]))

for document in testCorpus:
    document["ner_tags"] = list(
        map(lambda x: label2id[x], document["str_tags"]))

fullData = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(data=trainCorpus)),
    'test': Dataset.from_pandas(pd.DataFrame(data=testCorpus))
})


#
tokenized_datasets = fullData.map(
    tokenize_and_align_labels,
    batched=True
)

def train(config):
    training_args = TrainingArguments(
        output_dir=path,
        learning_rate= config.learning_rate, #2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay, #0.01,
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        optim="adamw_torch"
    )
    model = AutoModelForTokenClassification.from_pretrained(
        modelCheckpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    # OR load from checkpoint
    # model = AutoModelForTokenClassification.from_pretrained(
    #     outputModel, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    #  TRAIN

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    #
    print("Start training")
    start = time.time()
    trainer.train()
    print("Finished after " + str(datetime.timedelta(seconds=round(time.time() - start))))

    trainer.save_model(f"{path}/final")

    #========================Post-process===============================#

    df = pd.DataFrame(trainer.state.log_history)
    df.to_json(f"{path}/log_history.json")

    df = df[df.eval_runtime.notnull()]
    loss_fig = df.plot(x='epoch', y=['eval_loss'], kind='bar', title = f'{dataset} loss')
    loss_fig.figure.savefig(f'{path}/loss.png')

    eval_fig = df.plot(x='epoch', y=['eval_precision', 'eval_recall',
                                    'eval_f1'], kind='bar', figsize=(15, 9), title = f'{dataset} eval')
    eval_fig.figure.savefig(f'{path}/eval.png')


    # predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    # predictions = np.argmax(predictions, axis=2)

    # # Remove ignored index (special tokens)
    # true_predictions = [
    #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]

    # results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # results
    results = compute_metrics(trainer.predict(tokenized_datasets["test"]))
    with open(f"{path}/perfomance.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder)
    return results




# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    wandb.init(project=f"biomed-{dataset}")
    score = train(wandb.config)
    wandb.log({"score": score})

# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "eval/f1"},
    "parameters": {
        # "doc_len" : {"max": 512, "min": 100},
        "learning_rate" : {"max":2e-4, "min":2e-6},
        "num_epochs" : {"values": [5]},
        "weight_decay": {"values":[0, 0.1, 0.01]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"biomed-{dataset}")

wandb.agent(sweep_id, function=main, count=args.sweeps)
