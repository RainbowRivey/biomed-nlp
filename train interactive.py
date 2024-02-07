#%%
import argparse

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("dataset", type=str)
parser.add_argument("epochs", type=int)
args = parser.parse_args()


#%% To start in interactive mode without argparse
class args:
    ...
args.dataset = "SETH"
args.epochs = 5
#%%
from seqeval.metrics import classification_report
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,  DataCollatorForTokenClassification, get_scheduler
import evaluate
import time
import itertools
import datetime
import json
import torch
from tqdm.auto import tqdm
from numpyencoder import NumpyEncoder
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from sklearn.model_selection import train_test_split


#%%
modelCheckpoint = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
dataset = args.dataset # SETH tmvar Variome Variome120 amia
path = f"./_{dataset}-custom"
tokenizer = AutoTokenizer.from_pretrained(modelCheckpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")

training_args = TrainingArguments(
    output_dir=path,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    save_total_limit=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    optim="adamw_torch"
)


######################################
#  Read IOB
#%%

def convertToCorpus(inputString):
    documents = []
    document = None
    for line in inputString:
        if line.startswith("#"):
            if document:
                documents.append(document)
            document = {}
            document["id"] = line
            document["tokens"] = []
            document["str_tags"] = []
        else:
            iob = line.rsplit(",", 1)
            if len(iob) == 2:
                document["tokens"].append(iob[0])
                document["str_tags"].append(iob[1])
            else:
                print(line)
    return documents


print("Loading datasets")
# for local copies
# with open(f"./IOB/{dataset}-train.iob", "r") as train:
#     trainFile = train.read().split('\n')
#     trainFile.pop(0)
#     trainCorpus = convertToCorpus(trainFile)
# with open(f"./IOB/{dataset}-test.iob", "r") as test:
#     testFile = test.read().split('\n')
#     testFile.pop(0)
#     testCorpus = convertToCorpus(testFile)

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
val = int(len(trainCorpus)*0.9)
trainCorpus, validationCorpus = trainCorpus[:val], trainCorpus[val:]
validationCorpus
#%%

label_list = sorted(list(
    set(list(itertools.chain(*list(map(lambda x: x["str_tags"], trainCorpus)))))))
label2id = dict(zip(label_list, range(len(label_list))))
id2label = {v: k for k, v in label2id.items()}


#%% Defs
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
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#%%
for document in trainCorpus:
    document["ner_tags"] = list(
        map(lambda x: label2id[x], document["str_tags"]))

for document in testCorpus:
    document["ner_tags"] = list(
        map(lambda x: label2id[x], document["str_tags"]))
    

fullData = DatasetDict({
    'train': Dataset.from_pandas(trainCorpus),
    'validation': Dataset.from_pandas(validationCorpus),
    'test': Dataset.from_pandas(testCorpus)
})


#
tokenized_datasets = fullData.map(
    tokenize_and_align_labels,
    batched=True
)



# Load clear model
print("Loading base model")
model = AutoModelForTokenClassification.from_pretrained(
    modelCheckpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
# OR load from checkpoint
# model = AutoModelForTokenClassification.from_pretrained(
#     outputModel, num_labels=len(id2label), id2label=id2label, label2id=label2id)


#========================Custom training loop====================#

#%%
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
## optimizer & accelerator
optimizer = AdamW(model.parameters(), lr=2e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

## scheduler 
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        # true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions_gathered, labels_gathered)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions_gathered, labels_gathered)
        ]

    results = compute_metrics(true_predictions, true_labels)
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )
#======================Default training loop========================#
#%%
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# print("Start training")
# start = time.time()
# trainer.train()
# print("Finished after " + str(datetime.timedelta(seconds=round(time.time() - start))))

# trainer.save_model(f"{path}/final")

#========================Post-process===============================#
#%%
df = pd.DataFrame(trainer.state.log_history)
df.to_json(f"{path}/log_history.json")

df = df[df.eval_runtime.notnull()]
loss_fig = df.plot(x='epoch', y=['eval_loss'], kind='bar', title = f'{dataset} loss')
loss_fig.figure.savefig(f'{path}/loss.png')

eval_fig = df.plot(x='epoch', y=['eval_precision', 'eval_recall',
                                 'eval_f1'], kind='bar', figsize=(15, 9), title = f'{dataset} eval')
eval_fig.figure.savefig(f'{path}/eval.png')


predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = seqeval.compute(predictions=true_predictions, references=true_labels)
results

with open(f"{path}/perfomance.json", "w") as f:
  json.dump(results, f, cls=NumpyEncoder)
