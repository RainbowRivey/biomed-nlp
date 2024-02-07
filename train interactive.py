# %% Imports
from seqeval.metrics import classification_report
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,  DataCollatorForTokenClassification
import evaluate
import time
import itertools
import evaluate
import datetime
import torch


# %%
modelCheckpoint = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
outputModel = "./seth-finetune"
dataset = "amia"
tokenizer = AutoTokenizer.from_pretrained(modelCheckpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")
######################################
# %% Read IOB


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


with open(f"./IOB/{dataset}-train.iob", "r") as train:
    trainFile = train.read().split('\n')
    trainFile.pop(0)
    trainCorpus = convertToCorpus(trainFile)
with open(f"./IOB/{dataset}-test.iob", "r") as test:
    testFile = test.read().split('\n')
    testFile.pop(0)
    testCorpus = convertToCorpus(testFile)


# %%
label_list = sorted(list(
    set(list(itertools.chain(*list(map(lambda x: x["str_tags"], trainCorpus)))))))
label2id = dict(zip(label_list, range(len(label_list))))
id2label = {v: k for k, v in label2id.items()}


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


# %%
for document in trainCorpus:
    document["ner_tags"] = list(
        map(lambda x: label2id[x], document["str_tags"]))

for document in testCorpus:
    document["ner_tags"] = list(
        map(lambda x: label2id[x], document["str_tags"]))
# %%
fullData = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(data=trainCorpus)),
    'test': Dataset.from_pandas(pd.DataFrame(data=testCorpus))
})

# %% Test
###############################
document = fullData["train"][0]

tokenized_input = tokenizer(document["tokens"], is_split_into_words=True)
labels = document["ner_tags"]

tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
# print("Original-sentence:\t\t" +" ".join(document['tokens']))
# print("Transformer representation:\t" +" ".join(tokens))

word_ids = tokenized_input.word_ids()
# print(labels)
# print(align_labels_with_tokens(labels, word_ids))
# TODO Add here code to show how result would look like


# %%
tokenized_datasets = fullData.map(
    tokenize_and_align_labels,
    batched=True
)

# %% Load clear model
model = AutoModelForTokenClassification.from_pretrained(
    modelCheckpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
#%% OR load from checkpoint
model = AutoModelForTokenClassification.from_pretrained(
    outputModel, num_labels=len(id2label), id2label=id2label, label2id=label2id)
# %% TRAIN
training_args = TrainingArguments(
    output_dir=outputModel,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
start = time.time()
trainer.train()
print("Finished after " + str(datetime.timedelta(seconds=round(time.time() - start))))



# %%
pd.DataFrame(trainer.state.log_history).head(5)
df = pd.DataFrame(trainer.state.log_history)
df = df[df.eval_runtime.notnull()]
df.plot(x='epoch', y=['eval_loss'], kind='bar')
df.plot(x='epoch', y=['eval_precision', 'eval_recall',
        'eval_f1'], kind='bar', figsize=(15, 9))


# #%% TEST
# text = "Identification of four novel mutations in the factor VIII gene: three missense mutations (E1875G, G2088S, I2185T) and a 2-bp deletion (1780delTC)."
# inputs = tokenizer(text, return_tensors="pt")
# print(inputs)
# inputs.to(0)
# with torch.no_grad():
#   logits = model(**inputs).logits
#   # print(logits)

# predictions = torch.argmax(logits, dim=2)
# predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
# for token, label in zip(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), predicted_token_class):
#     print(token, label)

# # from transformers import pipeline
# # clf = pipeline("token-classification", model, tokenizer=tokenizer, device=0)
# # answer = clf(text)
# # print(answer)

#%%
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
# %%


trainer.save_model("seth-finetune-4ep")
# %%