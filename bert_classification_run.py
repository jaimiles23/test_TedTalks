import random
import numpy as np
import os
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertTokenizer, BertConfig, WarmupLinearSchedule
import re
import pandas as pd
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score, roc_curve
import timeit
from datetime import date
import matplotlib.pyplot as plt

start = timeit.default_timer()


# Your statements here
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


# For reproducible results
topic = 'technology'
seed_everything()
SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
TRAIN_FILE_PATH = "2024-01-21_data.json"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 24
NUM_EPOCHS = 40  #
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 3
THRESHOLD = 0.50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


class BertClassifier(nn.Module):

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        # Binary classification problem (num_labels = 2)
        self.num_labels = config.num_labels
        # Pre-trained BERT model
        self.bert = BertModel(config)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # A single layer classifier added on top of BERT to fine tune for binary classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Weight initialization
        torch.nn.init.xavier_normal_(self.classifier.weight)
        # # activation for classifier
        # self.softmax = nn.Softmax(dim=1)
        # # setting threshold
        # self.threshold = nn.Threshold(THRESHOLD,0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        # Forward pass through pre-trained BERT
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # Last layer output (Total 12 layers)
        pooled_output = outputs[-1]

        pooled_output = self.dropout(pooled_output)

        pooled_output = self.classifier(pooled_output)

        # pooled_output = self.softmax(pooled_output)

        # pooled_output = self.threshold(pooled_output)

        return pooled_output


class SequenceDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer, regex_transformations={}):
        # Read JSON file and assign to headlines variable (list of strings)
        df = pd.read_json(dataset_file_path, lines=True)
        # df = df.drop(['article_link'], axis=1)
        self.headlines = df.values
        # Regex Transformations can be used for data cleansing.
        # e.g. replace
        #   '\n' -> ' ',
        #   'wasn't -> was not
        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, index):
        is_business, is_entertainment, is_politics, is_technology, headline = \
        self.headlines[index]
        for regex, value_to_replace_with in self.regex_transformations.items():
            headline = re.sub(regex, value_to_replace_with, headline)

        # Convert input string into tokens with the special BERT Tokenizer which can handle out-of-vocabulary words using subgrams
        # e.g. headline = Here is the sentence I want embeddings for.
        #      tokens = [here, is, the, sentence, i, want, em, ##bed, ##ding, ##s, for, .]
        if len(headline) > MAX_SEQ_LENGTH:
            headline = headline[:MAX_SEQ_LENGTH]
        tokens = self.tokenizer.tokenize(headline)

        # Add [CLS] at the beginning and [SEP] at the end of the tokens list for classification problems
        tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
        # Convert tokens to respective IDs from the vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Segment ID for a single sequence in case of classification is 0.
        segment_ids = [0] * len(input_ids)

        # Input mask where each valid token has mask = 1 and padding has mask = 0
        input_mask = [1] * len(input_ids)

        # padding_length is calculated to reach max_seq_length
        padding_length = MAX_SEQ_LENGTH - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        input_mask = input_mask + [0] * padding_length
        segment_ids = segment_ids + [0] * padding_length

        assert len(input_ids) == MAX_SEQ_LENGTH
        assert len(input_mask) == MAX_SEQ_LENGTH
        assert len(segment_ids) == MAX_SEQ_LENGTH

        return (torch.tensor(input_ids, dtype=torch.long, device=DEVICE), \
                torch.tensor(segment_ids, dtype=torch.long, device=DEVICE), \
                torch.tensor(input_mask, device=DEVICE), \
                torch.tensor(is_business, dtype=torch.long, device=DEVICE), \
                torch.tensor(is_entertainment, dtype=torch.long, device=DEVICE), \
                torch.tensor(is_politics, dtype=torch.long, device=DEVICE), \
                torch.tensor(is_technology, dtype=torch.long, device=DEVICE))


def make_weights_for_balanced_classes(txts, nclasses, class_name):
    n_txts = len(txts)
    count_technology = [0] * nclasses
    count_business = [0] * nclasses
    count_entertainment = [0] * nclasses
    count_politics = [0] * nclasses
    for is_business, is_entertainment, is_politics, is_technology, _ in txts:
        count_business[is_business] += 1
        count_entertainment[is_entertainment] += 1
        count_politics[is_politics] += 1
        count_technology[is_technology] += 1

    weight_technology = [0.] * nclasses
    weight_business = [0.] * nclasses
    weight_entertainment = [0.] * nclasses
    weight_politics = [0.] * nclasses
    for i in range(nclasses):
        weight_technology[i] = float(n_txts) / float(count_technology[i])
        weight_business[i] = float(n_txts) / float(count_business[i])
        weight_entertainment[i] = float(n_txts) / float(count_entertainment[i])
        weight_politics[i] = float(n_txts) / float(count_politics[i])
    weights = [0] * n_txts
    for idx, (is_business, is_entertainment, is_politics, is_technology, txt) in enumerate(txts):
        if class_name == 'technology':
            weights[idx] = weight_technology[is_technology]
        elif class_name == 'business':
            weights[idx] = weight_business[is_business]
        elif class_name == 'entertainment':
            weights[idx] = weight_entertainment[is_entertainment]
        elif class_name == 'politics':
            weights[idx] = weight_politics[is_politics]
    return weights


# Load BERT default config object and make necessary changes as per requirement
config = BertConfig(hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    num_labels=2)

# Create our custom BERTClassifier model object
model = BertClassifier(config)
# model = BertModel.from_pretrained('bert-base-uncased')
model.to(DEVICE)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_val_dataset = SequenceDataset(TRAIN_FILE_PATH, tokenizer)

validation_split = 0.2
dataset_size = len(train_val_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
shuffle_dataset = True

if shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
txts = train_val_dataset.headlines
weights = make_weights_for_balanced_classes(txts, config.num_labels, class_name=topic)
train_weights, val_weights = weights[split:], weights[:split]
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
validation_sampler = WeightedRandomSampler(val_weights, len(val_weights))
train_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

print('Training Set Size {}, Validation Set Size {}'.format(len(train_indices), len(val_indices)))

criterion = nn.CrossEntropyLoss()

# Adam Optimizer with very small learning rate given to BERT
optimizer = torch.optim.Adam([
    {'params': model.bert.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 3e-4}
])

# Learning rate scheduler
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEPS,
                                 t_total=len(train_loader) // GRADIENT_ACCUMULATION_STEPS * NUM_EPOCHS)

model.zero_grad()
epoch_iterator = trange(int(NUM_EPOCHS), desc="Epoch")
training_acc_list, validation_acc_list = [], []

for epoch in epoch_iterator:
    epoch_loss = 0.0
    train_correct_total = 0

    # Training Loop
    train_iterator = tqdm(train_loader, desc="Train Iteration")
    for step, batch in enumerate(train_iterator):
        model.train(True)
        # Here each element of batch list refers to one of [input_ids, segment_ids, attention_mask, labels]
        inputs = {
            'input_ids': batch[0].to(DEVICE),
            'token_type_ids': batch[1].to(DEVICE),
            'attention_mask': batch[2].to(DEVICE)
        }

        if topic == 'business':
            labels = batch[3].to(DEVICE)
        elif topic == 'entertainment':
            labels = batch[4].to(DEVICE)
        elif topic == 'politics':
            labels = batch[5].to(DEVICE)
        elif topic == 'technology':
            labels = batch[6].to(DEVICE)
        logits = model(**inputs)

        loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        epoch_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scheduler.step()
            optimizer.step()
            model.zero_grad()

        _, predicted = torch.max(logits.data, 1)
        correct_reviews_in_batch = (predicted == labels).sum().item()
        train_correct_total += correct_reviews_in_batch
    print()
    print("-------------------------------------------------------------------------------------")
    print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

    # Validation Loop
    with torch.no_grad():
        val_correct_total = 0
        model.train(False)
        val_iterator = tqdm(val_loader, desc="Validation Iteration")
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        preds_list = []
        labels_list = []
        preds_prob_list = []
        for step, batch in enumerate(val_iterator):
            inputs = {
                'input_ids': batch[0].to(DEVICE),
                'token_type_ids': batch[1].to(DEVICE),
                'attention_mask': batch[2].to(DEVICE)
            }

            if topic == 'business':
                labels = batch[3].to(DEVICE)
            elif topic == 'entertainment':
                labels = batch[4].to(DEVICE)
            elif topic == 'politics':
                labels = batch[5].to(DEVICE)
            elif topic == 'technology':
                labels = batch[6].to(DEVICE)
            logits = model(**inputs)

            softmax = nn.Softmax(dim=1)
            predicted_prob = softmax(logits.data)
            _, predicted = torch.max(logits.data, 1)
            preds_prob_list_sub = []
            for idx, i in enumerate(labels):
                preds_prob_list_sub.append(predicted_prob[idx][1])
            correct_reviews_in_batch = (predicted == labels).sum().item()
            val_correct_total += correct_reviews_in_batch

            tp, fp, tn, fn = confusion(predicted, labels)
            true_positives += tp
            false_positives += fp
            true_negatives += tn
            false_negatives += fn
            preds_list += predicted.tolist()
            labels_list += labels.tolist()
            preds_prob_list += preds_prob_list_sub

        training_acc_list.append(train_correct_total * 100 / len(train_indices))
        validation_acc_list.append(val_correct_total * 100 / len(val_indices))

        '''
        Please calculate recall, precision, false_positive_rate, false_negative_rate and f1_score
        '''
        recall = float("nan")
        precision = float("nan")
        false_positive_rate = float("nan")
        false_negative_rate = float("nan")
        f1_score = float("nan")

        auc = roc_auc_score(labels_list, preds_list)

        print('Training Accuracy {:.4f}% - Validation Accurracy {:.4f}%'.format(
            train_correct_total * 100 / len(train_indices), val_correct_total * 100 / len(val_indices)))
        print("Precision {:.4f} - Recall {:.4f} - F1-score - {:.4f}".format(precision, recall, f1_score))
        print("False Positive Rate {:.4f} - False Negative Rate {:.4f}".format(false_positive_rate, false_negative_rate))
        print("roc_auc_score {:.4f}".format(auc))

    print("-------------------------------------------------------------------------------------")

stop = timeit.default_timer()
torch.save(model.state_dict(),
           "./ratio_adjusted_mybertmodel_" + str(topic) + "_" + str(date.today()) + "_epoch_" + str(
               NUM_EPOCHS) + "BATCH_SIZE" + str(BATCH_SIZE) + "_stop" + str(stop) + "_.pth")

print('Running Time: ', stop - start, 's | ', (stop - start) / 60, "min")
# make roc curve plot
fpr, tpr, _ = roc_curve(torch.tensor(labels_list).detach().cpu().numpy(),
                        torch.tensor(preds_prob_list).detach().cpu().numpy())
auc = roc_auc_score(torch.tensor(labels_list).detach().cpu().numpy(),
                    torch.tensor(preds_prob_list).detach().cpu().numpy())
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.savefig('validation_roc_' + str(topic) + '_.png')
plt.show()