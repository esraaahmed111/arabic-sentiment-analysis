# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f' Device: {DEVICE}')
print(f' GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')


# Load data
df = pd.read_csv('arb_sentiment.csv')

df = df.rename(columns={'review_description': 'text', 'rating': 'sentiment'})
df = df[['text', 'sentiment', 'company']]


# EDA
label_names = {1: 'Positive', -1: 'Negative', 0: 'Neutral'}
df['sentiment_label'] = df['sentiment'].map(label_names)

print('Sentiment distribution:')
print(df['sentiment_label'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['sentiment_label'].value_counts().plot(
    kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c', '#3498db'])
axes[0].set_title('Sentiment Count', fontsize=13)
axes[0].tick_params(axis='x', rotation=0)

df['sentiment_label'].value_counts().plot(
    kind='pie', ax=axes[1], autopct='%1.1f%%',
    colors=['#2ecc71', '#e74c3c', '#3498db'])
axes[1].set_title('Sentiment %', fontsize=13)
axes[1].set_ylabel('')
plt.tight_layout()
plt.show()

# Reviews per company
plt.figure(figsize=(12, 4))
df['company'].value_counts().plot(kind='bar', color='#3498db', edgecolor='black')
plt.title('Reviews per Company', fontsize=13)
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Preprocessing
df = df.dropna(subset=['text', 'sentiment'])
print(f'After dropping nulls: {df.shape}')

def clean_arabic_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_arabic_text)
df = df[df['clean_text'].str.strip() != '']
print(f'After removing empty texts: {df.shape}')
df[['text', 'clean_text', 'sentiment_label']].head()


# Label Encoding & Train/Test Split
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])
readable_names = ['Negative', 'Neutral', 'Positive']
print('Label mapping:', dict(zip(le.classes_, range(len(le.classes_)))))

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label'])

print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')

# Class weights to handle imbalance (Neutral is only ~5%)
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f'Class weights: {dict(zip(readable_names, class_weights.round(3)))}')

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


# Model 1 TF-IDF + Logistic Regression
print("\n" + "="*55)
print("MODEL 1: TF-IDF + Logistic Regression")
print("="*55)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                        min_df=2, analyzer='char_wb')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f'TF-IDF matrix shape: {X_train_tfidf.shape}')

lr_model = LogisticRegression(max_iter=500, C=1.0, random_state=42,
                               class_weight='balanced')
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)
acc_lr    = accuracy_score(y_test, y_pred_lr)

print(f'\nAccuracy: {acc_lr*100:.2f}%\n')
print(classification_report(y_test, y_pred_lr, target_names=readable_names))
plot_confusion_matrix(y_test, y_pred_lr, readable_names,
                      'Confusion Matrix — TF-IDF + Logistic Regression')


# Model 2 Bidirectional LSTM
print("\n" + "="*55)
print("MODEL 2: Bidirectional LSTM")
print("="*55)

def build_vocab(texts, max_vocab=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

vocab   = build_vocab(X_train)
MAX_LEN = 50
print(f'Vocabulary size: {len(vocab):,}')

def text_to_seq(text, vocab, max_len=MAX_LEN):
    tokens = text.split()[:max_len]
    seq    = [vocab.get(t, 1) for t in tokens]
    seq   += [0] * (max_len - len(seq))
    return seq

X_train_seq = np.array([text_to_seq(t, vocab) for t in X_train])
X_test_seq  = np.array([text_to_seq(t, vocab) for t in X_test])
print(f'Sequence shape — Train: {X_train_seq.shape} | Test: {X_test_seq.shape}')


class TweetDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels.values, dtype=torch.long)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

BATCH        = 64
train_loader = DataLoader(TweetDataset(X_train_seq, y_train),
                          batch_size=BATCH, shuffle=True,
                          pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(TweetDataset(X_test_seq, y_test),
                          batch_size=BATCH,
                          pin_memory=torch.cuda.is_available())
print(f'Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')


class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                                  batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb           = self.dropout(self.embedding(x))
        _, (hidden,_) = self.lstm(emb)
        hidden        = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))

NUM_CLASSES = len(le.classes_)  # 3
lstm_model  = LSTMSentiment(len(vocab), 128, 256, NUM_CLASSES).to(DEVICE)
criterion   = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer   = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
print(lstm_model)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_correct, preds_all = 0, 0, []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for seqs, labels in loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            if is_train: optimizer.zero_grad()
            out  = model(seqs)
            loss = criterion(out, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss    += loss.item()
            total_correct += (out.argmax(1) == labels).sum().item()
            preds_all.extend(out.argmax(1).cpu().numpy())
    return total_loss/len(loader), total_correct/len(loader.dataset), preds_all

EPOCHS = 5
train_losses, val_accs = [], []

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc, _ = run_epoch(lstm_model, train_loader, optimizer)
    _,       vl_acc, _ = run_epoch(lstm_model, test_loader)
    train_losses.append(tr_loss)
    val_accs.append(vl_acc)
    print(f'Ep {ep}/{EPOCHS} | loss {tr_loss:.4f} | train {tr_acc:.4f} | val {vl_acc:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, marker='o', color='#e74c3c')
axes[0].set_title('LSTM Train Loss')
axes[0].set_xlabel('Epoch')
axes[1].plot(val_accs, marker='o', color='#2ecc71')
axes[1].set_title('LSTM Val Accuracy')
axes[1].set_xlabel('Epoch')
plt.tight_layout()
plt.show()

_, acc_lstm, y_pred_lstm = run_epoch(lstm_model, test_loader)
print(f'\nLSTM Accuracy: {acc_lstm*100:.2f}%\n')
print(classification_report(y_test, y_pred_lstm, target_names=readable_names))
plot_confusion_matrix(y_test, y_pred_lstm, readable_names, 'Confusion Matrix — BiLSTM')


# Model 3 AraBERT
print("\n" + "="*55)
print("MODEL 3: AraBERT")
print("="*55)

ARABERT_MODEL = 'aubmindlab/bert-base-arabertv02'
BERT_MAX_LEN  = 64
BERT_BATCH    = 32
BERT_EPOCHS   = 3
BERT_LR       = 2e-5

TRAIN_N = min(30000, len(X_train))
TEST_N  = min(8000,  len(X_test))

Xb_train = X_train.iloc[:TRAIN_N].reset_index(drop=True)
yb_train = y_train.iloc[:TRAIN_N].reset_index(drop=True)
Xb_test  = X_test.iloc[:TEST_N].reset_index(drop=True)
yb_test  = y_test.iloc[:TEST_N].reset_index(drop=True)

print(f'AraBERT train size: {len(Xb_train):,} | test size: {len(Xb_test):,}')

tokenizer = AutoTokenizer.from_pretrained(ARABERT_MODEL)
print('Tokenizer loaded.')


class ArabertDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = tokenizer(self.texts[i], max_length=BERT_MAX_LEN,
                        padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[i], dtype=torch.long)
        }

bert_train_loader = DataLoader(ArabertDataset(Xb_train, yb_train),
                                batch_size=BERT_BATCH, shuffle=True,
                                pin_memory=torch.cuda.is_available())
bert_test_loader  = DataLoader(ArabertDataset(Xb_test, yb_test),
                                batch_size=BERT_BATCH,
                                pin_memory=torch.cuda.is_available())
print(f'Train batches: {len(bert_train_loader)} | Test batches: {len(bert_test_loader)}')


bert_model = AutoModelForSequenceClassification.from_pretrained(
                 ARABERT_MODEL, num_labels=NUM_CLASSES).to(DEVICE)

bert_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer_b    = AdamW(bert_model.parameters(), lr=BERT_LR, eps=1e-8)
total_steps    = len(bert_train_loader) * BERT_EPOCHS
scheduler      = get_linear_schedule_with_warmup(
                     optimizer_b,
                     num_warmup_steps=total_steps // 10,
                     num_training_steps=total_steps)
print('AraBERT model ready.')


def bert_train_epoch(model, loader):
    model.train()
    total_loss, total_correct = 0, 0
    for batch in tqdm(loader, desc='Training', leave=False):
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labs = batch['label'].to(DEVICE)
        optimizer_b.zero_grad()
        out  = model(ids, attention_mask=mask)
        loss = bert_criterion(out.logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_b.step()
        scheduler.step()
        total_loss    += loss.item()
        total_correct += (out.logits.argmax(1) == labs).sum().item()
    return total_loss/len(loader), total_correct/len(loader.dataset)

def bert_eval(model, loader):
    model.eval()
    total_correct, preds_all = 0, []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labs = batch['label'].to(DEVICE)
            out  = model(ids, attention_mask=mask)
            p    = out.logits.argmax(1)
            total_correct += (p == labs).sum().item()
            preds_all.extend(p.cpu().numpy())
    return total_correct/len(loader.dataset), preds_all

print('Training AraBERT...\n')
for ep in range(1, BERT_EPOCHS + 1):
    tr_loss, tr_acc = bert_train_epoch(bert_model, bert_train_loader)
    vl_acc, _       = bert_eval(bert_model, bert_test_loader)
    print(f'Ep {ep}/{BERT_EPOCHS} | loss {tr_loss:.4f} | train {tr_acc:.4f} | val {vl_acc:.4f}')

bert_model.save_pretrained('./arabert_sentiment')
tokenizer.save_pretrained('./arabert_sentiment')
print('AraBERT model saved to ./arabert_sentiment\n')

acc_bert, y_pred_bert = bert_eval(bert_model, bert_test_loader)
print(f'AraBERT Accuracy: {acc_bert*100:.2f}%\n')
print(classification_report(list(yb_test), y_pred_bert, target_names=readable_names))
plot_confusion_matrix(list(yb_test), y_pred_bert, readable_names, 'Confusion Matrix — AraBERT')


# Model Comparison
models = ['TF-IDF + LogReg', 'BiLSTM', 'AraBERT']
accs   = [acc_lr*100, acc_lstm*100, acc_bert*100]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accs, color=['#3498db', '#e67e22', '#9b59b6'],
               width=0.5, edgecolor='black')
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')
plt.ylim(0, 100)
plt.title('Model Accuracy Comparison', fontsize=15)
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.show()

print('\n Final Results')
print('-'*35)
for m, a in zip(models, accs):
    print(f'  {m:<22}: {a:.2f}%')
print(f'\n Best: {models[accs.index(max(accs))]} ({max(accs):.2f}%)')


# Inference
def predict_tfidf(text):
    vec   = tfidf.transform([clean_arabic_text(text)])
    pred  = lr_model.predict(vec)[0]
    proba = lr_model.predict_proba(vec)[0]
    return le.classes_[pred], {c: f"{p:.2%}" for c,p in zip(le.classes_, proba)}

def predict_lstm(text):
    seq = torch.tensor([text_to_seq(clean_arabic_text(text), vocab)],
                        dtype=torch.long).to(DEVICE)
    lstm_model.eval()
    with torch.no_grad():
        out   = lstm_model(seq)
        proba = torch.softmax(out, 1)[0].cpu().numpy()
    return le.classes_[out.argmax(1).item()], {c: f"{p:.2%}" for c,p in zip(le.classes_, proba)}

def predict_arabert(text):
    enc = tokenizer(clean_arabic_text(text), max_length=BERT_MAX_LEN,
                    padding='max_length', truncation=True, return_tensors='pt')
    bert_model.eval()
    with torch.no_grad():
        out   = bert_model(enc['input_ids'].to(DEVICE),
                           attention_mask=enc['attention_mask'].to(DEVICE))
        proba = torch.softmax(out.logits, 1)[0].cpu().numpy()
    return le.classes_[out.logits.argmax(1).item()], {c: f"{p:.2%}" for c,p in zip(le.classes_, proba)}


sample_texts = [
    "التطبيق رائع جداً وسهل الاستخدام وأنصح به الجميع",
    "تجربة سيئة جداً والتطبيق لا يعمل ولن أستخدمه مرة أخرى",
    "التطبيق مقبول ويؤدي الغرض أحياناً",
    "خدمة ممتازة والتوصيل كان سريع جداً شكراً",
    "أسوأ تجربة في حياتي المنتج لا يساوي شيء",
]

print("\n Live Predictions\n" + "="*65)
for text in sample_texts:
    print(f"\n {text}")
    for name, fn in [("TF-IDF+LR",  predict_tfidf),
                     ("BiLSTM",      predict_lstm),
                     ("AraBERT",     predict_arabert)]:
        label, probs = fn(text)
        print(f"   {name:<12}: {label:<10} | {probs}")
print("\n" + "="*65)
