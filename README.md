End-to-end NLP pipeline that classifies Arabic company reviews as **Positive**, **Negative**, or **Neutral**
using three models of increasing complexity on **40,000+ real Arabic reviews**.

## Models Used

| Model | Approach | Library |
|---|---|---|
| TF-IDF + Logistic Regression | Traditional ML | Scikit-learn |
| Bidirectional LSTM | Deep Learning | PyTorch |
| AraBERT | Transformer | HuggingFace |

# Results

| Model | Accuracy |
|---|---|
| TF-IDF + LogReg | ~70% |
| BiLSTM | ~74% |
| AraBERT | ~82% |
