import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
OUTPUT_DIR = "./results"
SAVE_DIR = "./fine_tuned_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

def load_sample_data():
    texts = [
        "I love this product, it works great!",
        "This service is terrible, would not recommend.",
        "The item arrived on time and as described.",
        "Very disappointed with the quality of this item.",
    ]
    labels = [1, 0, 1, 0]
    return texts, labels

def preprocess_data(texts, labels):
    processed_texts = []
    for text in texts:
        text = text.lower()
        text = " ".join(text.split())
        processed_texts.append(text)
    return processed_texts, labels

def create_dataset(texts, labels):
    dataset_dict = {
        "text": texts,
        "label": labels
    }
    return Dataset.from_dict(dataset_dict)

texts, labels = load_sample_data()
processed_texts, labels = preprocess_data(texts, labels)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = create_dataset(train_texts, train_labels)
test_dataset = create_dataset(test_texts, test_labels)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])
tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")
tokenized_train_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting model training...")
trainer.train()

print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

print("Generating detailed classification report...")
predictions = trainer.predict(tokenized_test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = tokenized_test_dataset["labels"]

print(classification_report(y_true, y_pred))

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model and tokenizer saved to {SAVE_DIR}")

def predict_sentiment(text):
    text = text.lower()
    text = " ".join(text.split())
    inputs = tokenizer(
        text, 
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    label_map = {0: "Negative", 1: "Positive"}
    confidence = predictions[0][predicted_class].item() * 100
    return label_map[predicted_class], confidence

test_examples = [
    "The customer service was exceptional and they resolved my issue quickly.",
    "The product broke after just one week of use."
]

print("\nTesting the model on new examples:")
for example in test_examples:
    sentiment, confidence = predict_sentiment(example)
    print(f"Text: {example}")
    print(f"Predicted sentiment: {sentiment} (Confidence: {confidence:.2f}%)")
    print("-" * 50)

print("\nFine-tuning process completed!")
