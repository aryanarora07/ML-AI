# 1. Environment Setup
# Install required libraries
!pip install transformers datasets torch evaluate accelerate

# Verify GPU availability
import torch

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# 2. Preprocessing Data
from datasets import load_dataset
from transformers import AutoTokenizer

# Load IMDB dataset
imdb_dataset = load_dataset("imdb")
print(imdb_dataset)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize function with truncation and padding
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization to dataset
tokenized_imdb = imdb_dataset.map(tokenize_function, batched=True)

# Rename column and set format for PyTorch
tokenized_imdb = tokenized_imdb.rename_column("label", "labels")
tokenized_imdb.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print(f"Training examples: {len(tokenized_imdb['train'])}")
print(f"Test examples: {len(tokenized_imdb['test'])}")

# 3. Model Training
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)

# Define metrics for evaluation
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# 4. Save and Evaluate
# Save the model
trainer.save_model("./distilbert-imdb-sentiment")

# Evaluate on test set
test_results = trainer.evaluate()
print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")

# Inference example
from transformers import pipeline

# Load the fine-tuned model for inference
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="./distilbert-imdb-sentiment",
    tokenizer=tokenizer
)

# Test with sample reviews
sample_reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "What a waste of time. Terrible acting and a boring plot."
]

results = sentiment_analyzer(sample_reviews)
for review, result in zip(sample_reviews, results):
    print(f"Review: {review}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}\n")
