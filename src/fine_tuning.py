from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer, IntervalStrategy, \
    EarlyStoppingCallback

validation_set_size = SimpleNamespace(**{
    "xs": 10,
    "s": 100,
    "m": 1000,
    "l": 10000,
    "xl": 100000,
})

settings = SimpleNamespace(**{
    "re_train_net": False,
    "seed": 42,
    "validation_size": validation_set_size.xl,
    "train_test_splitfactor": 0.7,
    "test_eval_splitfactor": 0.5,
    "use_hard_validation_size": False,
    "eval_for_comparing": False,
    "early_stopping": True,
    "num_epochs": 10,
    # "review_lengths": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 750000],
    "review_lengths": [750000],
    "model_names": [
       # 'bert-base-cased',
       # 'distilbert-base-cased',
       # 'distilbert-base-uncased',
        'distilroberta-base'
    ],
    # "model_names": ['distilbert-base-cased'],
    # "model_names": ['distilbert-base-uncased'],
    # "model_names": ['bert-base-cased'],
    # "model_names": ['bert-base-uncased'],
    # "model_names": 'distilroberta-base',
})


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1).astype(float).tolist()
    labels = np.argmax(labels, axis=1).astype(float).tolist()
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    mse = mean_squared_error(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "mse": mse, "f1": f1}


def get_trainer(
        model_name,
        train_dataset,
        val_dataset,
        save_path,
        learning_rate=2e-5,
        logging_steps=100,
        auto_find_batch_size=True,
        num_train_epochs=3,
        weight_decay=0.01,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type='multi_label_classification',
        num_labels=5,
    )

    if settings.early_stopping:
        args = TrainingArguments(
            output_dir=save_path,
            evaluation_strategy=IntervalStrategy.STEPS,  # "steps"
            eval_steps=500,  # Evaluation and Save happens every 50 steps
            save_total_limit=1,  # Only last 5 models are saved. Older ones are deleted.
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            push_to_hub=False,
            metric_for_best_model='f1',
            auto_find_batch_size=auto_find_batch_size,
            load_best_model_at_end=True,
            logging_dir="./logs"
        )
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    else:
        args = TrainingArguments(
            output_dir=save_path,  # output directory
            num_train_epochs=num_train_epochs,  # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=16,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=weight_decay,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            auto_find_batch_size=auto_find_batch_size,
            save_strategy="no",
            # no_cuda=True
        )
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
        )

    return trainer
