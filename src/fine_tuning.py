from types import SimpleNamespace

from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer

settings = SimpleNamespace(**{
    "re_train_net": True,
    # "model_names": ['distilbert-base-cased',
    #                 'distilbert-base-uncased',
    #                 'bert-base-cased',
    #                 'bert-base-uncased',
    #                 'distilroberta-base']
    # "model_name": 'distilbert-base-cased'
    # "model_name": 'distilbert-base-uncased'
    # "model_name": 'bert-base-cased'
    "model_name": 'bert-base-uncased'
    # "model_name": 'distilroberta-base'
})


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
        num_labels=5
    )

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
        save_strategy="no"
        #no_cuda=True
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    return trainer
