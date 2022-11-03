# %%
# do imports
import datetime
import time

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    mean_squared_error
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import preprocessing
from fine_tuning import settings
from helpers import print_task_header, ReviewDataset, printProgressBar, make_predictions, get_cm_as_dict, \
    print_subtask_header, read_base_data, get_class_report_as_dict, sample_random_points, write_dict_to_json, \
    save_confusion_matrix, create_dir_if_not_exist
from src import fine_tuning, helpers

review_lengths = settings.review_lengths

for length in review_lengths:
    try:
        model_names = settings.model_names
    except AttributeError:
        print("Couldn't find a model to train")
        break

    for model_name in model_names:

        title = f"Fine-tuning and classification for {model_name} with {review_lengths} data points"
        print_task_header(title, len(title) + 4, True)
        books_data = read_base_data(f'../data/Reviews_data/reviews750000.csv')

        # step: clear cuda cache
        title = f'Clearing cuda cache'
        print_subtask_header(title, len(title) + 4, True, 1)
        t_start = time.time()
        torch.cuda.empty_cache()
        t_end = time.time()
        t_passed = round(t_end - t_start, 2)
        print(f"Done in: {t_passed}s")

        # step: split data
        title = f'Splitting data into train, test, eval'
        print_subtask_header(title, len(title) + 4, True, 1)
        t_start = time.time()
        # index to split data on

        test_data_changed = False
        validation_base_size = int(length * (1 - settings.train_test_splitfactor) * settings.test_eval_splitfactor)
        train_and_eval_size = length - validation_base_size
        if validation_base_size < settings.validation_size or settings.use_hard_validation_size:
            books_data = sample_random_points(books_data, train_and_eval_size + settings.validation_size,
                                              seed=settings.seed)
            index = int(length * settings.train_test_splitfactor)
            validation_index = index + validation_base_size
            test_data_changed = True
            train_texts, train_labels = books_data[:index].review.tolist(), books_data[:index].score.tolist()
            test_texts, test_labels = books_data[validation_index:].review.tolist(), \
                                      books_data[validation_index:].score.tolist()
            val_texts, val_labels = books_data[index:validation_index].review.tolist(),\
                                    books_data[index:validation_index].score.tolist()


        else:
            books_data = sample_random_points(books_data, length, seed=settings.seed)
            index = int(len(books_data) * settings.train_test_splitfactor)
            train_texts, train_labels = books_data[:index].review.tolist(), books_data[:index].score.tolist()
            test_data, test_labels = books_data[index:].review.tolist(), books_data[index:].score.tolist()
            test_texts, val_texts, test_labels, val_labels = train_test_split(test_data, test_labels,
                                                                              test_size=settings.test_eval_splitfactor)

        t_end = time.time()
        t_passed = round(t_end - t_start, 2)
        print(f"Done in: {t_passed}s")

        train_length = len(train_labels)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except EnvironmentError:
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        title = f'Tokenization'
        print_subtask_header(title, len(title) + 4, True, 1)
        train_encodings = helpers.tokenize(tokenizer, train_texts, "train_encodings")
        val_encodings = helpers.tokenize(tokenizer, val_texts, "val_encodings")
        test_encodings = helpers.tokenize(tokenizer, test_texts, "test_encodings")

        title = f'Dataset creation'
        print_subtask_header(title, len(title) + 4, True, 1)
        t_start = time.time()
        train_dataset = ReviewDataset(train_encodings, preprocessing.one_hot_encode(train_labels))
        val_dataset = ReviewDataset(val_encodings, preprocessing.one_hot_encode(val_labels))
        test_dataset = ReviewDataset(test_encodings, preprocessing.one_hot_encode(test_labels))
        t_end = time.time()
        t_passed = round(t_end - t_start, 2)
        print(f"Done in: {t_passed}s")

        epochs = f"_ne{settings.num_epochs}" if settings.num_epochs != 3 else ""
        name_suffix = "_early" if settings.early_stopping else ""
        ft_model_name = f'{model_name}-fine_tuned-{train_length}({length}){name_suffix}{epochs}'
        base_path = f'./local_{model_name}/{ft_model_name}'
        final_model_path = f'{base_path}/final'

        if helpers.is_dir_empty(final_model_path) or settings.re_train_net:
            title = f"Fine-tuning '{model_name}' with {train_length} data points."
            print_task_header(title, len(title) + 2, True)

            trainer = fine_tuning.get_trainer(
                # trainer arguments
                model_name=model_name,
                save_path=base_path,
                num_train_epochs=settings.num_epochs,
                # trainer sets
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            ft_start = time.time()
            trainer.train()
            ft_end = time.time()
            trainer.save_model(final_model_path)
            res = trainer.evaluate()
            train_time = round(ft_end - ft_start, 2)
            print(res)
            eval_results = {
                "name": ft_model_name,
                "training_time": str(datetime.timedelta(seconds=train_time)),
                "results": res
            }
            helpers.write_dict_to_json(name="eval", path=final_model_path, results=eval_results)

        else:
            print(
                f"\n\n\n> Skipping training: Fine-tuned '{model_name}' already trained on {review_lengths} data points.")

        num_test_points = len(test_labels)
        title = f"Classification with fine-tuned '{model_name}' on {num_test_points} data points."
        print_task_header(title, len(title) + 2, True, 1)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load pre-trained model from harddrive
        load_model = AutoModelForSequenceClassification.from_pretrained(
            final_model_path,
            problem_type='multi_label_classification'
        ).to(device)

        t_start = time.time()
        printProgressBar(0, num_test_points, prefix='Progress:', suffix='Complete', length=50, time=0)
        predictions = make_predictions(test_texts, t_start, num_test_points, tokenizer, load_model, device)
        t_end = time.time()
        class_time = round(t_end - t_start, 2)

        print_task_header("Classification Results", len("Classification Results") + 4, True, 1)

        print(f"Classification took {class_time}s")

        acc_b = accuracy_score(test_labels, predictions)
        mse = mean_squared_error(test_labels, predictions)
        class_report = classification_report(test_labels, predictions, digits=3)
        cm = confusion_matrix(test_labels, predictions)

        print(f"Accuracy: {round(acc_b * 100, 2)}%")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification report:\n{class_report}")
        print("")
        test_dict = {
            "model_name": ft_model_name,
            "results": {
                "classification_time": f"{class_time}s",
                "accuracy": round(acc_b, 5),
                "MSE": round(mse, 5),
                "confusion_matrix": get_cm_as_dict(cm),
                "class_report": get_class_report_as_dict(class_report)
            }
        }

        postfix = f"{num_test_points}" if test_data_changed else "basic"
        if settings.eval_for_comparing:
            selected_path = f'./tests/{num_test_points}_points'
            selected_name = f"{ft_model_name}_testing"
            create_dir_if_not_exist(selected_path)
        else:
            selected_path = final_model_path
            selected_name = "testing"

        save_confusion_matrix(matrix=cm, postfix=postfix, path=selected_path, name=selected_name)
        write_dict_to_json(name=selected_name, path=selected_path, results=test_dict, postfix=postfix)
