import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)

from data import prepare_data, DataCollator

try:
    import wandb

    WANDB = True
except ImportError:
    logging.info("Wandb not installed. Skipping tracking.")
    WANDB = False

WANDB = False

MODELS = {
    "BERT": "bert-base-uncased",
    "ROBERTA": "roberta-base",
    "DEBERTA": "microsoft/deberta-base",
    "ERNIE": "nghuyong/ernie-2.0-base-en",
    "DISTILBERT": "distilbert-base-uncased",
    "ALBERT": "albert-base-v2",
}
VALID_MODELS = list(MODELS.keys())


def compute_metrics(pred):
    """
    Compute the metrics for the given predictions.
    :param pred: the predictions
    :return: accuracy
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def train_transformer(
    model,
    input_path,
    output_dir,
    training_batch_size,
    eval_batch_size,
    learning_rate,
    num_train_epochs,
    weight_decay,
    disable_tqdm=False,
):
    """
    Train and fine-tune the model using HuggingFace's PyTorch implementation and the Trainer API.
    """

    # training params
    model_ckpt = MODELS[model]
    print(model_ckpt)

    # I prototype on mac m1 which can use Metal Performance Shaders
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS Device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA Device.")
    else:
        device = torch.device("cpu")
        print("Using CPU Device.")

    output = f"{output_dir}/{model_ckpt}-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # max length of 512 for ERNIE as it is not predefined in the model
    if model == "ERNIE":
        test_data, train_data, validation_data, labels = prepare_data(
            input_path, tokenizer, device, max_length=512
        )
    else:
        test_data, train_data, validation_data, labels = prepare_data(
            input_path, tokenizer, device
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=len(labels)
    ).to(device)
    logging_steps = len(train_data) // training_batch_size

    # train
    if WANDB:
        wandb.watch(model)
        training_args = TrainingArguments(
            output_dir=output,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=training_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            disable_tqdm=disable_tqdm,
            logging_steps=logging_steps,
            log_level="error",
            logging_dir="./logs",
            report_to="wandb",
        )
    else:
        training_args = TrainingArguments(
            output_dir=output,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=training_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            disable_tqdm=disable_tqdm,
            logging_steps=logging_steps,
            log_level="error",
            logging_dir="./logs",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollator(tokenizer, device),
    )

    trainer.train()

    evaluate_trainer(trainer, test_data, output_dir)

    # save model
    model.save_pretrained(f"{output}/model")


def evaluate_trainer(trainer, test_data, output_dir):
    """
    Evaluate the fine-tuned trainer on the test set.
    Therefore, the accuracy is computed and a confusion matrix is generated.
    """

    # accuracy
    prediction_output = trainer.predict(test_data)
    logging.info(f"Prediction metrics: {prediction_output.metrics}")

    # confusion matrix
    y_preds = np.argmax(prediction_output.predictions, axis=1)
    y_true = prediction_output.label_ids
    cm = confusion_matrix(y_true, y_preds)
    logging.info(f"Confusion matrix:\n{cm}")

    # create file if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save results to file
    with open(f"{output_dir}/eval_results.json", "a") as f:
        f.write("\n")
        json.dump(prediction_output.metrics, f)

    if WANDB:
        wandb.log(prediction_output.metrics)


def evaluate_model(model, test_data, eval_batch_size, device, output_dir):
    """
    Evaluate a trained PyTorch model.
    Therefore, the accuracy and loss are calculated.
    """
    data_loader = DataLoader(test_data, shuffle=False, batch_size=eval_batch_size)
    all_logits = []
    all_targets = []
    eval_steps, eval_loss = 0, 0.0
    for batch in tqdm(data_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            flat_inputs, lengths, labels = batch
            outputs = model(flat_inputs, lengths, labels)
            all_targets.append(labels.detach().cpu())

        eval_steps += 1
        loss, logits = outputs[:2]
        eval_loss += loss.mean().item()
        all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    eval_loss /= eval_steps
    preds = np.argmax(logits, axis=1)
    acc = (preds == targets).sum() / targets.size

    # create file if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # append result to file
    with open(f"{output_dir}/eval_results.json", "a") as f:
        f.write("\n")
        json.dump({"acc": acc, "model parameter": str(model)}, f)

    if WANDB:
        wandb.log({"eval/loss": eval_loss, "eval/accuracy": acc})

    return acc, eval_loss


def eval_for_different_alpha(init_acc, batch_size, dataset, device, model, alpha):
    """
    Evaluate the model for different alpha values.
    The values are chosen in a range from 0.1 to 0.9 with a step size of 0.1.

    :param init_acc: the initial accuracy
    :param batch_size: the batch size
    :param dataset: the dataset
    :param device: the device
    :param model: the model
    :param alpha: the initial alpha value
    """
    best_acc = init_acc
    # test different alpha values
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        acc = model.evaluate(dataset, batch_size, device, a)
        logging.info(f"alpha: {a}, acc: {acc}")
        if acc > best_acc:
            best_acc = acc
            alpha = a
    if WANDB:
        wandb.log(
            {
                "eval/accuracy": acc,
                "best alpha": alpha,
                "eval/accuracy with fix alpha": init_acc,
            }
        )


def main():
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Run text classification on the given dataset with the given model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # general arguments
    parser.add_argument("model", type=str, choices=VALID_MODELS, help="Model to use.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="./input/dialogsum_clustered.csv",
        help="Path to the input data containing the clusers of DialogSum data.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory."
    )
    parser.add_argument("--log_level", type=str, default="info", help="Log level.")
    parser.add_argument("--log_to_file", action="store_true", help="Log to file.")
    parser.add_argument("--log_file", type=str, default="log.txt", help="Log file.")

    # training arguments
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.00, help="Weight decay."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of warmup steps."
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout.")

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    if args.log_to_file:
        logging.basicConfig(
            filename=f"{args.output_dir}/{args.log_file}", level=log_level
        )
    else:
        logging.basicConfig(level=log_level)

    # init wandb
    if WANDB:
        # init wandb name
        model = args.model
        name = model

        wandb.init(project="AUA-ST-Task", name=name, config=vars(args))
        config = wandb.config
    else:
        config = vars(args)

    logging.info("Starting...")
    logging.debug("Arguments: %s", args)

    # Start training
    logging.info("Loading data...")

    train_transformer(
        config["model"],
        config["input_path"],
        config["output_dir"],
        training_batch_size=config["batch_size"],
        eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
    )


if __name__ == "__main__":
    main()
