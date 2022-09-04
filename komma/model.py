import toml
import torch
import operator
import numpy as np
import os
import wandb

from komma.data import create_loader

from torch import nn, optim
from typing import Dict, Any

from sklearn import metrics

from transformers import BertConfig, BertModel, BertTokenizer
from alive_progress import alive_bar

DEMO_TEXT: str = "Nuværende ejere får nemlig en blivende skatterabat så de ikke skal betale den eventuelle stigning i ejendomsværdiskatten når vi går over til det nye system i 2024."


class CommaBERT(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(CommaBERT, self).__init__()

        segment_size = config["data"]["segment_size"]
        bert_uri = config["model"]["bert_uri"]

        bert_config = BertConfig.from_pretrained(bert_uri, output_hidden_states=True)

        self.bert = BertModel.from_pretrained(bert_uri, config=bert_config)

        self.batch_norm = nn.BatchNorm1d(36864)

        self.linear = nn.Linear(36864, 1)  # Magic number, BERT output hidden size ...
        self.dropout = nn.Dropout(config["model"]["dropout"])
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.bert(x)[2][-1]

        x = x.view(x.shape[0], -1)

        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.linear(x)

        return self.activation(x)


def evaluate(model, data_loader, config: Dict[str, Any], stage: str):
    criterion = operator.attrgetter(config["training"][stage]["criterion"])(nn)().cuda()

    val_losses = []
    val_accuracies = []
    val_f1s = []

    for inputs, labels in data_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)

            outputs_binary = list(map(lambda x: int(x > 0.6), outputs.view(-1)))

            val_losses.append(criterion(outputs.view(-1), labels))
            val_f1s.append(metrics.f1_score(labels, outputs_binary))
            val_accuracies.append(metrics.accuracy_score(labels, outputs_binary))

    val_loss = np.mean(val_losses)
    val_accuracy = np.mean(val_accuracies)
    val_f1 = np.mean(val_f1s)

    return {"loss": val_loss, "accuracy": val_accuracy, "f1": val_f1}


def train_stage(
    model,
    data_loader,
    val_data_loader,
    config: Dict[str, Any],
    stage: str,
    tokenizer: BertTokenizer,
):
    lr_schedules = config["training"][stage]["lr"]
    epoch_schedules = config["training"][stage]["epochs"]

    criterion = operator.attrgetter(config["training"][stage]["criterion"])(nn)().cuda()
    optimizer = operator.attrgetter(config["training"][stage]["optimizer"])(optim)(
        model.parameters(), lr=lr_schedules[0]
    )

    model = model.cuda()

    model.train()  # Gym time

    best_val_accuracy = 0.0

    for round_i, epochs in enumerate(epoch_schedules):
        optimizer.lr = lr_schedules[min(round_i, len(lr_schedules) - 1)]

        print(f"Schedule {round_i}:\n... lr={optimizer.lr}")

        with alive_bar(epochs * len(data_loader), title=f"Round {round_i}") as bar:
            for e in range(epochs):
                for inputs, labels in data_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()

                    output = model(inputs)

                    loss = criterion(output.view(-1), labels)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    bar()

                tm = evaluate(model, data_loader, config, stage)
                vm = evaluate(model, val_data_loader, config, stage)

                if vm["accuracy"] > best_val_accuracy:
                    best_val_accuracy = vm["accuracy"]
                    torch.save(
                        model.state_dict(),
                        os.path.join(config["model"]["output_dir"], f"model_{e}.pt"),
                    )
                    print(
                        f"Saved model at {best_val_accuracy}% validation accuracy ..."
                    )

                wandb.log(
                    {
                        "epoch": e,
                        "accuracy": tm["accuracy"],
                        "val_accuracy": vm["accuracy"],
                        "loss": tm["loss"],
                        "val_loss": vm["loss"],
                        "learning_rate": optimizer.lr,
                        "round": round_i,
                    }
                )

                bar.text = f'... acc={tm["accuracy"]}, f1={tm["f1"]}, val_acc={vm["accuracy"]}, val_f1={vm["f1"]}'


def train(config, data_loader, val_data_loader, tokenizer):
    model = CommaBERT(config)

    for stage in config["training"]:
        print(f"Training stage: {stage}")
        train_stage(model, data_loader, val_data_loader, config, stage, tokenizer)

    print(config)


if __name__ == "__main__":
    import sys
    from os.path import join, dirname

    data_path = join(dirname(__file__), "..", "data")

    config = toml.loads(open(sys.argv[1], "r", encoding="utf-8").read())

    wandb.init(
        project="comma-model",
        config=dict(
            dropout=config["model"]["dropout"],
            batch_size=config["data"]["batch_size"],
            segment_size=config["data"]["segment_size"],
        ),
    )

    tokenizer = BertTokenizer.from_pretrained(config["model"]["bert_uri"])
    data_loader = create_loader(
        join(data_path, "train.txt"), tokenizer, 48, config["data"]["batch_size"]
    )

    val_data_loader = create_loader(
        join(data_path, "val.txt"), tokenizer, 48, config["data"]["batch_size"]
    )

    print("Train !!!")
    print(f"- on {len(data_loader)} training data")
    print(f"- on {len(val_data_loader)} validation data")
    print()

    train(config, data_loader, val_data_loader, tokenizer)
