import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import nn
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from tqdm import tqdm

from src.config import get_config, get_weights_file_path
from src.dataset import BilingualDataset, causal_mask
from src.model import Transformer, build_transformer

logger = logging.getLogger(__name__)


def greedy_docode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    target_tokenizer: Tokenizer,
    max_len: int,
    device: torch.device,
):
    # these token ids are the same for both tokenizers (source and target)
    start_of_sentence_index = target_tokenizer.token_to_id("[SOS]")
    end_of_sentence_index = target_tokenizer.token_to_id("[EOS]")

    # Precompute the encoder output and resuse it
    encoder_output = model.encode(source=source, source_mask=source_mask)
    # Initalize the decoder input with the start of sentence token
    decoder_input = (
        torch.empty(1, 1)
        .fill_(start_of_sentence_index)
        .type_as(source)
        .to(device)
    )
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target sequence (decoder input)
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )
        # Calculate the output of the decoder
        decoder_output = model.decode(
            encoder_output=encoder_output,
            source_mask=source_mask,
            target=decoder_input,
            target_mask=decoder_mask,
        )
        # get the next token, after the last token we have given to the encoder
        probabilities_of_next_token = model.project(
            input=decoder_output[:, -1]
        )
        # select the token with the highest probability (greedy search)
        _, next_token = torch.max(probabilities_of_next_token, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .type_as(source)
                .fill_(next_token.item())
                .to(device),
            ],
            dim=1,
        )

        if next_token == end_of_sentence_index:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model: Transformer,
    validation_dataloader: DataLoader,
    target_tokenizer: Tokenizer,
    max_len: int,
    device: torch.device,
    print_message: Callable,
    global_step: int,
    writer: SummaryWriter,
    number_of_examples: int = 2,
):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen("stty size", "r") as console:
            _, console_width_str = console.read().split()
            console_width = int(console_width_str)
    except Exception:
        # If we can't get the console width, use 80 as default
        console_width = 80

    # don't train the model
    # i.e. calculate gradients only do the forward pass (inference)
    with torch.no_grad():
        for batch in validation_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Only one sentence at a time"

            model_output = greedy_docode(
                model=model,
                source=encoder_input,
                source_mask=encoder_mask,
                target_tokenizer=target_tokenizer,
                max_len=max_len,
                device=device,
            )
            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_output_text = target_tokenizer.decode(
                model_output.detach().cpu().numpy()
            )
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)

            print_message("-" * console_width)
            print_message(f"Source: {source_text}")
            print_message(f"Target: {target_text}")
            print_message(f"Predicted: {model_output_text}")

            if count == number_of_examples:
                break
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        cer = CharErrorRate()
        writer.add_scalar(
            "validation cer", cer(predicted, expected), global_step
        )
        writer.flush()

        # Compute the word error rate
        wer = WordErrorRate()
        writer.add_scalar(
            "validation wer", wer(predicted, expected), global_step
        )
        writer.flush()

        # Compute the BLEU metric
        bleu = BLEUScore()
        writer.add_scalar(
            "validation BLEU", bleu(predicted, expected), global_step
        )
        writer.flush()


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item["translation"][language]


def get_or_build_tokenizer(
    config: Dict[str, str], dataset: Dataset, language: str
):
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not Path.exists(self=tokenizer_path):
        tokenizer = Tokenizer(model=WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(
            iterator=get_all_sentences(dataset=dataset, language=language),
            trainer=trainer,
        )
        tokenizer.save(path=str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config: Dict[str, str]):
    dataset = load_dataset(
        "opus_books",
        f"{config['source_language']}-{config['target_language']}",
        split="train",
    )

    # Build tokenizers
    source_tokenizer = get_or_build_tokenizer(
        config=config, dataset=dataset, language=config["source_language"]
    )
    target_tokenizer = get_or_build_tokenizer(
        config=config, dataset=dataset, language=config["target_language"]
    )

    # train/test split 90/10
    train_dataset_size = int(len(dataset) * 0.9)
    validation_dataset_size = len(dataset) - train_dataset_size
    train_dataset_raw, validation_dataset_raw = random_split(
        dataset, [train_dataset_size, validation_dataset_size]
    )

    train_dataset = BilingualDataset(
        dataset=train_dataset_raw,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_language=config["source_language"],
        target_language=config["target_language"],
        sequence_length=int(config["sequence_length"]),
    )
    validation_dataset = BilingualDataset(
        dataset=validation_dataset_raw,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_language=config["source_language"],
        target_language=config["target_language"],
        sequence_length=int(config["sequence_length"]),
    )

    max_len_source = 0
    max_len_target = 0

    for item in dataset:
        source_ids = source_tokenizer.encode(
            item["translation"][config["source_language"]]
        ).ids
        target_ids = target_tokenizer.encode(
            item["translation"][config["target_language"]]
        ).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))

    logger.info(f"Max length source sentence: {max_len_source}")
    logger.info(f"Max length target sentence: {max_len_target}")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=True,
    )

    return (
        train_dataloader,
        validation_dataloader,
        source_tokenizer,
        target_tokenizer,
    )


def get_model(
    config: Dict[str, Any], source_vocab_size: int, target_vocab_size: int
) -> Transformer:
    model = build_transformer(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        sequence_length=int(config["sequence_length"]),
        target_sequence_length=int(config["sequence_length"]),
        embedding_dimension=int(config["embedding_dimension"]),
    )
    return model


def train_model(config: Dict[str, str]):
    # Define the device
    device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_built() or torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device_type}")
    if device_type == "cuda":
        logger.info(
            f"Device name: {torch.cuda.get_device_name(device=device_type)}"
        )
        memory = (
            torch.cuda.get_device_properties(device=device_type).total_memory
            / 1024**3
        )
        logger.info(f"Device memory: {memory} GB")
    device = torch.device(device_type)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    (
        train_dataloader,
        validation_dataloader,
        source_tokenizer,
        target_tokenizer,
    ) = get_dataset(config=config)
    model = get_model(
        config=config,
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
    ).to(device)
    # Tensorboard
    writer = SummaryWriter(log_dir=config["experiment_name"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), eps=1e-9
    )

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(
            config=config, epoch=config["preload"]
        )
        logger.info(f"Loading weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=source_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, int(config["number_of_epochs"])):
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Processing epoch {epoch:02d}",
            total=len(train_dataloader),
        )
        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            # Run the tensors through the transformer
            encoder_output = model.encode(
                source=encoder_input, source_mask=encoder_mask
            )
            decoder_output = model.decode(
                target=decoder_input,
                target_mask=decoder_mask,
                encoder_output=encoder_output,
                source_mask=encoder_mask,
            )
            project_output = model.project(input=decoder_output)

            label = batch["label"].to(device)

            loss = loss_fn(
                project_output.view(-1, target_tokenizer.get_vocab_size()),
                label.view(-1),
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss to tensorboard
            writer.add_scalar(
                tag="train loss",
                scalar_value=loss.item(),
                global_step=global_step,
            )
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            run_validation(
                model=model,
                validation_dataloader=validation_dataloader,
                target_tokenizer=target_tokenizer,
                max_len=int(config["sequence_length"]),
                device=device,
                print_message=batch_iterator.write,
                global_step=global_step,
                writer=writer,
            )

            # Update the global step
            global_step += 1

        # Save the model
        model_filename = get_weights_file_path(
            config=config, epoch=f"{epoch:02d}"
        )
        logger.info(f"Saving weights to {model_filename}")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "model_state_dict": model.state_dict(),
            },
            model_filename,
        )


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    # set log level to info
    logging.basicConfig(level=logging.INFO)
    config = get_config()
    train_model(config=config)
