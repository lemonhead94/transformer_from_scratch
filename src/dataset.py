from typing import Any, Dict

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        source_language: str,
        target_language: str,
        sequence_length: int,
    ) -> None:
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.sequence_length = sequence_length

        self.sos_token = torch.tensor(
            [source_tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [source_tokenizer.token_to_id("[EOS]")],
            dtype=torch.int64,
        )
        self.pad_token = torch.tensor(
            [source_tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: Any) -> Dict[str, torch.Tensor]:
        source_target_pair = self.dataset[index]
        source_text = source_target_pair["translation"][self.source_language]
        target_text = source_target_pair["translation"][self.target_language]

        encoder_input_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_text).ids

        encoder_number_paddding_tokens = (
            self.sequence_length
            - len(encoder_input_tokens)
            - 2  # -2 for sos and eos
        )
        decoder_number_paddding_tokens = (
            self.sequence_length
            - len(decoder_input_tokens)
            - 1  # only sos token
        )

        if (
            encoder_number_paddding_tokens < 0
            or decoder_number_paddding_tokens < 0
        ):
            raise ValueError("Sequence length is too short.")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(encoder_number_paddding_tokens),
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.pad_token.repeat(decoder_number_paddding_tokens),
            ]
        )

        # what we expect as output from the decoder
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(decoder_number_paddding_tokens),
            ]
        )

        assert len(encoder_input) == self.sequence_length
        assert len(decoder_input) == self.sequence_length
        assert len(label) == self.sequence_length

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .int(),  # used for self-attention
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .int()
            & causal_mask(decoder_input.size(0)),
            "label": label,
            "source_text": source_text,
            "target_text": target_text,
        }


def causal_mask(sequence_length: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, sequence_length, sequence_length)).type(
        torch.int
    )
    return mask == 0
