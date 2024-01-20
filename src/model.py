import math
from typing import Tuple

import torch
from torch import nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input) * math.sqrt(
            self.d_model
        )  # paper suggests to scale by sqrt(d_model)


class PositionalEncodding(nn.Module):
    def __init__(
        self, d_model: int, sequence_length: int, dropout: float
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(sequence_length, d_model)
        position = torch.arange(
            0, sequence_length, dtype=torch.float
        ).unsqueeze(dim=1)
        div_term = torch.exp(
            torch.arange(start=0, end=d_model, step=2).float()
            * (-math.log(10000.0) / d_model)
        )
        # apply the sin to even positions
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # apply the cos to odd positions
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(dim=0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pos_enc = input + (
            self.positional_encoding[:, : input.shape[1], :]
        ).requires_grad_(False)
        return self.dropout(pos_enc)


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon  # for numerical stability
        self.alpha = nn.Parameter(
            data=torch.ones(1)
        )  # multiplicative parameter
        self.beta = nn.Parameter(data=torch.zeros(1))  # additive parameter

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        return self.alpha * (input - mean) / (std + self.epsilon) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=d_model, out_features=d_ff
        )  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            in_features=d_ff, out_features=d_model
        )  # W2 and B2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (Batch, sequence_length, d_model)
        # -> (Batch, sequence_length, d_ff)
        # -> (Batch, sequence_length, d_model)
        lin1_out = self.linear1(input)
        relu_out = torch.relu(lin1_out)
        drop_out = self.dropout(relu_out)
        return self.linear2(drop_out)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert (
            d_model % num_heads == 0
        ), "d_model must be a multiple of num_heads"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]

        attention_scores = (
            query @ key.transpose(dim0=-2, dim1=-1)
        ) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask=(mask == 0), value=-1e9
            )
        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (Batch, num_heads, sequence_length, sequence_length)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        query = query.view(
            query.shape[0], query.shape[1], self.num_heads, self.d_k
        ).transpose(dim0=1, dim1=2)
        key = key.view(
            key.shape[0], key.shape[1], self.num_heads, self.d_k
        ).transpose(dim0=1, dim1=2)
        value = value.view(
            value.shape[0], value.shape[1], self.num_heads, self.d_k
        ).transpose(dim0=1, dim1=2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query=query, key=key, value=value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(dim0=1, dim1=2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.d_k)
        )
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, sublayer: nn.Module
    ) -> torch.Tensor:
        # The sublayer is either the MultiHeadAttentionBlock
        # or the FeedForwardBlock, both of which are nn.Module
        # We do this to apply skip connection to both of them
        return input + self.dropout(sublayer(self.norm(input)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            modules=[ResidualConnection(dropout=dropout) for _ in range(2)]
        )

    def forward(
        self, input: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        input = self.residual_connections[0](
            input=input,
            sublayer=lambda input: self.attention_block(
                query=input, key=input, value=input, mask=source_mask
            ),  # calling the forward method of MultiHeadAttentionBlock
        )
        return self.residual_connections[1](
            input=input, sublayer=self.feed_forward_block
        )


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            input = layer(input=input, source_mask=mask)
        return self.norm(input)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            modules=[ResidualConnection(dropout=dropout) for _ in range(3)]
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,  # from encoder, to mask the source sequence
        target_mask: torch.Tensor,  # from decoder, to mask the target sequence
    ) -> torch.Tensor:
        input = self.residual_connections[0](
            input=input,
            sublayer=lambda input: self.self_attention_block(
                query=input, key=input, value=input, mask=target_mask
            ),
        )
        input = self.residual_connections[1](
            input=input,
            sublayer=lambda input: self.cross_attention_block(
                query=input,
                key=encoder_output,
                value=encoder_output,
                mask=source_mask,
            ),
        )
        return self.residual_connections[2](
            input=input, sublayer=self.feed_forward_block
        )


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(
        self,
        input: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            input = layer(
                input=input,
                encoder_output=encoder_output,
                source_mask=source_mask,
                target_mask=target_mask,
            )  # calling the forward method of DecoderBlock
        return self.norm(input)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(
            input=self.linear(input), dim=-1
        )  # for numerical stability


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embed: InputEmbeddings,
        source_pos: PositionalEncodding,
        target_embed: InputEmbeddings,
        target_pos: PositionalEncodding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(
        self, source: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        source = self.source_embed(source)
        source = self.source_pos(source)
        return self.encoder(input=source, mask=source_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(
            input=target,
            encoder_output=encoder_output,
            source_mask=source_mask,
            target_mask=target_mask,
        )

    def project(self, input: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(input)


def build_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    sequence_length: int,
    target_sequence_len: int,
    d_model: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_encoder_layers: int = 6,
    d_ff: int = 2048,
    num_decoder_layers: int = 6,
) -> Transformer:
    # Create the embedding layers
    source_embed = InputEmbeddings(
        d_model=d_model, vocab_size=source_vocab_size
    )
    target_embed = InputEmbeddings(
        d_model=d_model, vocab_size=target_vocab_size
    )

    # Create the positional encoding layers
    source_pos = PositionalEncodding(
        d_model=d_model, sequence_length=sequence_length, dropout=dropout
    )
    target_pos = PositionalEncodding(
        d_model=d_model, sequence_length=target_sequence_len, dropout=dropout
    )

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_encoder_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        encoder_block = EncoderBlock(
            self_attention_block=encoder_self_attention_block,
            feed_forward_block=feed_forward_block,
            dropout=dropout,
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(num_decoder_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        decoder_block = DecoderBlock(
            self_attention_block=decoder_self_attention_block,
            cross_attention_block=decoder_cross_attention_block,
            feed_forward_block=feed_forward_block,
            dropout=dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(layers=nn.ModuleList(modules=encoder_blocks))
    decoder = Decoder(layers=nn.ModuleList(modules=decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(
        d_model=d_model, vocab_size=target_vocab_size
    )

    # Create the transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        source_embed=source_embed,
        source_pos=source_pos,
        target_embed=target_embed,
        target_pos=target_pos,
        projection_layer=projection_layer,
    )

    # Initialize the parameters with xaiver uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
