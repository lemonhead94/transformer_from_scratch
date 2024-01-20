import math
from typing import Tuple

import torch
from torch import nn


class LayerNormalization(nn.Module):
    def __init__(self, features: int, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon  # for numerical stability
        self.alpha = nn.Parameter(
            data=torch.ones(features)
        )  # multiplicative parameter
        self.bias = nn.Parameter(
            data=torch.zeros(features)
        )  # additive parameter

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        return self.alpha * (input - mean) / (std + self.epsilon) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        feedforward_hidden_layer_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=feedforward_hidden_layer_size,
        )  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            in_features=feedforward_hidden_layer_size,
            out_features=embedding_dimension,
        )  # W2 and B2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (Batch, sequence_length, embedding_dimension)
        # -> (Batch, sequence_length, feedforward_hidden_layer_size)
        # -> (Batch, sequence_length, embedding_dimension)
        linear1_output = self.linear1(input)
        relu_output = torch.relu(linear1_output)
        drop_output = self.dropout(relu_output)
        return self.linear2(drop_output)


class InputEmbedding(nn.Module):
    def __init__(self, embedding_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dimension
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (batch, sequence_length)
        # --> (batch, sequence_length, embedding_dimension)
        # paper suggests to scale by sqrt(embedding_dimension)
        return self.embedding(input) * math.sqrt(self.embedding_dimension)


class PositionalEncoding(nn.Module):
    def __init__(
        self, embedding_dimension: int, sequence_length: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(sequence_length, embedding_dimension)
        position = torch.arange(
            0, sequence_length, dtype=torch.float
        ).unsqueeze(dim=1)
        div_term = torch.exp(
            torch.arange(start=0, end=embedding_dimension, step=2).float()
            * (-math.log(10000.0) / embedding_dimension)
        )
        # apply the sin to even positions
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # apply the cos to odd positions
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(dim=0)
        # the positional encoding is a constant, so we register it as a buffer
        # it's not learned by the model hence we don't register it as a param
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        positional_encoded_input = input + (
            self.positional_encoding[:, : input.shape[1], :]
        ).requires_grad_(False)
        return self.dropout(positional_encoded_input)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.layer_normalization = LayerNormalization(features=features)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, sublayer: nn.Module
    ) -> torch.Tensor:
        # The sublayer is either the MultiHeadAttentionBlock
        # or the FeedForwardBlock, both of which are nn.Module
        # We do this to apply skip connection to both of them
        return input + self.dropout(sublayer(self.layer_normalization(input)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self, embedding_dimension: int, number_of_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.dropout = nn.Dropout(dropout)

        assert (
            embedding_dimension % number_of_heads == 0
        ), "embedding_dimension must be a multiple of number_of_heads"

        self.d_k = (
            embedding_dimension // number_of_heads
        )  # Dimension of vector seen by each head
        self.w_q = nn.Linear(
            in_features=embedding_dimension, out_features=embedding_dimension
        )
        self.w_k = nn.Linear(
            in_features=embedding_dimension, out_features=embedding_dimension
        )
        self.w_v = nn.Linear(
            in_features=embedding_dimension, out_features=embedding_dimension
        )
        self.w_o = nn.Linear(
            in_features=embedding_dimension, out_features=embedding_dimension
        )

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
            # Write a very low value (-inf) to the positions where mask == 0
            attention_scores = attention_scores.masked_fill(
                mask=(mask == 0), value=-1e9
            )
        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (Batch, number_of_heads, sequence_length, sequence_length)
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
            query.shape[0], query.shape[1], self.number_of_heads, self.d_k
        ).transpose(dim0=1, dim1=2)
        key = key.view(
            key.shape[0], key.shape[1], self.number_of_heads, self.d_k
        ).transpose(dim0=1, dim1=2)
        value = value.view(
            value.shape[0], value.shape[1], self.number_of_heads, self.d_k
        ).transpose(dim0=1, dim1=2)

        (
            context_vector,
            self.attention_scores,
        ) = MultiHeadAttentionBlock.attention(
            query=query, key=key, value=value, mask=mask, dropout=self.dropout
        )

        # concatenate the weights of all heads
        context_vector = (
            context_vector.transpose(dim0=1, dim1=2)
            .contiguous()
            .view(context_vector.shape[0], -1, self.number_of_heads * self.d_k)
        )
        # project this high-dimensional vector back down to the
        # original embedding dimension
        return self.w_o(context_vector)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            modules=[
                ResidualConnection(features=features, dropout=dropout)
                for _ in range(2)
            ]
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
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_normalization = LayerNormalization(features=features)

    def forward(
        self, input: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            input = layer(input=input, source_mask=source_mask)
        return self.layer_normalization(input)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
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
            modules=[
                ResidualConnection(features=features, dropout=dropout)
                for _ in range(3)
            ]
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_output: torch.Tensor,  # aka source
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
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_normalization = LayerNormalization(features=features)

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
        return self.layer_normalization(input)


class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.linear_projection = nn.Linear(
            in_features=embedding_dimension, out_features=vocab_size
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (batch, sequence_length, embedding_dimension)
        # --> (batch, sequence_length, vocab_size)
        return self.linear_projection(input)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embedding: InputEmbedding,
        source_positional_encoder: PositionalEncoding,
        target_embedding: InputEmbedding,
        target_positional_encoder: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_positional_encoder = source_positional_encoder
        self.target_positional_encoder = target_positional_encoder
        self.projection_layer = projection_layer

    def encode(
        self, source: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        source_embedding = self.source_embedding(source)
        source_embedding_positional_encoded = self.source_positional_encoder(
            source_embedding
        )
        return self.encoder(
            input=source_embedding_positional_encoded, source_mask=source_mask
        )

    def decode(
        self,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_embedding = self.target_embedding(target)
        target_embedding_positional_encoded = self.target_positional_encoder(
            target_embedding
        )
        return self.decoder(
            input=target_embedding_positional_encoded,
            encoder_output=encoder_output,
            source_mask=source_mask,
            target_mask=target_mask,
        )

    def project(self, input: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(input)


def build_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    source_sequence_length: int,
    target_sequence_length: int,
    embedding_dimension: int = 512,
    number_of_heads: int = 8,
    dropout: float = 0.1,
    number_of_encoder_layers: int = 6,
    feedforward_hidden_layer_size: int = 2048,
    number_of_decoder_layers: int = 6,
) -> Transformer:
    # Create the embedding layers
    source_embedding = InputEmbedding(
        embedding_dimension=embedding_dimension, vocab_size=source_vocab_size
    )
    target_embedding = InputEmbedding(
        embedding_dimension=embedding_dimension, vocab_size=target_vocab_size
    )

    # Create the positional encoding layers
    source_positional_encoder = PositionalEncoding(
        embedding_dimension=embedding_dimension,
        sequence_length=source_sequence_length,
        dropout=dropout,
    )
    target_positional_encoder = PositionalEncoding(
        embedding_dimension=embedding_dimension,
        sequence_length=target_sequence_length,
        dropout=dropout,
    )

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(number_of_encoder_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        feed_forward_block = FeedForwardBlock(
            embedding_dimension=embedding_dimension,
            feedforward_hidden_layer_size=feedforward_hidden_layer_size,
            dropout=dropout,
        )
        encoder_block = EncoderBlock(
            features=embedding_dimension,
            self_attention_block=encoder_self_attention_block,
            feed_forward_block=feed_forward_block,
            dropout=dropout,
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(number_of_decoder_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        feed_forward_block = FeedForwardBlock(
            embedding_dimension=embedding_dimension,
            feedforward_hidden_layer_size=feedforward_hidden_layer_size,
            dropout=dropout,
        )
        decoder_block = DecoderBlock(
            features=embedding_dimension,
            self_attention_block=decoder_self_attention_block,
            cross_attention_block=decoder_cross_attention_block,
            feed_forward_block=feed_forward_block,
            dropout=dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(
        features=embedding_dimension,
        layers=nn.ModuleList(modules=encoder_blocks),
    )
    decoder = Decoder(
        features=embedding_dimension,
        layers=nn.ModuleList(modules=decoder_blocks),
    )

    # Create the projection layer
    projection_layer = ProjectionLayer(
        embedding_dimension=embedding_dimension, vocab_size=target_vocab_size
    )

    # Create the transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        source_embedding=source_embedding,
        source_positional_encoder=source_positional_encoder,
        target_embedding=target_embedding,
        target_positional_encoder=target_positional_encoder,
        projection_layer=projection_layer,
    )

    # Initialize the parameters with xaiver uniform
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer
