import math
from typing import Tuple

import torch
from torch import nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x) * math.sqrt(
            self.d_model
        )  # paper suggests to scale by sqrt(d_model)


class PositionalEncodding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # apply the sin to even positions
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # apply the cos to odd positions
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x(self.positional_encoding[:, : x.shape[1], :]).requires_grad_(False)  # type: ignore
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon  # for numerical stability
        self.alpha = nn.Parameter(torch.ones(1))  # multiplicative parameter
        self.beta = nn.Parameter(torch.zeros(1))  # additive parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)  # W1 and B1
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)  # W2 and B2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, seq_len, d_model)
        # -> (Batch, seq_len, d_ff)
        # -> (Batch, seq_len, d_model)
        x = self.linear1(x)
        x = torch.relu(input=x)
        x = self.dropout(x)
        return self.linear2(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)

        assert d_model % num_heads == 0, "d_model must be a multiple of num_heads"

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

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask=mask == 0, value=-1e9)
        attention_scores = torch.softmax(
            input=attention_scores, dim=-1
        )  # (Batch, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(p=attention_scores)

        return attention_scores @ value, attention_scores

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.w_q(
            input=query
        )  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(
            input=key
        )  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(
            input=value
        )  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, num_heads, d_k)
        # transpose to (Batch, num_heads, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query=query, key=key, value=value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.d_k)
        )
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x=x)))


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
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](
            x,
            lambda x: self.attention_block(
                query=x, key=x, value=x, mask=src_mask
            ),  # calling the forward method of MultiHeadAttentionBlock
        )
        return self.residual_connections[1](x, self.feed_forward_block)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x=x)


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
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,  # coming from the encoder, to mask the source sequence
        target_mask: torch.Tensor,  # coming from the decoder, to mask the target sequence
    ) -> torch.Tensor:
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(
                query=x, key=x, value=x, mask=target_mask
            ),
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                query=x, key=encoder_output, value=encoder_output, mask=src_mask
            ),
        )
        return self.residual_connections[2](x, self.feed_forward_block)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                target_mask=target_mask,
            )  # calling the forward method of DecoderBlock
        return self.norm(x=x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.linear(x), dim=-1)  # for numerical stability


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        src_pos: PositionalEncodding,
        target_embed: InputEmbeddings,
        target_pos: PositionalEncodding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(x=src, mask=src_mask)

    def decode(
        self,
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(
            x=target,
            encoder_output=encoder_output,
            src_mask=src_mask,
            target_mask=target_mask,
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_len: int,
    target_sq_len: int,
    d_model: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_encoder_layers: int = 6,
    d_ff: int = 2048,
    num_decoder_layers: int = 6,
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    target_embed = InputEmbeddings(d_model=d_model, vocab_size=target_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncodding(d_model=d_model, seq_len=src_len, dropout=dropout)
    target_pos = PositionalEncodding(
        d_model=d_model, seq_len=target_sq_len, dropout=dropout
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
    encoder = Encoder(layers=nn.ModuleList(encoder_blocks))
    decoder = Decoder(layers=nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=target_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        src_pos=src_pos,
        target_embed=target_embed,
        target_pos=target_pos,
        projection_layer=projection_layer,
    )

    # Initialize the parameters with xaiver uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
