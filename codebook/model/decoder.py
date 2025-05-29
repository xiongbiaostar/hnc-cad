from .layers.transformer import *
from .layers.improved_transformer import *
import torch.nn as nn
import torch
from config import *
from model.network import *


class SolidDecoder(nn.Module):
    """
  Transformer-based decoder for solid
  """

    def __init__(self):
        """
    Initializes model.
    """
        super(SolidDecoder, self).__init__()
        self.embed_dim = DECODER_CONFIG['embed_dim']
        self.param_embed = Embedder(2 ** BIT, 32)
        self.param_fc = nn.Sequential(
            nn.Linear(32 * SOLID_PARAM_SEQ, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
        )

        self.pos_embed = PositionalEncoding(max_len=MAX_SOLID + 1, d_model=self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(32))

        layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, nhead=DECODER_CONFIG['num_heads'],
                                                 dim_feedforward=DECODER_CONFIG['hidden_dim'],
                                                 dropout=DECODER_CONFIG['dropout_rate'])
        self.network = TransformerDecoder(layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

        self.param_logit1 = nn.Linear(self.embed_dim, 32 * SOLID_PARAM_SEQ)
        self.param_logit2 = nn.Linear(32, 2 ** BIT)

    def forward(self, param, seq_mask, ignore_mask, latent_code):
        """ forward pass """
        bs = len(param)
        p_embeds = self.param_embed(param)
        p_embeds[ignore_mask] = self.mask_token  # replaced with masked token
        p_embeds = p_embeds.flatten(start_dim=2, end_dim=3)
        p_embeds = self.param_fc(p_embeds.flatten(0, 1)).unflatten(0, (p_embeds.shape[0], p_embeds.shape[1]))
        box_embeds = p_embeds

        input_embeds = self.pos_embed(torch.cat([latent_code, box_embeds], axis=1).transpose(0, 1))

        # Pass through decoder
        seq_mask = torch.cat([(torch.zeros([bs, 1]) == 1).cuda(), seq_mask], axis=1)
        decoder_out = self.network(tgt=input_embeds, tgt_key_padding_mask=seq_mask, memory=None)
        decoder_out = decoder_out[1:].transpose(1, 0)

        param_logits1 = self.param_logit1(decoder_out)
        param_logits2 = self.param_logit2(
            param_logits1.view(param_logits1.shape[0], param_logits1.shape[1], SOLID_PARAM_SEQ, -1))

        return param_logits2


class ProfileDecoder(nn.Module):
    """
  Transformer-based decoder for profile
  """

    def __init__(self):
        super(ProfileDecoder, self).__init__()
        self.embed_dim = DECODER_CONFIG['embed_dim']
        self.bbox_embed = Embedder(2 ** BIT, 32)
        self.type_embed = Embedder(TYPE_PARAM_PAD, 32)
        self.bbox_fc = nn.Sequential(
            nn.Linear(32 * PROFILE_PARAM_SEQ, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
        )

        self.pos_embed = PositionalEncoding(max_len=MAX_PROFILE + 1, d_model=self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(32))

        layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, nhead=DECODER_CONFIG['num_heads'],
                                                 dim_feedforward=DECODER_CONFIG['hidden_dim'],
                                                 dropout=DECODER_CONFIG['dropout_rate'])
        self.network = TransformerDecoder(layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

        self.param_logit1 = nn.Linear(self.embed_dim, 32 * PROFILE_PARAM_SEQ)
        self.bbox_logit = nn.Linear(32, 2 ** BIT)
        self.room_type_logit = nn.Linear(32, TYPE_PARAM_PAD)

    def forward(self, coord, seq_mask, ignore_mask, latent_code):
        """ forward pass """
        bs = len(coord)
        p_embed = self.bbox_embed(coord[:, :, 1:])  # coord(6,20,4)
        type_embeddings = self.type_embed(coord[:, :, 0:1])
        p_embeds = torch.cat((type_embeddings, p_embed), dim=2)

        p_embeds[ignore_mask] = self.mask_token
        p_embeds = p_embeds.flatten(start_dim=2, end_dim=3)
        bbox_embeds = self.bbox_fc(p_embeds.flatten(0, 1)).unflatten(0, (p_embeds.shape[0], p_embeds.shape[1]))

        input_embeds = self.pos_embed(torch.cat([latent_code, bbox_embeds], axis=1).transpose(0, 1))

        # Pass through decoder
        seq_mask = torch.cat([(torch.zeros([bs, 1]) == 1).cuda(), seq_mask], axis=1)
        decoder_out = self.network(tgt=input_embeds, tgt_key_padding_mask=seq_mask, memory=None)
        decoder_out = decoder_out[1:]
        decoder_out = decoder_out.transpose(0, 1)

        # Logits fc
        param_logits1 = self.param_logit1(decoder_out)
        param_logits1 = param_logits1.view(param_logits1.shape[0], param_logits1.shape[1], PROFILE_PARAM_SEQ, -1)
        bbox_features = param_logits1[:, :, 1:, :]
        room_type_features = param_logits1[:, :, 0:1, :]

        bbox_logits = self.bbox_logit(bbox_features)
        room_type_logits = self.room_type_logit(room_type_features)

        return bbox_logits, room_type_logits


class LoopDecoder(nn.Module):
    """
  Transformer-based decoder for loop
  """

    def __init__(self):
        """
    Initializes.
    """
        super(LoopDecoder, self).__init__()
        self.embed_dim = DECODER_CONFIG['embed_dim']
        self.param_embed = Embedder(2 ** BIT + LOOP_PARAM_PAD, 32)
        self.param_fc = nn.Sequential(
            nn.Linear(32 * LOOP_PARAM_SEQ, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
        )

        self.type_embed = Embedder(TYPE_PARAM_PAD, 32)
        self.pos_embed = PositionalEncoding(max_len=MAX_LOOP + 1, d_model=self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(32))

        layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, nhead=DECODER_CONFIG['num_heads'],
                                                 dim_feedforward=DECODER_CONFIG['hidden_dim'],
                                                 dropout=DECODER_CONFIG['dropout_rate'])
        self.network = TransformerDecoder(layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

        self.param_logit1 = nn.Linear(self.embed_dim, 32 * LOOP_PARAM_SEQ)
        self.bbox_logit = nn.Linear(32, 2 ** BIT + LOOP_PARAM_PAD)
        self.room_type_logit = nn.Linear(32, TYPE_PARAM_PAD)

    def forward(self, coord, seq_mask, ignore_mask, latent_code):
        """ forward pass """
        bs = len(coord)
        p_embed = self.param_embed(coord[:, 1:, :])
        type_embeddings = self.type_embed(coord[:, 0:1, :])
        p_embeds = torch.cat((type_embeddings, p_embed), dim=1)
        p_embeds[ignore_mask] = self.mask_token
        p_embeds = p_embeds.flatten(start_dim=2, end_dim=3)
        p_embeds = self.param_fc(p_embeds.flatten(0, 1)).unflatten(0, (p_embeds.shape[0], p_embeds.shape[1]))

        input_embeds = self.pos_embed(torch.cat([latent_code, p_embeds], axis=1).transpose(0, 1))
        seq_mask = torch.cat([(torch.zeros([bs, 1]) == 1).cuda(), seq_mask], axis=1)

        decoder_out = self.network(tgt=input_embeds, tgt_key_padding_mask=seq_mask, memory=None)
        decoder_out = decoder_out[1:].transpose(0, 1)

        param_logits1 = self.param_logit1(decoder_out)
        param_logits1 = param_logits1.view(param_logits1.shape[0], param_logits1.shape[1], LOOP_PARAM_SEQ, -1)
        bbox_features = param_logits1[:, 1:, :, :]
        room_type_features = param_logits1[:, 0:1, :, :]

        bbox_logits = self.bbox_logit(bbox_features)

        room_type_logits = self.room_type_logit(room_type_features)

        return bbox_logits, room_type_logits

class CombinedDecoder(nn.Module):
    """
    Transformer-based decoder for combined profile and loop decoding.
    """

    def __init__(self):
        super(CombinedDecoder, self).__init__()
        self.embed_dim = DECODER_CONFIG['embed_dim']

        # Embedders
        self.profile_bbox_embed = Embedder(2 ** BIT, 32)
        self.profile_type_embed = Embedder(TYPE_PARAM_PAD, 32)
        self.loop_param_embed = Embedder(2 ** BIT + LOOP_PARAM_PAD, 32)
        self.loop_type_embed = Embedder(TYPE_PARAM_PAD, 32)

        # Fully connected layers
        self.profile_bbox_fc = nn.Sequential(
            nn.Linear(32 * PROFILE_PARAM_SEQ, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
        )
        self.loop_param_fc = nn.Sequential(
            nn.Linear(32 * LOOP_PARAM_SEQ, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
        )

        # Positional encoding
        self.profile_pos_embed = PositionalEncoding(max_len=MAX_PROFILE + 1, d_model=self.embed_dim)
        self.loop_pos_embed = PositionalEncoding(max_len=MAX_LOOP + 1, d_model=self.embed_dim)

        # Mask token for missing values
        self.mask_token = nn.Parameter(torch.zeros(32))

        # Transformer decoder layers
        profile_layers = TransformerDecoderLayerImproved(
            d_model=self.embed_dim,
            nhead=DECODER_CONFIG['num_heads'],
            dim_feedforward=DECODER_CONFIG['hidden_dim'],
            dropout=DECODER_CONFIG['dropout_rate'],
        )
        loop_layers = TransformerDecoderLayerImproved(
            d_model=self.embed_dim,
            nhead=DECODER_CONFIG['num_heads'],
            dim_feedforward=DECODER_CONFIG['hidden_dim'],
            dropout=DECODER_CONFIG['dropout_rate'],
        )
        self.profile_network = TransformerDecoder(profile_layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))
        self.loop_network = TransformerDecoder(loop_layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

        # Output layers
        self.profile_param_logit = nn.Linear(self.embed_dim, 32 * PROFILE_PARAM_SEQ)
        self.profile_bbox_logit = nn.Linear(32, 2 ** BIT)
        self.profile_room_type_logit = nn.Linear(32, TYPE_PARAM_PAD)

        self.loop_param_logit = nn.Linear(self.embed_dim, 32 * LOOP_PARAM_SEQ)
        self.loop_bbox_logit = nn.Linear(32, 2 ** BIT + LOOP_PARAM_PAD)
        self.loop_room_type_logit = nn.Linear(32, TYPE_PARAM_PAD)

    def forward(self, profile_coord, loop_coord, profile_seq_mask, loop_seq_mask, profile_ignore_mask, loop_ignore_mask, latent_code):
        """
        Forward pass for combined profile and loop decoding.
        """
        bs = len(profile_coord)

        # Profile embedding
        profile_p_embed = self.profile_bbox_embed(profile_coord[:, :, 1:])
        profile_type_embeddings = self.profile_type_embed(profile_coord[:, :, 0:1])
        profile_embeds = torch.cat((profile_type_embeddings, profile_p_embed), dim=2)
        profile_embeds[profile_ignore_mask] = self.mask_token
        profile_embeds = profile_embeds.flatten(start_dim=2, end_dim=3)
        profile_bbox_embeds = self.profile_bbox_fc(profile_embeds.flatten(0, 1)).unflatten(0, (profile_embeds.shape[0], profile_embeds.shape[1]))

        profile_input_embeds = self.profile_pos_embed(torch.cat([latent_code, profile_bbox_embeds], axis=1).transpose(0, 1))
        profile_seq_mask = torch.cat([(torch.zeros([bs, 1]) == 1).cuda(), profile_seq_mask], axis=1)

        # Loop embedding
        loop_p_embed = self.loop_param_embed(loop_coord[:, 1:, :])
        loop_type_embeddings = self.loop_type_embed(loop_coord[:, 0:1, :])
        loop_embeds = torch.cat((loop_type_embeddings, loop_p_embed), dim=1)
        loop_embeds[loop_ignore_mask] = self.mask_token
        loop_embeds = loop_embeds.flatten(start_dim=2, end_dim=3)
        loop_embeds = self.loop_param_fc(loop_embeds.flatten(0, 1)).unflatten(0, (loop_embeds.shape[0], loop_embeds.shape[1]))

        loop_input_embeds = self.loop_pos_embed(torch.cat([latent_code, loop_embeds], axis=1).transpose(0, 1))
        loop_seq_mask = torch.cat([(torch.zeros([bs, 1]) == 1).cuda(), loop_seq_mask], axis=1)

        # Profile decoding
        profile_decoder_out = self.profile_network(tgt=profile_input_embeds, tgt_key_padding_mask=profile_seq_mask, memory=None)
        profile_decoder_out = profile_decoder_out[1:].transpose(0, 1)

        # Profile outputs
        profile_param_logits = self.profile_param_logit(profile_decoder_out)
        profile_param_logits = profile_param_logits.view(profile_param_logits.shape[0], profile_param_logits.shape[1], PROFILE_PARAM_SEQ, -1)
        profile_bbox_features = profile_param_logits[:, :, 1:, :]
        profile_room_type_features = profile_param_logits[:, :, 0:1, :]

        profile_bbox_logits = self.profile_bbox_logit(profile_bbox_features)
        profile_room_type_logits = self.profile_room_type_logit(profile_room_type_features)

        # Loop decoding
        loop_decoder_out = self.loop_network(tgt=loop_input_embeds, tgt_key_padding_mask=loop_seq_mask, memory=None)
        loop_decoder_out = loop_decoder_out[1:].transpose(0, 1)

        # Loop outputs
        loop_param_logits = self.loop_param_logit(loop_decoder_out)
        loop_param_logits = loop_param_logits.view(loop_param_logits.shape[0], loop_param_logits.shape[1], LOOP_PARAM_SEQ, -1)
        loop_bbox_features = loop_param_logits[:, 1:, :, :]
        loop_room_type_features = loop_param_logits[:, 0:1, :, :]

        loop_bbox_logits = self.loop_bbox_logit(loop_bbox_features)
        loop_room_type_logits = self.loop_room_type_logit(loop_room_type_features)

        return profile_bbox_logits, profile_room_type_logits, loop_bbox_logits, loop_room_type_logits