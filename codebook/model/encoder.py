import torch.nn as nn
from .layers.transformer import *
from .layers.improved_transformer import *
from config import *
from model.network import *


class SolidEncoder(nn.Module):

  def __init__(self):
    """
    Initializes Solid Model.
    """
    super(SolidEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.param_embed = Embedder(2 ** BIT, 32)  # nn.embeddding(2**6,32)
    self.param_fc = nn.Sequential(
      nn.Linear(32 * SOLID_PARAM_SEQ, self.embed_dim),  # 输入32*6=192，输出256维。
      nn.BatchNorm1d(self.embed_dim),
      nn.LeakyReLU(),
    )

    self.pos_embed = PositionalEncoding(max_len=MAX_SOLID, d_model=self.embed_dim)  # (5,256)
    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'],
                                                     dim_feedforward=ENCODER_CONFIG['hidden_dim'],
                                                     dropout=ENCODER_CONFIG['dropout_rate'])
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, ENCODER_CONFIG['num_layers'], encoder_norm)

    commitment_cost = 0.25
    decay = 0.99
    self.code_dim = 256
    self.codebook = VectorQuantizerEMA(SOLID_CODEBOOK_DIM, self.code_dim, commitment_cost,
                                       decay)  # (10000,256,0.25,0.99)

    self.bottleneck = nn.Sequential(
      nn.Linear(self.embed_dim, self.embed_dim),  # (256,256)
      nn.BatchNorm1d(self.embed_dim),
      nn.Tanh()
    )

  def forward(self, param, seq_mask):
    """ forward pass """
    p_embeds = self.param_embed(param).flatten(start_dim=2, end_dim=3)  # (6,5,6,32)----(6,5,192)
    p_embeds = self.param_fc(p_embeds.flatten(0, 1)).unflatten(0, (
    p_embeds.shape[0], p_embeds.shape[1]))  # MLP1         (6,5,192)----(30,192)----(30,256)----(6,5,256)
    box_embeds = p_embeds
    encoder_input = self.pos_embed(box_embeds.transpose(0, 1))  # (6,5,256)----(5,6,256)

    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)  # （5,6,256）

    # Avg pool latent code
    z_mask = (~seq_mask).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.embed_dim)  # seq_mask(6,5)-----(5,6,256)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0,
                                                                      keepdim=False)  # 计算加权平均值。(5,6,256)*(5,6,256)----(5,6,256)----(6,256)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)  # (6,256)----(6,256)----(1,6,256)

    vq_loss, quantized, encodings_flat, selection = self.codebook(code_encoded)  # (1,6,256)----
    latent_code = quantized.transpose(0, 1)  # quantized(1,6,256)----latent_code(6,1,256)

    return latent_code, vq_loss, selection, encodings_flat  # latent_code(6,1,256)对应codebook中256维向量表示，   loss   selection(6,1)6个分别对应的codebook索引值   encodings_flat(1,6,10000) 6个分别对应codebook中选中表示

  def count_code(self, param, seq_mask):
    """ Codebook usage """
    p_embeds = self.param_embed(param).flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(p_embeds.flatten(0, 1)).unflatten(0, (p_embeds.shape[0], p_embeds.shape[1]))
    box_embeds = p_embeds
    encoder_input = self.pos_embed(box_embeds.transpose(0, 1))

    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)

    # Avg pool latent code
    z_mask = (~seq_mask).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    code_dist = self.codebook.count_code(code_encoded)

    return code_dist, code_encoded


class ProfileEncoder(nn.Module):

  def __init__(self):
    """
    Initializes Profile Model.
    """
    super(ProfileEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.bbox_embed = Embedder(2 ** BIT, 32)
    self.bbox_fc = nn.Sequential(
      nn.Linear(32 * PROFILE_PARAM_SEQ, self.embed_dim),  # (32*4)
      nn.BatchNorm1d(self.embed_dim),
      nn.LeakyReLU(),
    )

    self.type_embed = Embedder(8, 32)

    self.pos_embed = PositionalEncoding(max_len=MAX_PROFILE, d_model=self.embed_dim)

    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'],
                                                     dim_feedforward=ENCODER_CONFIG['hidden_dim'],
                                                     dropout=ENCODER_CONFIG['dropout_rate'])
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, ENCODER_CONFIG['num_layers'], encoder_norm)

    commitment_cost = 0.25
    decay = 0.99
    self.code_dim = self.embed_dim
    self.codebook = VectorQuantizerEMA(PROFILE_CODEBOOK_DIM, self.code_dim, commitment_cost, decay)

    self.bottleneck = nn.Sequential(
      nn.Linear(self.embed_dim, self.embed_dim),
      nn.BatchNorm1d(self.embed_dim),
      nn.Tanh()
    )

  def forward(self, coord, seq_mask):
    """ forward pass """

    coord_data =  coord[:, :, 1:]
    type_data = coord[:, :, 0:1]
    p_embed = self.bbox_embed(coord[:, :, 1:])  # coord(6,20,4)

    type_embeddings = self.type_embed(coord[:, :, 0:1])
    combined_features = torch.cat((type_embeddings, p_embed), dim=2).flatten(start_dim=2, end_dim=3)

    coord_embed = self.bbox_fc(combined_features.flatten(0, 1)).unflatten(0, (combined_features.shape[0], combined_features.shape[1]))


    encoder_input = self.pos_embed(coord_embed.transpose(0, 1))

    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)

    # Avg pool latent code
    z_mask = (~seq_mask).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    vq_loss, quantized, encodings_flat, selection = self.codebook(code_encoded)
    latent_code = quantized.transpose(0, 1)

    return latent_code, vq_loss, selection, encodings_flat

  def count_code(self, coord, seq_mask):
    """ Codebook usage """
    p_embed = self.bbox_embed(coord).flatten(start_dim=2, end_dim=3)
    coord_embed = self.bbox_fc(p_embed.flatten(0, 1)).unflatten(0, (p_embed.shape[0], p_embed.shape[1]))
    encoder_input = self.pos_embed(coord_embed.transpose(0, 1))

    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)

    # Avg pool latent code
    z_mask = (~seq_mask).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    code_dist = self.codebook.count_code(code_encoded)

    return code_dist, code_encoded


class LoopEncoder(nn.Module):

  def __init__(self):
    """
    Initializes Loop Model.
    """
    super(LoopEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.param_embed = Embedder(2 ** BIT + LOOP_PARAM_PAD, 32)
    self.param_fc = nn.Sequential(
      nn.Linear(32 * LOOP_PARAM_SEQ, self.embed_dim),
      nn.BatchNorm1d(self.embed_dim),
      nn.LeakyReLU(),
    )

    self.type_embed = Embedder(TYPE_PARAM_PAD, 32)

    self.pos_embed = PositionalEncoding(max_len=MAX_LOOP, d_model=self.embed_dim)

    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'],
                                                     dim_feedforward=ENCODER_CONFIG['hidden_dim'],
                                                     dropout=ENCODER_CONFIG['dropout_rate'])
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, ENCODER_CONFIG['num_layers'], encoder_norm)

    commitment_cost = 0.25
    decay = 0.99
    self.code_dim = 256
    self.codebook = VectorQuantizerEMA(LOOP_CODEBOOK_DIM, self.code_dim, commitment_cost, decay)

    self.bottleneck = nn.Sequential(
      nn.Linear(self.embed_dim, self.embed_dim),
      nn.BatchNorm1d(self.embed_dim),
      nn.Tanh()
    )

  def forward(self, coord, seq_mask):
    """ forward pass """
    p_embed = self.param_embed(coord[:, :, :2])
    type_embeddings = self.type_embed(coord[:, :, 2]).view(coord.shape[0], coord.shape[1], 1, -1)
    combined_features = torch.cat((p_embed, type_embeddings), dim=2).flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(combined_features.flatten(0, 1)).unflatten(0, (combined_features.shape[0], combined_features.shape[1]))
    encoder_input = self.pos_embed(p_embeds.transpose(0, 1))

    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)

    # Avg pool latent code
    z_mask = (~seq_mask).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    vq_loss, quantized, encodings_flat, selection = self.codebook(code_encoded)
    latent_code = quantized.transpose(0, 1)
    return latent_code, vq_loss, selection, encodings_flat

  def count_code(self, coord, seq_mask):
    """ Codebook usage """
    p_embeds = self.param_embed(coord).flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(p_embeds.flatten(0, 1)).unflatten(0, (p_embeds.shape[0], p_embeds.shape[1]))
    encoder_input = self.pos_embed(p_embeds.transpose(0, 1))

    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)

    # Avg pool latent code
    z_mask = (~seq_mask).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    code_dist = self.codebook.count_code(code_encoded)

    return code_dist, code_encoded

class CombinedEncoderVQVAE2(nn.Module):
    def __init__(self):
      super(CombinedEncoderVQVAE2, self).__init__()
      self.embed_dim = ENCODER_CONFIG['embed_dim']

      # Shared Encoder Layers
      self.param_embed = Embedder(2 ** BIT + LOOP_PARAM_PAD, 32)
      self.bbox_embed = Embedder(2 ** BIT, 32)
      self.type_embed = Embedder(7, 32)

      self.shared_fc = nn.Sequential(
        nn.Linear(32 * PROFILE_PARAM_SEQ, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.LeakyReLU(),
      )

      self.pos_embed = PositionalEncoding(max_len=max(MAX_PROFILE, MAX_LOOP), d_model=self.embed_dim)

      self.shared_encoder = TransformerEncoder(
        TransformerEncoderLayerImproved(d_model=self.embed_dim,
                                        nhead=ENCODER_CONFIG['num_heads'],
                                        dim_feedforward=ENCODER_CONFIG['hidden_dim'],
                                        dropout=ENCODER_CONFIG['dropout_rate']),
        ENCODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

      # Bottom-level Quantizer for Loop
      self.bottom_codebook = VectorQuantizerEMA(LOOP_CODEBOOK_DIM, self.embed_dim, commitment_cost=0.25, decay=0.99)

      # Top-level Quantizer for Profile
      self.top_codebook = VectorQuantizerEMA(PROFILE_CODEBOOK_DIM, self.embed_dim, commitment_cost=0.25, decay=0.99)

    def forward(self, loop_coord, profile_coord, loop_seq_mask, profile_seq_mask):
      """ Forward pass for VQ-VAE-2 combined encoder """

      # Bottom-level (Loop)
      loop_param_embed = self.param_embed(loop_coord[:, 1:, :])
      loop_type_embeddings = self.type_embed(loop_coord[:, 0:1, :])
      loop_combined_features = torch.cat((loop_type_embeddings, loop_param_embed), dim=1).flatten(start_dim=2,
                                                                                                  end_dim=3)
      loop_features = self.shared_fc(loop_combined_features.flatten(0, 1)).unflatten(0, (
      loop_combined_features.shape[0], loop_combined_features.shape[1]))

      loop_positional = self.pos_embed(loop_features.transpose(0, 1))
      loop_outputs = self.shared_encoder(src=loop_positional, src_key_padding_mask=loop_seq_mask)

      # Bottom-level quantization for Loop
      loop_z_mask = (~loop_seq_mask).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.embed_dim)
      avg_loop_z = (loop_outputs * loop_z_mask).sum(dim=0) / loop_z_mask.sum(dim=0)
      loop_vq_loss, loop_quantized, _, _ = self.bottom_codebook(avg_loop_z.unsqueeze(0))
      loop_latent_code = loop_quantized.transpose(0, 1)

      # Top-level (Profile)
      profile_bbox_embed = self.bbox_embed(profile_coord[:, :, 1:])
      profile_type_embeddings = self.type_embed(profile_coord[:, :, 0:1])
      profile_combined_features = torch.cat((profile_type_embeddings, profile_bbox_embed), dim=2).flatten(start_dim=2,
                                                                                                          end_dim=3)
      profile_features = self.shared_fc(profile_combined_features.flatten(0, 1)).unflatten(0, (profile_combined_features.shape[0], profile_combined_features.shape[1]))

      upsampled_loop = F.interpolate(loop_quantized.transpose(0, 1), scale_factor=2, mode='nearest')
      profile_input = profile_features + upsampled_loop.transpose(0, 1)

      upsampled_loop = loop_quantized.transpose(0, 1).repeat(1, profile_features.size(1), 1)
      profile_input = profile_features + upsampled_loop


      profile_positional = self.pos_embed(profile_input.transpose(0, 1))
      profile_outputs = self.shared_encoder(src=profile_positional, src_key_padding_mask=profile_seq_mask)

      # Top-level quantization for Profile
      profile_z_mask = (~profile_seq_mask).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.embed_dim)
      avg_profile_z = (profile_outputs * profile_z_mask).sum(dim=0) / profile_z_mask.sum(dim=0)
      profile_vq_loss, profile_quantized, _, _ = self.top_codebook(avg_profile_z.unsqueeze(0))
      profile_latent_code = profile_quantized.transpose(0, 1)

      return loop_latent_code, profile_latent_code, loop_vq_loss + profile_vq_loss