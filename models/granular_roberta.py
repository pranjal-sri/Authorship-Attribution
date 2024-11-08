import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from types import SimpleNamespace
from .encoder import Encoder

class CustomRobertaModel(nn.Module):
    def __init__(self, config, model_to_adapt=None):
        super().__init__()
        self.embeddings = model_to_adapt.embeddings
        self.encoder = Encoder(config)

        if model_to_adapt is not None:
            self.initialize_base_roberta(model_to_adapt)

        self.n_layers = len(self.encoder.layer)

    def initialize_base_roberta(self, model_to_adapt):
        for i in range(len(self.encoder.layer)):
            self.encoder.layer[i].attention.load_state_dict(model_to_adapt.encoder.layer[i].attention.self.state_dict())
            self.encoder.layer[i].self_output.load_state_dict(model_to_adapt.encoder.layer[i].attention.output.state_dict())
            self.encoder.layer[i].intermediate.load_state_dict(model_to_adapt.encoder.layer[i].intermediate.state_dict())
            self.encoder.layer[i].output.load_state_dict(model_to_adapt.encoder.layer[i].output.state_dict())

    def forward(self, input_ids=None, attention_mask=None, granularity_mask=None):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        embedding_output = self.embeddings(input_ids=input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if granularity_mask is not None:
            assert granularity_mask.shape == (batch_size, self.n_layers, seq_length, seq_length)
        else:
            granularity_mask = torch.ones((batch_size, self.n_layers, seq_length, seq_length), device=device)
        granularity_mask = granularity_mask[:, :, None, :, :]
        granularity_mask = granularity_mask.to(dtype=torch.float)
        granularity_mask = (1.0 - granularity_mask) * -1e9

        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, granularity_mask)
        sequence_output = encoder_outputs
        return {'encoder_output': sequence_output, 'pooled_output': None}

class GranularRoberta(nn.Module):
  def __init__(self):
    super().__init__()
    roberta_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
    r_config = self.generate_roberta_config()
    self.roberta = CustomRobertaModel(r_config, roberta_model)
    del roberta_model
    self.linear = nn.Linear(r_config.hidden_size , r_config.hidden_size)

  @staticmethod
  def generate_roberta_config():
    roberta_config = {
    "vocab_size": 50265,
    "hidden_size": 768,
    "pad_token_id": 1,
    "max_position_embeddings": 514,
    "type_vocab_size": 1,
    "layer_norm_eps": 1e-5,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "num_hidden_layers": 6,
    "num_attention_heads": 12,
    "attention_probs_dropout_prob": 0.1,
    }

    return SimpleNamespace(**roberta_config)


  def set_trainable_layers(self, mode="linear_only", num_encoder_layers=None):
    """
    Set specific layers of the model as trainable or non-trainable.

    Args:
        mode (str): One of ["linear_only", "last_n_encoder", "full"]
        num_encoder_layers (int, optional): Number of encoder layers to unfreeze from the end
                                          when mode is "last_n_encoder"
    """
    # First freeze everything
    for param in self.parameters():
      param.requires_grad = False
      param.grad = None

    if mode == "linear_only":
      # Only train the final linear layer
      for param in self.linear.parameters():
          param.requires_grad = True

    elif mode == "last_n_encoder":
      if num_encoder_layers is None or num_encoder_layers < 1:
          raise ValueError("num_encoder_layers must be a positive integer")

      # Enable the linear layer
      for param in self.linear.parameters():
          param.requires_grad = True

      # Enable the last n encoder layers
      total_encoder_layers = len(self.roberta.encoder.layer)
      if num_encoder_layers > total_encoder_layers:
          raise ValueError(f"num_encoder_layers ({num_encoder_layers}) cannot be greater than "
                          f"total encoder layers ({total_encoder_layers})")

      start_idx = total_encoder_layers - num_encoder_layers
      for i in range(start_idx, total_encoder_layers):
          for param in self.roberta.encoder.layer[i].parameters():
              param.requires_grad = True

    elif mode == "full":
      # Enable all parameters
      for param in self.parameters():
          param.requires_grad = True

    else:
        raise ValueError("Invalid mode. Choose from: 'linear_only', 'last_n_encoder', 'full'")

    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in self.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")


  def mean_reduce_comment_embeddings(self, comment_embeds, attention_mask_encoder, eps = 1e-7):
    B, N, T, E = comment_embeds.shape
    assert attention_mask_encoder.shape == (B, N, T), f"Attention mask encoder must have the shape ({(B, N, T)}), received: {attention_mask_encoder.shape}"
    attention_mask_weights = attention_mask_encoder.view(B, N, T, 1)
    attention_mask_weights_sum = torch.clamp(attention_mask_weights.sum(dim = 2), min = eps) # B, N, 1
    return (comment_embeds * attention_mask_weights).sum(dim = 2) / attention_mask_weights_sum # B, N, E

  def attention_pooling(self, embeds: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Applies self-attention followed by max pooling to generate pooled embeddings.

    Args:
        embeds: Input embeddings of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Optional attention mask of shape (batch_size, seq_len)
                       with 1s for valid positions and 0s for padding

    Returns:
        attention_pooled: Pooled embeddings of shape (batch_size, hidden_dim)
    """
    # Get dimensions
    batch_size, seq_len, hidden_dim = embeds.shape

    # Create default attention mask if none provided
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, seq_len, device=embeds.device)

    # Prepare inputs for scaled dot product attention
    query = embeds.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim)
    key = embeds.unsqueeze(1)    # (batch_size, 1, seq_len, hidden_dim)
    value = embeds.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim)

    # Prepare attention mask
    attn_mask = attention_mask.bool()
    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

    # Calculate attention
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False
    )  # (batch_size, 1, seq_len, hidden_dim)

    # Remove head dimension
    attended = attended.squeeze(1)  # (batch_size, seq_len, hidden_dim)

    # Apply max pooling over sequence length dimension
    # First apply mask to ensure padded positions don't affect max
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(attended)
    attended_masked = attended * mask_expanded
    attended_masked[~mask_expanded.bool()] = float('-inf')

    # Max pool over sequence length
    attention_pooled = torch.max(attended_masked, dim=1)[0]  # (batch_size, hidden_dim)

    return attention_pooled

  def forward(self, input_ids, attention_mask, granularity_mask = None, episode_mask = None):
    B, N, T = input_ids.shape # eac input is a batch of episodes, where each episode is a sequence of T length
    if granularity_mask is None:
      granularity_mask = torch.ones((B, N, self.roberta.n_layers, T, T), device=input_ids.device)
    else:
      assert granularity_mask.shape == (B, N, self.roberta.n_layers, T, T), f"Granularity mask must have the shape ({(B, N, T, T)}), received: {granularity_mask.shape}"

    if attention_mask is None:
      attention_mask = torch.ones((B, N, T), device=input_ids.device)
    else:
      assert attention_mask.shape == (B, N, T), f"Attention mask must have the shape ({(B, N, T)}), received: {attention_mask.shape}"

    input_ids = input_ids.view(B*N, T)
    attention_mask = attention_mask.view(B*N, T)
    granularity_mask = granularity_mask.view(B*N, self.roberta.n_layers, T, T)

    roberta_output =  self.roberta(input_ids, attention_mask, granularity_mask)
    comment_embeddings = roberta_output['encoder_output']['last_state'].view(B, N, T, -1)

    if torch.isnan(comment_embeddings).any():
      print("\t\tcomment_embeddings is NaN!")

    attention_mask = attention_mask.view(B, N, T)


    if episode_mask is None:
      episode_mask = torch.ones((B, N), device=input_ids.device)
    else:
      assert episode_mask.shape == (B, N), f"Episode mask must have the shape ({(B, N)}), received: {episode_mask.shape}"


    episode_embeddings = self.mean_reduce_comment_embeddings(comment_embeddings, attention_mask) # B, N, E
    if torch.isnan(episode_embeddings).any():
      print("\t\tepisode_embeddings is NaN!")

    author_embeddings = self.attention_pooling(episode_embeddings, episode_mask)
    if torch.isnan(author_embeddings).any():
      print("\t\tauthor_embeddings is NaN after attention!")
    author_embeddings = self.linear(author_embeddings)
    if torch.isnan(author_embeddings).any():
      print("\t\tauthor_embeddings is NaN after linear!")
#    return author_embeddings, episode_embeddings, comment_embeddings
    return author_embeddings