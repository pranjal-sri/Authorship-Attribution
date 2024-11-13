import torch
import torch.nn as nn
import torch.nn.functional as F

HF_MASK_VALUE = -1e9

class CustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout_p = config.attention_probs_dropout_prob
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, granularity_mask=None,
                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        hidden_states = self.hidden_dropout(hidden_states)
        h = self.num_attention_heads
        B, T, C = hidden_states.shape
        k = self.key(hidden_states)
        q = self.query(hidden_states)
        v = self.value(hidden_states)
        q = q.view(B, T, h, -1).permute(0, 2, 1, 3)
        k = k.view(B, T, h, -1).permute(0, 2, 1, 3)
        v = v.view(B, T, h, -1).permute(0, 2, 1, 3)

        attention_mask_combined = torch.clamp(attention_mask + granularity_mask, min=HF_MASK_VALUE)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask_combined,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.hidden_dropout(out)
        return (out,)

class GELUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = F.gelu
    def forward(self, x):
        return self.act(x)

class CustomIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = GELUActivation()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class CustomSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class CustomOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class CustomLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomAttention(config)
        self.self_output = CustomSelfOutput(config)
        self.intermediate = CustomIntermediate(config)
        self.output = CustomOutput(config)

    def forward(self, hidden_states, attention_mask, granularity_mask):
        attention_outputs = self.attention(hidden_states, attention_mask, granularity_mask)
        attention_output = attention_outputs[0]
        self_output = self.self_output(attention_output, hidden_states)
        intermediate_output = self.intermediate(self_output)
        layer_output = self.output(intermediate_output, self_output)
        return layer_output

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([CustomLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, granularity_mask, return_all_states=True):
        all_hidden_states = () if return_all_states else None
        for i, layer_module in enumerate(self.layer):
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, granularity_mask[:, i, :, :, :])
            hidden_states = layer_outputs
        if all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_all_states:
            return {'last_state': hidden_states, 'all_states': all_hidden_states}
        else:
            return {'last_state': hidden_states} 