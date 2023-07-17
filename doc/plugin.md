
## QKVToContext  

BERT's model architecture is a multi-layer bidirectional Transformer encoder.   

https://github.com/NVIDIA/TensorRT/issues/2655#issue-1575787745   
https://developer.nvidia.com/zh-cn/blog/real-time-nlp-with-bert-using-tensorrt-updated/   

* 对应的 python 实现
```
class BertSelfAttention(nn.Module):
    distillation : Final[bool]
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        #Distillation specific
        self.distillation = getattr(config, 'distillation', False)
        if self.distillation:
            self.distill_state_dict = OrderedDict()
            self.distill_config = config.distillation_config
        else :
            self.distill_config = { 'use_attention_scores' : False, 'use_value_states' : False }

    def transpose_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        return x

    def transpose_key_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).permute(1, 2, 0)
        return x

    def forward(self, hidden_states, attention_mask):
        # (seq, bsz, hidden)
        batch_size = hidden_states.size(1)
        seq_length = hidden_states.size(0)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.bmm(query_layer, key_layer)
        # (bsz, heads, seq, seq)
        attention_scores = attention_scores.view(batch_size,
                                                 self.num_attention_heads,
                                                 seq_length, seq_length)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # (bsz, heads, seq, seq)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.view(batch_size * self.num_attention_heads,
                                               seq_length, seq_length)

        context_layer = torch.bmm(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 1).contiguous()
        # (seq, bsz, hidden)
        context_layer = context_layer.view(seq_length, batch_size, self.all_head_size)

        #Cache states if running distillation
        if self.distillation:
            if self.distill_config["use_attention_scores"]:
                self.distill_state_dict["attention_scores"] = attention_scores
            if self.distill_config["use_value_states"]:
                self.distill_state_dict["value_states"] = context_layer
        return context_layer

```

* onnx 子图
  

* 输入参数说明


* kernel 优化说明


