import torch
import pickle
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel

class T5TwoTower(T5EncoderModel):
    def __init__(self, t5_model, args=None):
        super(T5TwoTower, self).__init__(t5_model.config)
        self.args = args
        self.config = t5_model.config

        self.shared = t5_model.shared        
        self.encoder = t5_model.encoder

        self.linear = nn.Linear(self.config.d_model, self.config.d_model, bias=False)

        # self.linear.weight = nn.Parameter(pickle.load(open("linear.weight", "rb")))

        self.fct_loss = torch.nn.CrossEntropyLoss()

    def encode(self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # [bs, seq_len, hidden_dim]
        hidden_states = hidden_states * attention_mask.unsqueeze(2)  # mask embedding of pad tokens
        sequence_output = torch.sum(hidden_states, axis=1) / torch.sum(attention_mask, axis=1).unsqueeze(1) # [bs, hidden_dim]
        sequence_output = F.normalize(self.linear(sequence_output), p=2, dim=1)

        # sequence_output = self.projection(sequence_output)
        # sequence_output = sequence_output / (torch.norm(sequence_output, p=2, dim=-1)).unsqueeze(1) # [batch_size, hidden_dim]
        return sequence_output

    def forward(self, batch_data):
        """
        Args:
            query_ids ([type]): [description]
            query_attention_mask ([type]): [description]
            doc_ids ([type]): [description]
            doc_attention_mask ([type]): [description]
        """
        
        query_ids = batch_data["query_ids"] # [bs, sl]
        query_attention_mask = batch_data["query_attention_mask"] # [bs, sl]
        doc_ids = batch_data["doc_ids"] # [bs, sl]
        doc_attention_mask = batch_data["doc_attention_mask"] # [bs, sl]
        labels = batch_data["labels"] # 

        query_embeddings = self.encode(input_ids=query_ids, attention_mask=query_attention_mask, labels=labels) # [bs, hidden_state]
        doc_embeddings = self.encode(input_ids=doc_ids, attention_mask=doc_attention_mask, labels=labels) # [bs, hidden_state]

        similarities = torch.matmul(query_embeddings, doc_embeddings.T)/0.01 # [batch_size, batch_size] # temperature

        batch_y = torch.tensor(range(query_ids.shape[0])).to(query_ids.device)
        loss = self.fct_loss(similarities, batch_y.view(-1))

        y_pred = (torch.argmax(similarities, dim=1) == torch.tensor(range(query_ids.shape[0])).to(query_ids.device)) # 0/1 in every position

        return loss, y_pred, query_embeddings