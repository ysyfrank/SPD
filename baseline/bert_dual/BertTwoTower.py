import torch
import torch.nn as nn
import torch.nn.init as init

class BertTwoTower(nn.Module):
    def __init__(self, bert_model):
        super(BertTwoTower, self).__init__()
        self.bert_model = bert_model
        self.fct_loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch_data):
        """
        Args:
            query_ids ([type]): [description]
            query_attention_mask ([type]): [description]
            doc_ids ([type]): [description]
            ddoc_attention_mask ([type]): [description]
        """
        
        query_ids = batch_data["query_ids"] # [bs, sl]
        query_attention_mask = batch_data["query_attention_mask"] # [bs, sl]
        query_inputs = {'input_ids': query_ids, 'attention_mask': query_attention_mask}
        query_embeddings = self.bert_model(**query_inputs)[1] # [bs, hidden_state]
        
        doc_ids = batch_data["doc_ids"] # [bs, sl]
        doc_attention_mask = batch_data["doc_attention_mask"] # [bs, sl]
        doc_inputs = {'input_ids': doc_ids, 'attention_mask': doc_attention_mask}
        doc_embeddings = self.bert_model(**doc_inputs)[1] # [bs, hidden_state]

        similarities = torch.matmul(query_embeddings, doc_embeddings.T)/0.1 # [batch_size, batch_size]

        batch_y = torch.tensor(range(query_ids.shape[0])).to(query_ids.device)
        loss = self.fct_loss(similarities, batch_y.view(-1))

        y_pred = (torch.argmax(similarities, dim=1) == torch.tensor(range(query_ids.shape[0])).to(query_ids.device)) # 0/1 in every position

        return loss, y_pred, query_embeddings