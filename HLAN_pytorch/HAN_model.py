import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    A module that computes attention over an input tensor.

    Args:
        input_size (int): The size of the input tensor.
        output_size (int): The size of the output tensor.
    """
    def __init__(self, input_size, output_size, per_label_attention=True, num_classes=50):
        super(Attention, self).__init__()
        self.per_label_attention = per_label_attention
        self.linear = nn.Linear(input_size, output_size)
        if per_label_attention:
            self.context_vector = nn.Parameter(torch.randn((num_classes, output_size)))
        else:
            self.context_vector = nn.Parameter(torch.randn((output_size)))

    def forward(self, input_tensor):
        """
        Forward pass of the Attention module.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, output_size).
        """
        hidden_representation = torch.tanh(self.linear(input_tensor))
        attention_logits = torch.sum(hidden_representation * self.context_vector, dim=-1) # this is a stupid way to write matrix multiplication

        # for numerical stability, subtract the max of the attention logits
        attention_logits_max, _ = torch.max(attention_logits, dim=-1, keepdim=True)
        attention = F.softmax(attention_logits - attention_logits_max, dim=-1)

        output = torch.sum(input_tensor * attention.unsqueeze(-1), dim=-2)
        return output
    
class AttentionPerLabelWordLevel(nn.Module):
    """
    A module that computes attention over an input tensor.

    Args:
        input_size (int): The size of the input tensor.
        output_size (int): The size of the output tensor.
    """
    def __init__(self, input_size, output_size, num_classes=50, num_sentences=100, sentence_length=25):
        super(AttentionPerLabelWordLevel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_classes = num_classes
        self.num_sentences = num_sentences
        self.sentence_length = sentence_length

        self.linear = nn.Linear(input_size, output_size)
        self.context_vector = nn.Parameter(torch.randn((num_classes, output_size)))
    
    def forward(self, input_tensor):
        """
        Forward pass of the Attention module.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            output (torch.Tensor): 
            The output tensor of shape (batch_size, num_sentences, num_classes, output_size) if per_label_attention is True, 
            else (batch_size, num_sentences, output_size).
        """
        hidden_representation = torch.tanh(self.linear(input_tensor))
 
        # split both to word and sentence level
        hidden_representation_reshaped = hidden_representation.view(-1, self.num_sentences, self.sentence_length, self.output_size)
        context_vector_expanded = self.context_vector.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
        attention_logits = torch.matmul(hidden_representation_reshaped, context_vector_expanded)
        # for numerical stability, subtract the max of the attention logits
        attention_logits_max, _ = torch.max(attention_logits, dim=-2, keepdim=True) # max along the sentence_length dimension
        attention = F.softmax(attention_logits - attention_logits_max, dim=-1)
        
        # weighted by attention
        attention_reshaped = attention.unsqueeze(-1)
        input_tensor_reshaped = input_tensor.view(-1, self.num_sentences, self.sentence_length, 1, self.output_size)
        temp = attention_reshaped * input_tensor_reshaped
        output = torch.sum(temp, dim=2)
        
        return output
    
class AttentionPerLabelSentenceLevel(nn.Module):
    """
    A module that computes attention over an input tensor.

    Args:
        input_size (int): The size of the input tensor.
        output_size (int): The size of the output tensor.
    """
    def __init__(self, input_size, output_size, num_classes=50, num_sentences=100):
        super(AttentionPerLabelSentenceLevel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_classes = num_classes
        self.num_sentences = num_sentences

        self.linear = nn.Linear(input_size, output_size)
        self.context_vector = nn.Parameter(torch.randn((num_classes, output_size)))
    
    def forward(self, input_tensor):
        """
        Forward pass of the Attention module.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, num_sentences, num_classes, input_size).

        Returns:
            output (torch.Tensor): 
            The output tensor of shape (batch_size, num_sentences, num_classes, output_size) if per_label_attention is True, 
            else (batch_size, num_sentences, output_size).
        """
        hidden_representation = torch.tanh(self.linear(input_tensor))
 
        attention_logits = torch.sum(hidden_representation * self.context_vector.unsqueeze(0).unsqueeze(0), dim=-1)
        # for numerical stability, subtract the max of the attention logits
        attention_logits_max, _ = torch.max(attention_logits, dim=-2, keepdim=True) # max along the sentence_length dimension
        attention = F.softmax(attention_logits - attention_logits_max, dim=-1)
        
        # weighted by attention
        temp = attention.unsqueeze(-1) * input_tensor
        output = torch.sum(temp, dim=1)
        
        return output
    
class HierarchicalAttentionNetwork(nn.Module):
    """
    A Hierarchical Attention Network for document classification.

    Args:
        vocab_size (int): The size of the vocabulary.
        embed_size (int): The size of the word embeddings.
        hidden_size (int): The size of the hidden state of the GRU.
        num_classes (int): The number of classes to predict.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_sentences, sentence_length, num_classes=50, per_label_attention=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_sentences = num_sentences
        self.sentence_length = sentence_length
        self.num_classes = num_classes

        self.embeddings = nn.Embedding(vocab_size, embed_size) # ok
        self.word_gru = nn.GRU(embed_size, hidden_size, bidirectional=False) # paper model only has 1 set of parameters, don't have a separate parameters set for reverse direction
        self.sentence_gru = nn.GRU(2*hidden_size, 2*hidden_size, bidirectional=False) # same as above
        self.word_attention = AttentionPerLabelWordLevel(input_size=2*hidden_size, output_size=2*hidden_size, num_classes=num_classes, num_sentences=num_sentences, sentence_length=25) # need to be separate for each class depending on input parameter per_label_attention
        self.sentence_attention = AttentionPerLabelSentenceLevel(input_size=4*hidden_size, output_size=2*hidden_size, num_classes=num_classes, num_sentences=num_sentences)
        
        # last layer, use dot product then reduce sum instead of linear layer
        self.W = nn.Parameter(torch.randn((num_classes, hidden_size*4)))
        self.b = nn.Parameter(torch.randn((num_classes)))

    def forward(self, input_tensor):
        """
        Forward pass of the Hierarchical Attention Network.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length=2500).

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, num_classes).
        """
        embedded_documents = self.embeddings(input_tensor)

        # "bidirectional" with only one set of parameters in GRU
        output_words_left2right, _ = self.word_gru(embedded_documents)
        output_words_right2left, _ = self.word_gru(embedded_documents)
        # arrange to sentence, reverse, then arrange back to original dimension
        output_words_right2left_reshaped = output_words_right2left.view(-1, self.num_sentences, self.sentence_length, self.embed_size)
        output_words_right2left_reshaped_reversed = output_words_right2left_reshaped.flip(dims=[2])
        output_words_right2left_reversed = output_words_right2left_reshaped_reversed.view(-1, self.num_sentences * self.sentence_length, self.embed_size)
        output_words = torch.cat((output_words_left2right, output_words_right2left_reversed), dim=-1)

        # word_level attention
        output_words_attn = self.word_attention(output_words)
        # reshape to 3D tensor
        output_words_attn_reshaped = output_words_attn.view(-1, self.num_sentences*self.num_classes, 2*self.hidden_size)

        # "bidirectional" sentence GRU
        output_sentences_left2right, _ = self.sentence_gru(output_words_attn_reshaped)
        output_sentences_left2right = output_sentences_left2right.view(-1, self.num_sentences, self.num_classes, 2*self.hidden_size)
        output_sentences_right2left, _ = self.sentence_gru(output_words_attn_reshaped)
        output_sentences_right2left = output_sentences_right2left.view(-1, self.num_sentences, self.num_classes, 2*self.hidden_size).flip(dims=[1]) # to check if this is correct
        output_sentences = torch.cat((output_sentences_left2right, output_sentences_right2left), dim=-1)
        
        # sentence_level attention
        output_sentences_attn = self.sentence_attention(output_sentences)

        # final classification
        output = torch.sum(output_sentences_attn *  self.W.unsqueeze(0), dim=-1) + self.b
        return output