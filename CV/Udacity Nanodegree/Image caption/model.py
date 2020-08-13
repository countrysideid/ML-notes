import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
#         self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional = True)
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)
#         self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
        self.embed_size = embed_size
        
        self.hidden_size = hidden_size
        
#         self.sigmoid = nn.Sigmoid()
        
        self.num_layers =num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #initialize weights
        self.init_weights()
        
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, features, captions):

#         batch_size = features.size(0)
        captions = captions[:,:-1]
        captions =  self.embedding(captions) 
        inputs = torch.cat((features.unsqueeze(1), captions),dim=1)
        output,_ = self.lstm(inputs)
        output = self.linear(output)
        return output
    
#     def init_hidden(self):
# #         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
# #                 torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
#         weight = next(self.parameters()).data
#         hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_().to(device),
#                       weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_().to(device))
#         return hidden
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        flag = 0
        while (flag <max_len):
            output,_ = self.lstm(inputs, states)
            output = self.linear(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output,1)
            if predicted_index == 1:
                break
            
#             outputs.append(predicted_index.cpu().numpy()[0].item())
            outputs.append(predicted_index.item())
            flag += 1
            inputs = self.embedding(predicted_index).unsqueeze(1)
        return outputs
    
#         res = []
#         for i in range(max_len):
#             outputs, hidden = self.lstm(inputs, states)
# #             print('lstm output shape ', outputs.shape)
# #             print('lstm output.squeeze(1) shape ', outputs.squeeze(1).shape)
#             outputs = self.linear(outputs.squeeze(1))
# #             print('linear output shape ', outputs.shape)
#             target_index = outputs.max(1)[1]
# #             print('target_index shape ', target_index.shape)
#             res.append(target_index.item())
#             inputs = self.embedding(target_index).unsqueeze(1)
# #             print('new inputs shape ', inputs.shape, '\n')
#         return res
    
    
