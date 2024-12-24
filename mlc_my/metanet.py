# Define the SeparateConvMetaNet without padding
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaNet(nn.Module):
    def __init__(self, seq_len =768,max_shift=50, feature_dim=16, kernel_size=3,batch_size=64):
        super(MetaNet, self).__init__()
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.max_shift=max_shift

        # Feature extractors for output and label
        self.feature_extractor_out = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size),  # No padding
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size),  # No padding
            nn.ReLU()
        )
        self.feature_extractor_label = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size),  # No padding
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size),  # No padding
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(feature_dim) 
        # Linear transformations for h_out and h_label
        self.linear_out_L = nn.Linear(seq_len-4, 1)
        self.linear_label_L = nn.Linear(seq_len-4, 1)
        self.linear_label = nn.Linear(feature_dim, feature_dim)
        self.linear_out = nn.Linear(feature_dim, feature_dim)
     

        # Regressor to predict t
        self.regressor = nn.Sequential(
            nn.Linear(3*feature_dim , 32),  # Input dimension = feature_dim + cosine similarity
            # nn.Linear(1, 32), 
            nn.ReLU(),
            nn.Linear(32, 1)  # Output shift t
        )
        

    def compute_similarity(self, h_out, h_label): 
        # h_out_norm = F.normalize(h_out, p=2, dim=1)  # Normalize along feature dimension
        # h_label_norm = F.normalize(h_label, p=2, dim=1)  # Normalize along feature dimension
        similarity = torch.sum(h_out * h_label, dim=-1)  # Compute similarity along feature dim
 
        return similarity  

    def forward(self, y_out, y_label):
        b,_,_ =y_out.shape
        # Extract features
        h_out = self.feature_extractor_out(y_out)  # (batch_size, feature_dim, seq_len_out)
        h_label = self.feature_extractor_label(y_label)  # (batch_size, feature_dim, seq_len_label)

        # Compute cosine similarity along the sequence dimension
        similarity = self.compute_similarity(h_out, h_label)  # (batch_size, 1, seq_len)
        similarity = self.layer_norm(similarity)
    
        h_out = self.linear_out_L(h_out).squeeze(-1)
        h_label = self.linear_out_L(h_label).squeeze(-1)
        h_out = self.linear_out(h_out )  # (batch_size, feature_dim)
        h_label = self.linear_label(h_label )  # (batch_size, feature_dim)


        combined_features = torch.cat([similarity , h_out, h_label], dim=1)
        t = self.regressor(combined_features)  # (batch_size, 1) 
        # t = torch.tanh(t)
        t  = t * self.max_shift
        t = t.clamp(min=-self.max_shift, max=self.max_shift).round().long()
        return t.squeeze(-1)  # Output as a 1D tensor
    

# class MetaNetPseudoLabel(nn.Module):
#     def __init__(self, seq_len =768,max_shift=50, feature_dim=16, kernel_size=3,batch_size=64):
#         super(MetaNetPseudoLabel, self).__init__()
#         self.kernel_size = kernel_size
#         self.seq_len = seq_len
#         self.max_shift=max_shift

#         # Feature extractors for output and label
#         self.feature_extractor_out = nn.Sequential(
#             nn.Conv1d(1, feature_dim, kernel_size=kernel_size),  # No padding
#             # nn.ReLU(),
#             nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size),  # No padding
#             nn.ReLU()
#         )
#         self.feature_extractor_label = nn.Sequential(
#             nn.Conv1d(1, feature_dim, kernel_size=kernel_size),  # No padding
#             # nn.ReLU(),
#             nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size),  # No padding
#             nn.ReLU() 
#         )

#         self.deconv = nn.Sequential(
#             nn.ConvTranspose1d(feature_dim*2, feature_dim, kernel_size=3, stride=1, padding=1),
#             # nn.ReLU(),
#             nn.ConvTranspose1d(feature_dim, 1, kernel_size=3, stride=1, padding=1)
#         ) 
        
 
import torch.nn as nn

class MetaNetPseudoLabel(nn.Module):
    def __init__(self, seq_len=768, max_shift=50, feature_dim=16, kernel_size=3, batch_size=64):
        super(MetaNetPseudoLabel, self).__init__()
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.max_shift = max_shift

        # Feature extractors for output and label
        self.feature_extractor_out = nn.Sequential(
            nn.ReplicationPad1d(padding=(kernel_size // 2)),
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size, padding=0),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=(kernel_size // 2)),
 
        )
        self.feature_extractor_label = nn.Sequential(
            nn.ReplicationPad1d(padding=(kernel_size // 2)),
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size, padding=0),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=(kernel_size // 2)),
 
        )

        # Ensure the deconvolution outputs the same shape
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(feature_dim * 2, feature_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ConvTranspose1d(feature_dim, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
        )

    def forward(self, x_out, x_label):
        # Extract features
        features_out = self.feature_extractor_out(x_out)
        features_label = self.feature_extractor_label(x_label)
        
        # Concatenate features
        combined_features = torch.cat((features_out, features_label), dim=1)
        
        # Decode features
        output = self.deconv(combined_features)
        return output

    def forward(self, y_out, y_label):
        b,_,_ =y_out.shape
        # Extract features
        h_out = self.feature_extractor_out(y_out)  # (batch_size, feature_dim, seq_len_out)
        h_label = self.feature_extractor_label(y_label)  # (batch_size, feature_dim, seq_len_label)

        

        combined_features = torch.cat([ h_out, h_label], dim=1)
        pseudolabel = self.deconv(combined_features)  # (batch_size, 1) 
        # t = torch.tanh(t)
         
        return pseudolabel



class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MetaNetLoss(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MetaNetLoss, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    b,c,l=4,1,768
    x = torch.randn((b,c,l))
    y = torch.randn((b,c,l))
    model = MetaNetPseudoLabel(batch_size=b)
    t = model(x,y)
    print(t.shape)

