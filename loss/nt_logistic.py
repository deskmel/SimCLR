import torch
import numpy as np

class NTLogisticLoss(torch.nn.Module):
    def __init__(self,device,batch_size,temperature,use_cosine_similarity):
        super(NTLogisticLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.activation = torch.nn.Sigmoid()
    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)
    
    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
    
    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    def forward(self,zis,zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)*-1
        positives /= self.temperature
        positives= torch.log(self.activation(positives))
        negatives/= self.temperature
        negatives = torch.log(self.activation(negatives))
        logits = positives +negatives
        #print(logits.size())
        #rint(negatives.size())
        #logits = torch.cat((positives, negatives), dim=1)
        #logits /= self.temperature
        #logits = self.activation(logits)
        #logits = torch.log(logits)
        loss = torch.sum(logits)
        return -loss / (4*(self.batch_size-1)*self.batch_size)
if __name__ == "__main__":
    Loss = NTLogisticLoss('cpu',6,0.5,True)
    print(Loss.mask_samples_from_same_repr)
    xi = torch.rand((6,10))
    xj = torch.rand((6,10))
    loss = Loss(xi,xj)
    print(loss)