
import torch.nn as nn

class CRFCombinedModel(nn.Module):
    def __init__(self, base_model, crf):
        super(CRFCombinedModel, self).__init__()
        self.base_model = base_model
        self.crf = crf

    def forward(self, x, x_text=None, spatial_spacings=None):
        logits,reg_loss = self.base_model(x, x_text) 
        shape_img = logits.shape
        if len(shape_img) == 3:
            logits = logits.reshape(shape_img[0], 1, shape_img[1], shape_img[2])
        output = self.crf(logits, spatial_spacings=spatial_spacings)
        print(output.shape)
        return output , reg_loss
