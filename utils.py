from torchmetrics import Metric
import torch

class MSE_PSNR(Metric):
    def __init__(self, range=1.0):
        super(MSE_PSNR, self).__init__()

        self.range = torch.tensor(range)
        self.add_state("psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mse", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        mse_ = torch.mean(torch.square(torch.subtract(preds, target)))
        psnr_ = torch.multiply(10, torch.log10(torch.square(self.range) / mse_))

        self.mse = mse_
        self.psnr += psnr_

    def compute(self):
        return self.mse, self.psnr

if __name__ == '__main__':
    logits = torch.rand(32, 3, 64, 64)  # Example output images (32 samples, 3 channels, 64x64)
    labels = torch.rand(32, 3, 64, 64)

    psnr = MSE_PSNR()
    print(psnr(logits, labels))