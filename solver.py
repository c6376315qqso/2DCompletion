import model
import torchvision
import torch
import os
import copy
import numpy as np
TRAIN = 0


class Solver():
    def __init__(self, tran_loader, test_loader):
        self.tran_loader = tran_loader
        self.test_loader = test_loader
        self.net = None
        self.criterion = torch.nn.MSELoss()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        self.net = model.U_Net(img_ch=3, output_ch=3)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.net.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def predict_single(self, x):
        transform =  torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0,), (1,))
        ])
        x = transform(x)
        x = torch.unsqueeze(x, 0)
        x = x.to(self.device)
        y = self.net(x).squeeze(0)
        image_y = y.to('cpu').detach().numpy()
        # image_y = (image_y * 255).astype(np.uint8)
        image_y = image_y.transpose((1, 2, 0))
        return image_y


    def train(self):
        net_path = 'model.pkl'
        # print(self.net)
        if os.path.isfile(net_path):
            self.net.load_state_dict(torch.load(net_path))
            print('model loaded')
        if TRAIN:
            NUM_EPOCHS = 2
            for epoch in range(NUM_EPOCHS):
                self.net.train(True)
                epoch_loss = 0
                for i, (x, y) in enumerate(self.tran_loader):

                    x = x.to(self.device)
                    y = y.to(self.device)
                    try:
                        predict_x = self.net(x)
                    # print(predict_x.shape, y.shape)
                        if (predict_x.shape != y.shape):
                            print(epoch, i)
                        loss = self.criterion(predict_x, y)
                        epoch_loss += loss

                        self.net.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    except:
                        print('error in epoch %d it %d' % (epoch, i))
                        continue
                    if i % 100 == 99:
                        print('epoch[%d/%d], iteration[%d], loss: %.4f' %
                              (epoch+1, NUM_EPOCHS, i+1, epoch_loss / (i+1)))
            torch.save(self.net.state_dict(), net_path)
        self.net.train(False)
        self.net.eval()