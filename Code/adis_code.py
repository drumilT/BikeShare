##CNN in pytorch
##Please check for required imports


#################### Model Class ############################

in_ch = 1
h_ch1 = 5
h_ch2 = 3
out_ch = 1
kernel_size = 5
strides = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, h_ch1, kernel_size = kernel_size)
        self.conv2 = nn.Conv2d(h_ch1, h_ch2, kernel_size =
kernel_size, padding = 2)
        self.conv3 = nn.Conv2d(h_ch2, out_ch, kernel_size =
kernel_size, padding = 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
        return x

#################### Data Loader #################################

class BikeshareDataset(Dataset):
    def __init__(self):
#         xy = np.loadtxt('', delimiter = '', dtype = np.float32)
        fin = open('all_data_in.bin',"rb")
        fout = open("all_data_out.bin","rb")
        data_in = np.array(pickle.load(fin))
        data_out = np.array(pickle.load(fout))
        data_in = np.expand_dims(data_in, axis=1)
        data_out = np.expand_dims(data_out, axis=1)
#         print(data_in.shape)
#         print(data_out.shape)
        self.len = data_in.shape[0]
        self.x_data = torch.from_numpy(data_in).float()
        self.y_data = torch.from_numpy(data_out).float()
#         self.x_data,self.y_data=self.x_data.type(torch.DoubleTensor),self.y_data.type(torch.DoubleTensor)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
dataset = BikeshareDataset()
train_loader = DataLoader(dataset = dataset, batch_size = 730, shuffle = True)


#################### Training Function ##################################

def train(model, train_loader, optimizer, epoch):
    model.train
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            #output = model(data)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            #if batch_idx % 10 == 0:
            print('Train Epoch: {} \tStep: {} \tLoss: {}'.format(ep,
batch_idx, loss.item()))


##########################################################################


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
ep = 10
train(model, train_loader, optimizer, ep)