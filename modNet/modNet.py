import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from commpy.modulation import PSKModem, QAMModem

M = 16
modem = PSKModem(M)

class modNet(nn.Module):
    def __init__(self):
        super(modNet, self).__init__()
        self.fc1 = nn.Linear(int(np.log2(M)), 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class demodNet(nn.Module):
    def __init__(self):
        super(demodNet, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc2a = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, int(np.log2(M)))
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2a(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

# Generate tx symbols of size M
tx_data = np.random.randint(0, 2, int(np.log2(M)) * 1000).reshape(-1, int(np.log2(M)))
tx_data = torch.tensor(tx_data).float().view(-1, 1)

enc = modNet()
dec = demodNet()

opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.01)

SNRs = np.linspace(15, 20, 5)

test_ser_m = np.zeros(len(SNRs))
test_ser_b = np.zeros(len(SNRs))

for iter_snr in range(len(SNRs)):
    SNR_dB = SNRs[iter_snr]
    noise_std = np.sqrt(1 / (2 * 10 ** (SNR_dB / 10)))

    for i in range(500):
        opt.zero_grad()
        
        tx_data = np.random.randint(0, 2, int(np.log2(M)) * 10000).reshape(-1, int(np.log2(M)))
        tx_data = torch.tensor(tx_data).float()
        
        tx_sym = enc(tx_data)
        noise = torch.randn_like(tx_sym) * noise_std

        tx_sym = tx_sym / torch.sqrt(torch.sum(tx_sym ** 2, dim=1, keepdim=True))
        rx_sym = dec(tx_sym + noise)
        
        loss = torch.nn.BCELoss()(rx_sym, tx_data)
        loss.backward()
        
        opt.step()
        
        if i % 200 == 0:
            print(f"iter: {i} and loss: {loss.item():.4f}")

    # Test the model
    tx_data = np.random.randint(0, 2, int(np.log2(M)) * 1000).reshape(-1, int(np.log2(M)))

    tx_data = torch.tensor(tx_data).float()

    tx_sym = enc(tx_data)
    tx_sym = tx_sym / torch.sqrt(torch.sum(tx_sym ** 2, dim=1, keepdim=True))

    noise = torch.randn_like(tx_sym) * noise_std
    rx_sym = dec(tx_sym + noise)
    rx_sym = torch.round(rx_sym)

    tx_data = tx_data.detach().numpy().flatten()
    rx_sym = rx_sym.detach().numpy().flatten()

    test_ser_m[iter_snr] = np.mean((tx_data != rx_sym).astype(int)).item()


    # Testing with QAM Modulation
    tx_data_b = np.random.randint(0, 2, int(np.log2(M)) * 1000)

    tx_data = modem.modulate(tx_data_b) / np.sqrt(modem.Es)

    noise = (np.random.randn(1000) + 1j * np.random.randn(1000)) * noise_std
    rx_sym = modem.demodulate(tx_data + noise, demod_type='hard')

    test_ser_b[iter_snr] = np.mean((tx_data_b != rx_sym).astype(int)).item()

plt.plot(SNRs, test_ser_m)
plt.plot(SNRs, test_ser_b)
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate")
plt.legend(["Learned Modulation", "QAM"])
plt.show()


# Visualize the tx symbols

num_bits = int(np.log2(M))
integers = np.arange(M)
tx_data = np.array((integers[:, None] & (1 << np.arange(num_bits)[::-1])) > 0).astype(int).reshape(-1, num_bits)

tx_data = torch.tensor(tx_data).float()
tx_sym = enc(tx_data)
tx_sym = tx_sym / torch.sqrt(torch.sum(tx_sym ** 2, dim=1, keepdim=True))
tx_sym = tx_sym.detach().numpy()

fig = plt.figure()
for i in range(M):
    plt.scatter(tx_sym[i, 0], tx_sym[i, 1], label=f"Symbol {i}")

plt.xlabel("I")
plt.ylabel("Q")
plt.title("Learned Tx Symbols")
plt.show()

# Visualize Reference Tx Symbols
const = modem.constellation / np.sqrt(modem.Es)
modem.plot_constellation()
fig = plt.figure()
for i in range(M):
    plt.scatter(const[i].real, const[i].imag, label=f"Symbol {i}")

plt.xlabel("I")
plt.ylabel("Q")
plt.title("Reference Tx Symbols")
plt.show()