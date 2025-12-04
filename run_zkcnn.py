import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# Add project root to sys.path to allow importing zkcnn without installation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zkcnn import ZKProver, LeNetCifar, export_model

def train_and_run_proof():
    dataset_name = "CIFAR10"
    num_classes = 10
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    
    if dataset_name == "CIFAR100":
        num_classes = 100

    print(f"Preparing Data for {dataset_name}...")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Fake dataset for demo stability
    class FakeCIFAR10(torch.utils.data.Dataset):
        def __init__(self, size=100, transform=None):
            self.data = torch.randn(size, 3, 32, 32)
            self.targets = torch.randint(0, num_classes, (size,))
            self.transform = transform
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    trainset = FakeCIFAR10(size=100)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    testset = FakeCIFAR10(size=10)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)

    net = LeNetCifar(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Training (1 epoch for demo)...")
    for epoch in range(1):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print('Finished Training')

    # Get a sample image from test set
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Run inference in PyTorch for comparison
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print(f"PyTorch Prediction: {predicted.item()}")
    
    # Export for zkCNN
    # Ensure output dir exists
    os.makedirs("output/data", exist_ok=True)
    data_file = f'output/data/{dataset_name.lower()}_data.txt'
    result_file = f'output/data/{dataset_name.lower()}_result.csv'
    
    print(f"Exporting to {data_file}...")
    export_model(net, images[0], data_file)
    print("Export done.")
    
    # --- PROVE AND VERIFY ---
    print("\n--- Starting Zero-Knowledge Proof ---")
    prover = ZKProver()
    # model_type must match what the C++ CLI expects
    success = prover.prove(data_file, result_file, model_type="lenetCifar", num_classes=num_classes)
    
    if success:
        print("\n[SUCCESS] Proof Verified!")
        # Optional: Read back result
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                res = f.read().strip()
                print(f"zkCNN Inference Result: {res}")
                if str(res) == str(predicted.item()):
                    print("MATCH: PyTorch and zkCNN results agree.")
                else:
                    print("MISMATCH: PyTorch and zkCNN results differ (likely due to quantization effects).")

if __name__ == "__main__":
    train_and_run_proof()
