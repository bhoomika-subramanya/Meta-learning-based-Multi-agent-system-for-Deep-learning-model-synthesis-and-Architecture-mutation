import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import copy
from typing import List, Dict, Tuple
import uuid

# Define a neural network block
class NeuralBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, block_type: str = "fc"):
        super(NeuralBlock, self).__init__()
        self.block_type = block_type
        self.out_features = out_features  # Store out_features for mutation
        if block_type == "fc":
            self.layer = nn.Linear(in_features, out_features)
            self.activation = nn.ReLU()
        elif block_type == "conv":
            self.layer = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

# Neural Network Architecture
class NeuralArchitecture(nn.Module):
    def __init__(self, blocks: List[Dict], input_channels: int, input_dim: Tuple[int, int], output_dim: int):
        super(NeuralArchitecture, self).__init__()
        self.blocks = nn.ModuleList()
        self.input_dim = input_dim
        current_channels = input_channels  # For conv layers
        current_dim = input_dim[0] * input_dim[1]  # For fc layers (e.g., 28*28 for MNIST)
        self.input_is_conv = blocks and blocks[0]["type"] == "conv"
        
        for i, block_config in enumerate(blocks):
            block_type = block_config["type"]
            out_features = block_config["out_features"]
            if block_type == "conv":
                block = NeuralBlock(
                    in_features=current_channels,
                    out_features=out_features,
                    block_type=block_type
                )
                current_channels = out_features
                current_dim = input_dim[0] * input_dim[1] * current_channels  # Update for potential fc transition
            else:  # fc
                # If previous block was conv, flatten the input features
                if i > 0 and blocks[i-1]["type"] == "conv":
                    in_features = current_channels * input_dim[0] * input_dim[1]
                else:
                    in_features = current_dim
                block = NeuralBlock(
                    in_features=in_features,
                    out_features=out_features,
                    block_type=block_type
                )
                current_dim = out_features
                current_channels = out_features  # Update for potential conv transition
            self.blocks.append(block)
        
        # Output layer: Check the last block's type
        if blocks and blocks[-1]["type"] == "conv":
            self.flatten = nn.Flatten()
            self.output_layer = nn.Linear(current_channels * input_dim[0] * input_dim[1], output_dim)
        else:
            self.flatten = None
            self.output_layer = nn.Linear(current_dim, output_dim)
    
    def forward(self, x):
        if self.input_is_conv:
            x = x.view(-1, 1, self.input_dim[0], self.input_dim[1])  # Ensure conv input shape
        else:
            x = x.view(-1, self.input_dim[0] * self.input_dim[1])  # Flatten for fc
        
        for i, block in enumerate(self.blocks):
            if i > 0 and block.block_type == "fc" and self.blocks[i-1].block_type == "conv":
                x = x.view(x.size(0), -1)  # Flatten before fc block
            x = block(x)
        
        if self.flatten is not None:
            x = self.flatten(x)
        x = self.output_layer(x)
        return x

# Model Generator Agent
class ModelGeneratorAgent:
    def __init__(self):
        self.architecture_space = {
            "block_types": ["fc", "conv"],
            "out_features": [16, 32, 64, 128, 256]
        }
    
    def generate_architecture(self, input_channels: int, input_dim: Tuple[int, int], output_dim: int, max_blocks: int = 5) -> NeuralArchitecture:
        num_blocks = random.randint(1, max_blocks)
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                block_type = random.choice(self.architecture_space["block_types"])
            else:
                if blocks[i-1]["type"] == "fc":
                    block_type = "fc"
                else:
                    block_type = random.choice(self.architecture_space["block_types"])
            out_features = random.choice(self.architecture_space["out_features"])
            blocks.append({"type": block_type, "out_features": out_features})
        return NeuralArchitecture(blocks, input_channels, input_dim, output_dim)

# Evaluator Agent
class EvaluatorAgent:
    def __init__(self, dataset: str = "MNIST"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.train_loader, self.test_loader = self._load_dataset()
    
    def _load_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        return train_loader, test_loader
    
    def evaluate(self, model: NeuralArchitecture, epochs: int = 1) -> float:
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        model.train()
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        return accuracy

# Mutator Agent
class MutatorAgent:
    def __init__(self):
        self.mutation_rate = 0.3
    
    def mutate_architecture(self, architecture: NeuralArchitecture) -> NeuralArchitecture:
        # Create block configurations from NeuralBlock objects
        blocks = [
            {"type": block.block_type, "out_features": block.out_features}
            for block in architecture.blocks
        ]
        blocks = copy.deepcopy(blocks)  # Deep copy the list of dictionaries
        input_channels = 1  # MNIST input channels
        input_dim = (28, 28)  # MNIST input dimensions
        output_dim = 10  # MNIST classes
        
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(["add", "remove", "modify"])
            if mutation_type == "add" and len(blocks) < 5:
                if not blocks or blocks[-1]["type"] != "fc":
                    new_block_type = random.choice(["fc", "conv"])
                else:
                    new_block_type = "fc"
                new_block = {
                    "type": new_block_type,
                    "out_features": random.choice([16, 32, 64, 128, 256])
                }
                blocks.append(new_block)
            elif mutation_type == "remove" and len(blocks) > 1:
                blocks.pop(random.randint(0, len(blocks) - 1))
            elif mutation_type == "modify" and blocks:
                block_idx = random.randint(0, len(blocks) - 1)
                new_type = random.choice(["fc", "conv"])
                if block_idx > 0 and blocks[block_idx-1]["type"] == "fc":
                    new_type = "fc"  # Prevent conv after fc
                blocks[block_idx]["type"] = new_type
                blocks[block_idx]["out_features"] = random.choice([16, 32, 64, 128, 256])
        
        return NeuralArchitecture(blocks, input_channels, input_dim, output_dim)

# Meta-Learning Controller
class MetaLearningController:
    def __init__(self, num_iterations: int = 10):
        self.generator = ModelGeneratorAgent()
        self.evaluator = EvaluatorAgent()
        self.mutator = MutatorAgent()
        self.num_iterations = num_iterations
        self.best_architecture = None
        self.best_accuracy = 0.0
    
    def run(self):
        input_channels = 1  # MNIST input channels
        input_dim = (28, 28)  # MNIST input dimensions
        output_dim = 10  # MNIST classes
        population = [self.generator.generate_architecture(input_channels, input_dim, output_dim) for _ in range(5)]
        
        for iteration in range(self.num_iterations):
            scores = []
            for arch in population:
                accuracy = self.evaluator.evaluate(arch)
                scores.append((arch, accuracy))
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_architecture = copy.deepcopy(arch)
            
            scores.sort(key=lambda x: x[1], reverse=True)
            population = [arch for arch, _ in scores[:3]]
            
            new_population = [copy.deepcopy(arch) for arch in population]
            while len(new_population) < 5:
                parent = random.choice(population)
                mutated = self.mutator.mutate_architecture(parent)
                new_population.append(mutated)
            
            population = new_population
            print(f"Iteration {iteration + 1}, Best Accuracy: {self.best_accuracy:.4f}")
        
        return self.best_architecture, self.best_accuracy

# Main execution
if __name__ == "__main__":
    controller = MetaLearningController(num_iterations=5)
    best_architecture, best_accuracy = controller.run()
    print(f"Final Best Accuracy: {best_accuracy:.4f}")