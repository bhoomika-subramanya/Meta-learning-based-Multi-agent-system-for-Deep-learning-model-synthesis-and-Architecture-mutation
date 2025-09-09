# Design and Implementation of a Meta-Learning Based Multi-Agent System  
### for Deep Learning Model Synthesis and Architecture Mutation  

##  Overview
This project implements a **meta-learning based multi-agent framework** that automates the design, training, and evolution of deep learning architectures.  
Instead of manually engineering neural networks, our system uses **autonomous agents** to generate, mutate, and optimize models for image classification tasks such as **MNIST** and **CIFAR-10**.  

The framework demonstrates a step towards **Artificial General Intelligence (AGI)** by enabling **AI systems to design better AI systems**.  

---

##  Key Features
- **Multi-Agent System**  
  - **Planner** – Generates sophisticated architectures (LLM-powered).  
  - **CodeGen** – Produces optimized PyTorch code.  
  - **Training** – Executes robust training loops.  
  - **Evaluator** – Computes fitness (accuracy, loss).  
  - **Evolver** – Mutates architectures using evolutionary algorithms (NEAT-inspired).  
  - **Memory** – Stores successful configurations (ChromaDB-based).  
  - **Reflexion** – Refines strategies via iterative feedback (ReAct-style).  

- **Frameworks Used:**  
  - PyTorch → Model building & training  
  - DEAP → Evolutionary algorithms  
  - Streamlit → Visualization dashboard  

- **Datasets:** MNIST (handwritten digits) and CIFAR-10 (colored images).  

---

## 📂 Project Structure
├── app/ # Streamlit dashboards & visualization
├── core/ # Core agent implementations
│ ├── planner.py
│ ├── codegen.py
│ ├── training.py
│ ├── evaluator.py
│ ├── evolver.py
│ ├── memory.py
│ └── reflexion.py
├── models/ # Generated model architectures
├── data/ # Datasets (MNIST, CIFAR-10)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── main.py # Entry point to run the system


---

## Installation & Setup
1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
## Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\\Scripts\\activate    # On Windows

## Install dependencies
pip install -r requirements.txt

📊 Results
Successfully evolved CNN and FC architectures.
Achieved high accuracy on MNIST and promising results on CIFAR-10.
Demonstrated rapid adaptation of models through meta-learning.

Contribution & Future Work
Scale to larger datasets (ImageNet, multimodal data).
Introduce pooling, BatchNorm, and dropout layers in search space.
Multi-objective optimization (accuracy vs FLOPs/latency).
Cloud-based distributed orchestration for faster evolution.
