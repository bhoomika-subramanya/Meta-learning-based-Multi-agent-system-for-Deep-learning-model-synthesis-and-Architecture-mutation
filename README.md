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
