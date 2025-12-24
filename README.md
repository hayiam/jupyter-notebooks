# Rag_impoved_3.ipynb - sample Multi-Agent Code Generation with Online RAG

This file is an interactive Python notebook demonstrating an **enhanced multi-agent code generation system** powered by Retrieval-Augmented Generation (RAG) and online search capabilities. The system generates, executes, critiques, and iteratively improves code based on user requests.

## 🚀 Overview

The system implements a collaborative multi-agent architecture where different agents work together to:
1. **Generate** code based on user requests using local knowledge and online searches
2. **Execute** the generated code to verify functionality
3. **Critique** the code quality and suggest improvements
4. **Iteratively refine** the code over multiple improvement cycles

## 📋 System Components

### 1. EnhancedRAGSystem
- Manages a knowledge base of programming concepts
- Performs semantic search using cosine similarity
- Integrates online search for real-time information
- Supports both local and online knowledge sources

### 2. CodeGenerator
- Generates Python/TensorFlow/NumPy code based on user queries
- Uses RAG context for informed generation
- Incorporates online search results when needed

### 3. CodeExecutor
- Safely executes generated code in isolated environments
- Captures execution output and errors
- Provides feedback on code functionality

### 4. CodeCritic
- Analyzes generated code for quality issues
- Suggests improvements based on best practices
- Uses RAG context for informed critiques

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages: `requests`, `numpy`, `scikit-learn`, `tensorflow`, `beautifulsoup4`


# Best_path.ipynb - neural Network Path Analysis and Enhancement

## 📁 Project Overview
This Jupyter notebook implements a novel neural network architecture with **path-aware learning** capabilities. The model can identify the strongest information pathways through a network and dynamically enhance them during training, potentially improving learning efficiency and performance.

## 🧠 Key Features

### 1. **PathAwareDense Layer**
- Custom Keras layer with neuron value tracking
- Monitors activation statistics during training
- Computes combined neuron importance metrics (activation ratio + weight magnitude)

### 2. **PathAwareModel Architecture**
- Extends Keras Sequential model
- Builds a graph representation of the neural network
- Identifies the strongest paths from input to output neurons
- Dynamically enhances weights along optimal paths

### 3. **Path Analysis Capabilities**
- Constructs directed graphs representing neuron connections
- Calculates path strength based on connection weights and neuron values
- Visualizes network structure and information flow

## 🛠 Technical Implementation

### Dependencies
```python
numpy
pandas
tensorflow/keras
networkx
scikit-learn
matplotlib
seaborn
```

### Core Components

#### **PathAwareDense Layer**
- Tracks neuron activation counts and values
- Updates neuron importance during training
- Incorporates weight magnitude in importance calculations

#### **Path Analysis Methods**
- `build_path_network()`: Creates graph representation
- `find_strongest_path()`: Identifies optimal network pathways
- `enhance_path_weights()`: Boosts weights along strong paths

## 📊 Training Process

The model implements a unique training loop that:
1. Trains on mini-batches
2. Identifies strongest network paths
3. Enhances weights along those paths
4. Tracks metrics including path strength evolution

### Performance Metrics Tracked
- Training and validation accuracy
- Training and validation loss
- Path strength over epochs

## 📈 Results Visualization

The notebook includes plotting capabilities for:
- Accuracy/loss curves during training
- Path strength evolution over epochs
- Network graph visualization (optional)

## 🔬 Research Implications

This implementation explores:
- Dynamic network adaptation during training
- Quantitative path strength metrics
- Relationship between activation patterns and network efficiency
- Potential for reduced training time through focused enhancement

# Adaptive_follow_path.ipynb and Final_adaptive.ipynb - adaptive Neural Network with Path Tracking, deep analysis

This files contains an implementation of an adaptive neural network with path tracking capabilities for analyzing learning dynamics during training.

The code demonstrates how custom layer implementations can provide deep insights into the black box of neural network learning, making the training process more transparent and interpretable.

# Cycle_network.ipynb - cycle Network: Neural Network with Path Awareness and Dynamic Architecture Adjustment

This file contains an experimental implementation of a neural network model that features **path-aware learning** and **dynamic architecture adjustment** capabilities. The model automatically monitors its learning process, identifies strong information pathways, and adapts its architecture in real-time to optimize performance.

## Key Features

### 🔍 **Path-Aware Learning**
- Custom `PathAwareDense` layer that tracks neuron activations and information value
- **Neuron value metrics**: Combines activation frequency and weight magnitude
- **Real-time path analysis**: Identifies the strongest information pathways through the network
- **Automatic weight enhancement**: Strengthens weights along stable pathways

### 🏗️ **Dynamic Architecture Adjustment**
- **Automatic layer addition**: Adds new hidden layers based on path strength metrics
- **Batch size optimization**: Dynamically adjusts batch size during training
- **Speed monitoring**: Tracks training speed and detects performance degradation
- **Multi-restart capability**: Supports up to 10 training restarts with different architectures

### 📊 **Comprehensive Monitoring**
- **Path strength tracking**: Monitors the strength of information pathways over time
- **Stability counters**: Tracks short-term and long-term stability of network paths
- **Performance metrics**: Records accuracy, loss, and training speed for each epoch
- **Visualization-ready data**: All metrics are stored for analysis and plotting

## Architecture

### Core Components

1. **`PathAwareDense` Layer**
   - Custom Keras layer with neuron value tracking
   - Activation counting and weight magnitude analysis
   - Information value calculation (70% activation ratio + 30% weight magnitude)

2. **`PathAwareModel` Class**
   - Extends Keras Sequential model
   - NetworkX-based graph representation of neural pathways
   - Automatic path discovery and analysis
   - Dynamic architecture modification

3. **Training Pipeline**
   - Automatic stability detection (short-term: 6 epochs, long-term: 6 epochs)
   - Speed monitoring every 3 epochs
   - Batch size adjustment (16 → 32 → 64)
   - Layer addition based on path strength

## Training Process

The model implements an **adaptive training loop** with these key steps:

1. **Initial Training**: Starts with 2 hidden layers (12 and 8 neurons)
2. **Path Analysis**: Identifies strongest pathways after each epoch
3. **Stability Check**: Monitors path stability over time
4. **Weight Enhancement**: Boosts weights along stable paths
5. **Architecture Adjustment**: Adds new layers when path strength indicates capacity is needed
6. **Speed Monitoring**: Adjusts batch size based on training speed
7. **Restart Logic**: Restarts training with new architecture when long-term stability is lost

### Training Parameters
- **Max restarts**: 10
- **Speed degradation threshold**: 20%
- **Min speed**: 400 samples/sec
- **Speed check interval**: 3 epochs
- **Stability thresholds**: 6 epochs (short), 6 epochs (long)

## Usage

### Data Generation
The implementation includes synthetic data generation:
- 1000 samples with 20 features
- Uniform distribution from -1 to 1
- Random binary classification labels
- 80/20 train/test split with validation subset

## Technical Details

### Dependencies
- TensorFlow 2.x
- NetworkX
- NumPy
- Matplotlib
- scikit-learn

## Research Implications

This implementation demonstrates several novel concepts:
1. **Path-based network analysis**: Treating neural networks as information flow graphs
2. **Dynamic architecture evolution**: Growing networks based on internal metrics
3. **Stability-driven training**: Using path stability as a learning signal
4. **Speed-aware optimization**: Balancing performance with computational efficiency


# Labirinth_transformer.ipynb - Maze Transformer: Pathfinding in 2D and 3D Mazes with Transformer Networks
This project implements Transformer-based neural networks for solving pathfinding problems in 2D and 3D mazes. The models learn to navigate from a start position to a goal by predicting optimal actions at each step.

## Features
2D Maze Generation: DFS-based maze generation with adjustable complexity

3D Maze Generation: Randomized Prim's algorithm for 3D maze creation

Transformer Architecture: Custom Transformer networks for 2D and 3D environments

Multi-dimensional Support: Works with both 2D and 3D spatial environments

Visualization: Interactive 3D visualization using Plotly

Pathfinding Algorithms: A* algorithm for optimal path generation

Training Pipeline: Complete training and evaluation pipeline

## Project Structure
The notebook contains several main components:

### 1. 2D Maze Components
Maze Generation: DFS algorithm for creating solvable 2D mazes

2D Transformer: Transformer model with positional encoding for 2D coordinates

2D Tokenizer: Converts maze states to token sequences with agent/goal positions

### 2. 3D Maze Components
3D Maze Generation: Randomized Prim's algorithm for 3D maze creation

3D Transformer: Extended Transformer for 3D environments with 6 possible actions

3D Tokenizer: Handles 3D coordinate systems and maze representation

### 3. Core Algorithms
Pathfinding: Optimal path calculation for training data generation

Data Generation: Automatic generation of training samples from maze solutions

Training Pipeline: Complete training loop with batching and optimization

### 4. Visualization
3D Maze Visualization: Interactive Plotly visualizations showing:

Maze walls (gray cubes)

Free space (colored by distance from start)

Agent path (red line)

Start (green diamond) and goal (blue cross) positions

