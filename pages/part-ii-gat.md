---
layout: page
title: "Part II: Graph Attention Networks (GAT)"
---

# Part II: Graph Attention Networks (GAT)

GAT content here...

[Previous: Part I - Introduction](/pages/part-i-introduction.md/)
[Next: Part III - GAT Implementation on COBRE-CogniNet](/pages/part-iii-cobre-implementation.md/)

# Part-II: Graph Attention Networks (GAT)

**Preliminary Reading Recommendations**
Before delving into the specifics of Graph Attention Networks (GAT), it is beneficial to familiarize yourself with the basics of graph networks and attention mechanisms in neural networks. Below are some resources that provide foundational knowledge:

- **Graph Neural Networks Overview**: This article introduces the fundamental concepts of GNNs and explains their application across various data types, from images to text. [Learn more about Graph Neural Networks](https://h2o.ai/wiki/self-attention/#:~:text=Self%2Dattention%20is%20a%20mechanism,sequence%20by%20attending%20to%20itself).
- **Attention Mechanisms in Deep Learning**: This resource delves into the various types of attention mechanisms, including self-attention, which will be our focus. [Discover Attention Mechanisms](https://www.scaler.com/topics/deep-learning/attention-mechanism-deep-learning/)

**Introduction to Graph Attention Networks (GAT)**
Introduced by Petar Veličković, Guillem Cucurull, and Arantxa Casanova at ICLR 2018, GATs leverage a novel neural network architecture that incorporates stacks of self-attentional layers for node classification in graph-structured data. This approach allows the model to assign varying importance to different nodes within a neighborhood, thus enhancing the specificity and accuracy of the classification process.

**Advantages of GATs:**

1. **Operational Efficiency**: GATs eliminate the need for costly matrix operations like inversions and can be parallelized across node pairs, enhancing computational efficiency.
2. **Flexibility**: The model can adapt to nodes with varying degrees by assigning arbitrary weights to their neighbors.
3. **Inductive Learning Capability**: GATs do not require prior knowledge of the graph structure, making them suitable for inductive learning tasks where the model encounters completely unseen graphs.

**Core Concept:**
The main principle of GAT is to compute the hidden representations of each node by focusing on its neighbors through a self-attention mechanism.

**Glimpse into the Architecture:**
The GAT architecture processes input node features ℎ={ℎ~1,ℎ~2,...,ℎ~𝑁}_h_={_h_~1,_h_~2,...,_h_~_N_}, where 𝑁*N* is the number of nodes and 𝐹*F* is the number of features per node, and outputs a new set of node features. This involves:

![Untitled](<Part-II%20Graph%20Attention%20Networks%20(GAT)%20303766cead4c4b74a4e3d3f70f591f5b/Untitled.png>)

- **Linear Transformation**: Each node feature undergoes a transformation to increase feature expressiveness.
- **Self-Attention Mechanism**: Calculates un-normalized attention scores between node pairs to determine the influence of one node on another.
  ![Screenshot 2024-04-28 at 7.44.22 PM.png](<Part-II%20Graph%20Attention%20Networks%20(GAT)%20303766cead4c4b74a4e3d3f70f591f5b/Screenshot_2024-04-28_at_7.44.22_PM.png>)
  ![Screenshot 2024-04-28 at 7.39.55 PM.png](<Part-II%20Graph%20Attention%20Networks%20(GAT)%20303766cead4c4b74a4e3d3f70f591f5b/Screenshot_2024-04-28_at_7.39.55_PM.png>)
- **Normalization of Attention Scores**: Applies a softmax function to convert attention scores into a probability distribution, ensuring comparability across nodes.
  ![Screenshot 2024-04-28 at 8.14.10 PM.png](<Part-II%20Graph%20Attention%20Networks%20(GAT)%20303766cead4c4b74a4e3d3f70f591f5b/Screenshot_2024-04-28_at_8.14.10_PM.png>)
  ![Screenshot 2024-04-28 at 8.15.29 PM.png](<Part-II%20Graph%20Attention%20Networks%20(GAT)%20303766cead4c4b74a4e3d3f70f591f5b/cdd74c61-ec4e-4b04-bffa-48b6edc42875.png>)
- **Feature Aggregation**: Uses the normalized attention scores to compute a weighted sum of the features, forming the output features for each node.

**Model Evaluation:**

The effectiveness of GAT was demonstrated using different datasets, both in transductive and inductive settings. For transductive learning, citation networks like Cora, Citeseer, and PubMed were used, while for inductive learning, a protein-protein interaction dataset tested the model’s adaptability to unseen graphs. The GAT achieved a high accuracy rate, outperforming other models in various scenarios. The performance scores for multiple datasets from the table is shared below.

**The dataset structural information used in the page**

![Screenshot 2024-04-28 at 8.40.36 PM.png](<Part-II%20Graph%20Attention%20Networks%20(GAT)%20303766cead4c4b74a4e3d3f70f591f5b/Screenshot_2024-04-28_at_8.40.36_PM.png>)

**Transductive learning dataset Accuracy scores**

![Screenshot 2024-04-28 at 8.43.37 PM.png](<Part-II%20Graph%20Attention%20Networks%20(GAT)%20303766cead4c4b74a4e3d3f70f591f5b/Screenshot_2024-04-28_at_8.43.37_PM.png>)

**Inductive learning dataset accuracy scores**

![Screenshot 2024-04-28 at 8.43.15 PM.png](<Part-II%20Graph%20Attention%20Networks%20(GAT)%20303766cead4c4b74a4e3d3f70f591f5b/Screenshot_2024-04-28_at_8.43.15_PM.png>)

**Practical Implications:**

The ability of GATs to consider node relationships dynamically makes it highly applicable to diverse fields, including medical datasets involving images and clinical assessments.

**Next Steps:**

We will now explore the application of GAT on the COBRE dataset, which includes MRI images from individuals diagnosed with schizophrenia and healthy controls.

[Explore the implementation of GAT on the COBRE dataset](https://chatgpt.com/c/7833c581-fa55-4327-9f7a-2d46312f6100#) .
