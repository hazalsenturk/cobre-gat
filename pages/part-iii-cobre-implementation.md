---
layout: page
title: "Part III: GAT Implementation on COBRE-CogniNet"
permalink: /pages/part-iii-cobre-implementation/
---

# Part III: GAT Implementation on COBRE-CogniNet

COBRE implementation content here...

[Previous: Part II - Graph Attention Networks (GAT)](/pages/part-ii-gat.md/)
[Next: Part IV - COBRE Classification Model with GAT](/pages/part-iv-classification-model.md/)

# Part-III: GAT Implementation on COBRE- CogniNet

[https://github.com/utoprey/CogniNet](https://github.com/utoprey/CogniNet)

In this section, we'll delve into the remarkable research conducted by the CogniNet team from Skoltech. Their study involves a comparative analysis of functional MRI (fMRI) data from the well-established Schizophrenia unimodal dataset, using cutting-edge techniques. Additionally, they set a benchmark for machine learning methods in this area.

The analysis focuses on two datasets: the Schizophrenia dataset from the Center for Biomedical Research Excellence (COBRE) and the Autism Brain Imaging Data Exchange (ABIDE). Our discussion will center on the COBRE dataset.

We'll explore both classical machine learning and deep learning models that the study utilized, serving as a robust guide for researchers working with medical data. The key areas we'll cover include:

- Overview of the COBRE dataset
- Data preprocessing techniques
- Methods implemented in the study
- Detailed descriptions of the implementations
- Comparative performance of the various methods

Let's begin with a brief overview of the COBRE dataset. The Center for Biomedical Research Excellence (COBRE) dataset includes MRI data from 72 individuals diagnosed with schizophrenia and 75 healthy controls, all between the ages of 18 to 65. Exclusion criteria for this dataset are:

- History of neurological disorders
- Mental retardation
- Severe head trauma resulting in more than 5 minutes of unconsciousness
- Substance abuse or dependency within the previous year

The features within the dataset are labeled using the Automated Anatomical Labeling (AAL) framework.

For the deep learning models discussed in the study, there is an additional preprocessing step involved. This preprocessing pipeline includes the construction of graphs where adjacency matrices are defined, organizing the fMRI data into a graph data structure. The PyTorch Geometric library was used for this purpose.

## Classical Machine Learning Implementation

In their innovative study, the CogniNet team at Skoltech focused on optimizing classical machine learning algorithms specifically for the COBRE dataset. This involved meticulously tuning the hyperparameters to enhance model performances, below are the hyperparameters that showed the best results.

**Tuned hyperparameters of the ML algorithms for COBRE Dataset**

![Screenshot 2024-04-30 at 5.46.11 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/99f26bee-f6ac-4d47-8663-76f44dca511e.png)

**Results from Classical Machine Learning Methods:**

The team's findings, illustrated in the provided screenshots, show the comparative performance of these classical models, highlighting the standout success of the Support Vector Machine (SVM) which outperformed other models in accuracy and reliability.

![Screenshot 2024-04-30 at 5.48.14 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/Screenshot_2024-04-30_at_5.48.14_PM.png)

![Screenshot 2024-04-30 at 6.12.26 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/Screenshot_2024-04-30_at_6.12.26_PM.png)

![Screenshot 2024-04-30 at 6.16.31 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/c7250db2-987b-49c6-8353-c9d2e4ad6d38.png)

![Screenshot 2024-04-30 at 6.17.16 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/Screenshot_2024-04-30_at_6.17.16_PM.png)

**Transitioning to Advanced Deep Learning Techniques:**

Building on the classical approaches, the CogniNet study also ventured into more advanced deep learning techniques to classify fMRI images of participants as either no Schizophrenia indication or Schizophrenia diagnosed.

The motivation for exploring these advanced methods stemmed from the successful application of graph structures in medical research, which have shown significant benefits across various studies, including those on COVID-19.

These compelling results have sparked my interest in exploring Graph Neural Networks (GNNs) more deeply, particularly their applications in mental health disorders such as schizophrenia. Let's delve into the specifics of how these models were utilized in the study.

The research team employed two primary models: the GATConv, representing an implementation of Graph Neural Networks, and the Multi-Layer Perceptron (MLP). Within the context of the study's deep learning applications, it was observed that the MLPs generally performed better than the GATConv models. This insight is particularly valuable as it highlights the potential nuances and considerations when applying different types of neural networks to psychiatric datasets.

![Screenshot 2024-04-30 at 6.49.14 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/Screenshot_2024-04-30_at_6.49.14_PM.png)

The MLP model significantly outperforms GATConv in terms of the F1 score, indicating better precision and recall balance. The relatively narrower confidence interval for MLP suggests more consistent performance across different test sets.

Again, MLP shows superior performance with a higher accuracy and a tighter confidence interval than GATConv. This suggests that MLP is more reliable for correctly classifying both positive and negative cases in the dataset.

The ROC-AUC scores also favor MLP, indicating a better overall ability to distinguish between the classes at various threshold settings

![Screenshot 2024-04-30 at 8.00.41 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/Screenshot_2024-04-30_at_8.00.41_PM.png)

![Screenshot 2024-04-30 at 8.00.53 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/Screenshot_2024-04-30_at_8.00.53_PM.png)

![Screenshot 2024-04-30 at 8.01.06 PM.png](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/Screenshot_2024-04-30_at_8.01.06_PM.png)

**Deep Learning Model Insights:**

Among the deep learning techniques, the study utilized both traditional Multi-Layer Perceptrons (MLP) and innovative Graph Attention Networks (GATConv). Interestingly, MLPs performed better in this specific context, possibly due to the relatively small sample size of 147 participants, which could hinder the effectiveness of GNNs.

This MLP vs GNN performance differences has been discussed in other stories. If further interested check these out:

[](https://www.sciencedirect.com/science/article/pii/S0893608023006020)

[aclanthology.org](https://aclanthology.org/2023.acl-long.597.pdf)

Despite the superior performance of MLPs in this study, it's worth noting the limitations due to the small dataset size, which might not fully exploit the capabilities of Graph Attention Networks (GATs). GATs are particularly promising for medical data analysis, especially in larger datasets where their sophisticated attention mechanisms can provide deeper insights into complex biological structures.

Graph Attention Networks (GATs) represent a significant advancement in the field of machine learning, particularly in their application to complex data types such as those found in medical research. As we delve deeper into the capabilities of GATs, it's important to understand both their theoretical foundations and practical applications.

**Exploring the Foundations of GATs:**
For a comprehensive understanding of Graph Attention Networks, I recommend reviewing the foundational papers that discuss the original concepts and methodologies of GATs. These resources provide crucial insights into how GATs employ attention mechanisms to enhance model performance in various tasks.

- [Deep Dive into Graph Attention Networks (GAT)](https://www.notion.so/Part-II-Graph-Attention-Networks-GAT-303766cead4c4b74a4e3d3f70f591f5b?pvs=21)

**Introducing BrainGAT:**
A specific implementation detailed in the recent study is BrainGAT, which adapts GAT for analyzing neurological data. The architecture of BrainGAT:

![Untitled](Part-III%20GAT%20Implementation%20on%20COBRE-%20CogniNet%20d4692377292e49b79256bddca71d7455/Untitled.png)

- **Convolutions to Learn the Data (convs):** The model incorporates three convolutional layers utilizing the attention mechanism from the GATConv architecture with a dual-head setup. This structure was optimized based on comparative analysis of different layer architectures in the Pytorch Geometric library. Being tested on 1-2-3 layer architectures.Followed by linear transformations and LeakyReLU activation function with the slope value 0.2 for the negative values.
- **Classification Layer (fcn):** A dense layer follows the convolutional stages, classifying subjects into 'Healthy' or 'Schizophrenic' categories. This layer, taking 1600 input features and outputting two, is crucial for identifying subtle patterns not immediately apparent.

**Understanding Sequential Modules in GNNs:**
Sequential modules in PyTorch are used to build modular GNN architectures. They ensure that each layer processes inputs derived from the output of its predecessor, thereby maintaining a specific execution order. For more details:

[torch_geometric.nn — pytorch_geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)

**Conclusions on Model Selection:**
This study exemplifies the data-dependent nature of model or architecture selection. The performance of different models with varied hyperparameters can vary significantly based on data size, type, and quality. Graph Attention Networks, with their flexible implementation across different data types, are particularly fascinating for their ability to uncover relational information between data points through their attention mechanisms.

**Innovative Applications in Biological Research:**
Professor Marinka Zitnik from Harvard University discusses cutting-edge ML/AI applications in biological data. She highlights two key findings:

1. **Drug Repurposing:** The use of subgraphs and GNN architecture in drug interaction maps has revealed that certain medications can be repurposed for other diseases—a revelation that became particularly relevant during the COVID-19 pandemic.
2. **Drug Interaction Mechanisms:** GNNs have shown that the effects of drugs might be mediated through indirect interactions rather than direct protein-receptor engagements, suggesting a paradigm shift in understanding drug mechanisms.

These insights not only demonstrate the potential of GNNs in advancing drug development and repurposing but also highlight their broader applicability in reshaping approaches to disease treatment and understanding complex biological interactions.

If you are interested in the full discussion:

[https://youtu.be/9nACsKGisOI?si=A0pBIlbkP0LftBF5](https://youtu.be/9nACsKGisOI?si=A0pBIlbkP0LftBF5)
