---
layout: page
title: "Part IV: COBRE Classification Model with GAT"
permalink: /cobre-gat/pages/part-iv-classification-model/
---

# Part IV: COBRE Classification Model with GAT

Classification model content here...

[Previous: Part III - GAT Implementation on COBRE-CogniNet](/cobre-gat/pages/part-iii-cobre-implementation/)

# Part-IV: COBRE Classification Model with GAT

In today's tutorial, we'll dive into a practical guide on how to implement a Graph Attention Network (GAT) using the COBRE dataset. If you're not familiar with GATs yet, don't worry—I'll explain them in simple terms so you can grasp the concept quickly. You can find more detailed information about GATs [here](https://www.notion.so/Part-II-Graph-Attention-Networks-GAT-303766cead4c4b74a4e3d3f70f591f5b?pvs=21).

This hands-on session is based on a notebook originally shared by the CogniNet team, which you can access [here](https://www.notion.so/Deprecated-Part-III-GAT-Implementation-on-COBRE-CogniNet-28a04b2853af4e0f9fd88cbb2eb47e14?pvs=21). I've tweaked the original code to optimize it further, allowing us to gather more performance data and create visualizations to enhance our understanding of this incredible technology.

CogniNet is a sophisticated tool that helps categorize individuals in the COBRE dataset as either healthy or within the schizophrenia spectrum. It does this by applying advanced machine learning methods to neuroimaging data. Specifically, CogniNet uses GATs to analyze connectivity matrices derived from MRI scans. These techniques give us a deeper insight into brain connectivity patterns, which can be key indicators of mental health conditions. As such, CogniNet is an invaluable asset for researchers exploring schizophrenia and related disorders. Join me as we explore how this powerful tool works and what it can do!

Before we start coding, let's set up our environment:

### **Getting Ready:**

1. **Download the COBRE folder:** First, you need to get the COBRE folder from the CogniNet repository. Make sure you maintain the directory structure as it is. If you choose to change it, remember to update the import paths in your code accordingly.
2. **Set up Google Colab:** Open Google Colab to create a new notebook. This will be our workspace where we'll write and run our code.

Here's what the directory structure in the COBRE folder looks like:

![Screenshot 2024-05-08 at 2.24.46 PM.png](Part-IV%20COBRE%20Classification%20Model%20with%20GAT%20b70cf5e2f7d5483f8808f08e03803340/Screenshot_2024-05-08_at_2.24.46_PM.png)

### **Installation Steps:**

1. **Install Necessary Libraries:**

   - We'll need to install several libraries to get started: PyTorch, torch-scatter, torch-sparse, torch-cluster, and PyTorch Geometric.
   - **Tip:** Installation might take some time (approximately 30 minutes to an hour if you're using a CPU), so feel free to take a coffee break while you wait!

   ```python
   !pip install torch
   !pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
   !pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
   !pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
   !pip install git+https://github.com/pyg-team/pytorch_geometric.git
   !pip install torch_geometric
   ```

2. **Import the Libraries:**

   - Once the installations are complete, we'll import the necessary modules into our notebook.

   ```python
   import os
   import scipy
   import json
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   from scipy import io
   from typing import Union, Any, Optional, List, Dict, Tuple
   from tqdm import tqdm

   from dataclasses import dataclass, field
   from pathlib import Path

   import torch
   from torch import nn
   from torch.nn import functional as F
   from torch import Tensor
   from torch.nn import Parameter, Linear

   from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, auc
   from sklearn.preprocessing import label_binarize

   import torch_geometric
   import torch_geometric as pyg
   from torch_geometric.typing import Size, OptTensor
   from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing, GATConv, GCNConv, PNAConv
   from torch_geometric.nn import Sequential as pygSequential
   from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
   from torch_geometric.nn.inits import glorot, zeros
   from torch_geometric.utils import softmax
   from torch_geometric.data import Data, InMemoryDataset
   from torch_geometric.loader import DataLoader
   import seaborn as sns

   ```

Now that we have everything set up, we're ready to dive into the coding part! Let's explore how these tools can help us analyze the neuroimaging data from the COBRE dataset.

Alright, let's jump into the coding section! We'll explore how to leverage these tools to analyze the neuroimaging data from the COBRE dataset.

### **Setting Up the Environment:**

1. **Check Your Setup:**
   - First, we'll check the versions of PyTorch and PyTorch Geometric and ensure we're using the GPU (if available) or default to CPU otherwise.
     ```python
     print('torch', torch.__version__)
     print('torch_geometric', torch_geometric.__version__)
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print(device)
     ```
2. **Prepare the Data:**

- The MRI images from the COBRE dataset have been pre-processed and labeled according to the Automated Anatomical Labeling (AAL) atlas. We'll work with the connectivity matrices, which serve as the feature matrices derived from these MRI images.
  ```python
  cobre_csvs_path =  '../cobre/aal/'
  ```
- **Convert Connectivity Matrices to PyTorch Geometric Data:**

  - We'll write functions to transform the connectivity matrices into a format suitable for PyTorch Geometric.

  ```python
  def cm_to_edges(cm: np.ndarray):
      """
      Convert CM to (edge_index, edge_weights) of a fully connected weighted graph
      (including self-loops with zero weights)
      return: (edge_index, edge_weights)
      """
      cm = torch.FloatTensor(cm)
      index = (torch.isnan(cm) == 0).nonzero(as_tuple=True)
      edge_attr = torch.abs(cm[index])

      return torch.stack(index, dim=0), edge_attr

  def prepare_pyg_data(
      cm: np.ndarray,
      subj_id: str,
      targets: int,
  ) -> Data:

      # fully connected graph
      n = cm.shape[0]
      edge_index, edge_attr = cm_to_edges(cm)

      # compute initial node embeddings -> just original weights
      x = torch.from_numpy(cm).to(device)

      # get encoded labels
      y = F.one_hot(torch.tensor([targets-1]), num_classes=2).to(device) # we took targets-1 because targets in abide are 1 and 2

      data = Data(
          edge_index=edge_index,
          edge_attr=edge_attr,
          x=x,
          num_nodes=n,
          y=y,
          subj_id=subj_id,
      )

      return data
  ```

### **Analyzing the Dataset:**

1. **Explore the Targets:**

   - Let's look at the first few entries of the target data from the COBRE dataset, which includes details like age, gender, diagnosis, imaging technology, and protocol.
     ```python
     cobre_y = pd.read_csv('cobre/cobre_targets.tsv', delimiter='\t')
     cobre_y.head()
     ```

   The data contains information about the subjects age, gender, diagnosis (Dx), imaging technology and protocol. See the table below:

   ![Screenshot 2024-05-08 at 3.04.19 PM.png](Part-IV%20COBRE%20Classification%20Model%20with%20GAT%20b70cf5e2f7d5483f8808f08e03803340/Screenshot_2024-05-08_at_3.04.19_PM.png)

1. **Data Splitting:**

- We use predefined splits for training and testing, loading these from a JSON file.
- **Reading and Preparing Data:**

  - We'll read the feature matrices and prepare the data for training.

  ```python
  cobre_splits = json.load(open('cobre/cobre_splits_new.json')) # here we also use the same train test splits as for classic ML
  train_ids = [i['train']+i['valid'] for i in cobre_splits['train']]
  train_ids = train_ids[0]+train_ids[1]+train_ids[2]+train_ids[3]+train_ids[4]
  train_ids = list(set(train_ids))
  test_ids = cobre_splits['test']
  group = np.array(cobre_y.Dx)
  ```

  ```python
  X = []
  Y = []
  subj_ids = []
  missed_ids = []
  for subj, y in zip(train_ids+test_ids, group):
      try:
          X.append(np.array(pd.read_csv('cobre/aal/sub-'+subj+'.csv').drop('Unnamed: 0', axis=1)))
          Y.append(y)
          subj_ids.append(str(subj))
      except:
          missed_ids.append(str(subj))

  X = np.array(X)
  Y = np.array(Y)

  dict_X = dict(zip(subj_ids, X))
  dict_Y = dict(zip(cobre_y[cobre_y.Subjectid.isin(subj_ids)].Subjectid, cobre_y[cobre_y.Subjectid.isin(subj_ids)].Dx))
  xx, yy = dict_Y.keys(), dict_Y.values()
  yy = [1 if i == 'Schizophrenia_Strict' else 2 for i in yy]
  dict_Y = dict(zip(xx, yy))
  ```

1. **Setting Up Data Loaders:**

- Create data loaders for both PyTorch Geometric and traditional neural network formats, as they require different handling.
  [Why two different loaders: ](https://www.notion.so/Why-two-different-loaders-b89832494055443b94099d7c88fed961?pvs=21)

  ```python
  # creation of PyG dataloaders and common dataloaders
  # PyG data has specific batching

  train_inps, train_tgts = [], []
  train_loader_pyg = []
  for x, subj_id, y in zip(X, train_ids, Y):
      train_loader_pyg.append(prepare_pyg_data(dict_X[subj_id], subj_id, dict_Y[subj_id]))
      train_inps.append(dict_X[subj_id])
      train_tgts.append(F.one_hot(torch.tensor([dict_Y[subj_id]-1]), num_classes=2).to(device))

  test_inps, test_tgts = [], []
  test_loader_pyg = []
  for x, subj_id, y in zip(X, test_ids, Y):
      test_loader_pyg.append(prepare_pyg_data(dict_X[subj_id], subj_id, dict_Y[subj_id]))
      test_inps.append(dict_X[subj_id])
      test_tgts.append(F.one_hot(torch.tensor([dict_Y[subj_id]-1]), num_classes=2).to(device))

  dataset = torch.utils.data.TensorDataset(torch.Tensor(train_inps).to(device), torch.cat(train_tgts, axis=0).to(device))
  train_loader_cobre = torch.utils.data.DataLoader(dataset, batch_size=16)
  dataset = torch.utils.data.TensorDataset(torch.Tensor(test_inps).to(device), torch.cat(test_tgts, axis=0).to(device))
  test_loader_cobre = torch.utils.data.DataLoader(dataset, batch_size=16)

  train_loader_cobre_pyg = DataLoader(train_loader_pyg, batch_size=8)
  test_loader_cobre_pyg = DataLoader(test_loader_pyg, batch_size=8)
  ```

Let's proceed to verify the dimensions of the data loaded into our training dataset loaders:

1. **Check the Data Shape:**

   - For the traditional neural network format:

     ```python
     next(iter(train_loader_cobre))[0].shape, next(iter(train_loader_cobre))[1].shape

     # Output: (torch.Size([16, 116, 116]), torch.Size([16, 2]))
     ```

   - For the PyTorch Geometric format:

     ```python

     next(iter(train_loader_cobre_pyg))  # x

     # Output: DataBatch(x=[928, 116], edge_index=[2, 107648], edge_attr=[107648], y=[8, 2], num_nodes=928, subj_id=[8], batch=[928], ptr=[9])
     ```

Now, let’s write functions to train our model and evaluate its performance across various metrics like F1 Score, Accuracy, and ROC-AUC.

### **Writing Training and Evaluation Functions:**

1. **Evaluate Function:**

   - This function will handle the model evaluation, calculating loss and performance metrics.

   ```python
   def evaluate(model, device, loader):
       model.eval()  # Set model to evaluation mode

       loss_all = 0
       y_pred = []
       y_true = []

       with torch.no_grad():
           for data in loader:
               data = data.to(device)
               out = model(data)

               # Calculate binary cross-entropy from logits
               loss = F.binary_cross_entropy_with_logits(out, data.y.float().to(device))
               loss_all += loss.item()

               # Convert logits to probabilities using sigmoid
               probabilities = out.detach().cpu().numpy()
               y_pred.extend(probabilities)
               y_true.extend(data.y.detach().cpu().numpy())

       y_pred = np.array(y_pred)
       y_true = np.array(y_true)

       f1, acc, roc_auc = f1_score(y_true.argmax(-1), y_pred.argmax(-1)), accuracy_score(y_true.argmax(-1), y_pred.argmax(-1)), roc_auc_score(y_true.argmax(-1), y_pred[:, 1])

       return loss_all/len(loader.dataset), f1, acc, roc_auc, y_pred, y_true
   ```

2. **Train and Evaluate Function:**

   - This function manages both training and evaluation, iterating over epochs and updating model weights.

   ```python
   def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler=None, device='cuda', n_epochs=20, verbose=True):
       model.train()  # Set model to train mode

       # Initialize lists to store metrics and final predictions
       metrics = {
           'train_f1s': [],
           'train_accs': [],
           'train_roc_aucs': [],
           'train_losses': [],
           'test_f1s': [],
           'test_accs': [],
           'test_roc_aucs': [],
           'test_losses': [],
           'final_test_y_pred': None,
           'final_test_y_true': None
       }

       for i in range(n_epochs):
           loss_all = 0
           for data in train_loader:
               data = data.to(device)
               optimizer.zero_grad()
               out = model(data)
               loss = F.cross_entropy(out, data.y.to(device).float())
               loss.backward()
               optimizer.step()
               loss_all += loss.item()

           if scheduler:
               scheduler.step()

           epoch_loss = loss_all / len(train_loader.dataset)
           metrics['train_losses'].append(epoch_loss)

           train_loss, train_f1, train_acc, train_roc_auc, _, _ = evaluate(model, device, train_loader)
           metrics['train_f1s'].append(train_f1)
           metrics['train_accs'].append(train_acc)
           metrics['train_roc_aucs'].append(train_roc_auc)

           test_loss, test_f1, test_acc, test_roc_auc, test_y_pred, test_y_true = evaluate(model, device, test_loader)
           metrics['test_losses'].append(test_loss)
           metrics['test_f1s'].append(test_f1)
           metrics['test_accs'].append(test_acc)
           metrics['test_roc_aucs'].append(test_roc_auc)

           if i == n_epochs - 1:  # Save predictions from the last epoch
               metrics['final_test_y_pred'] = test_y_pred
               metrics['final_test_y_true'] = test_y_true

           if verbose:
               print(f'(Epoch {i}) | Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.2f}%')

       return metrics
   ```

   These functions provide a comprehensive view of our model's performance and are crucial for understanding how well our model can predict the different classifications within the COBRE dataset. Next, we'll set up our model configuration and proceed with training.

Next, let’s build the MLP (Multilayer Perceptron) layers that will be used in the BrainGAT architecture, termed Neurograph, which is based on benchmarks for graph machine learning in brain connectomics. We'll start by defining our MLP layer configuration using Python's **`dataclasses`** for better structure and readability.

### **Building the MLP Layers for BrainGAT:**

1. **Define the MLPlayer Class:**

   - This class will hold the configuration for a single MLP layer.

   ```python
   @dataclass
   class MLPlayer:
       """ Config of one MLP layer """
       out_size: int = 10
       act_func: Optional[str] = 'ReLU'
       act_func_params: Optional[dict] = None
       dropout: Optional[float] = None
   ```

2. **Define the MLPConfig Class:**

   - This class will be used to define the configuration of the entire MLP model.

   ```python
   @dataclass
   class MLPConfig:
       """ Config of MLP model """
       # layers define only hidden dimensions, so final MLP will have n+1 layer.
       # So, if you want to create a 1-layer network, just leave layers empty

       # in and out sizes are optional and usually depend on upstream model and the task
       # for now, they are ignored
       in_size: Optional[int] = None
       out_size: Optional[int] = None

       # act func for the last layer. None -> no activation function
       act_func: Optional[str] = None
       act_func_params: Optional[dict] = None
       layers: List[MLPlayer] = field(default_factory=lambda: [
           MLPlayer(
               out_size=32, dropout=0.6, act_func='LeakyReLU', act_func_params=dict(negative_slope=0.2)
           ),
           MLPlayer(
               out_size=32, dropout=0.6, act_func='LeakyReLU', act_func_params=dict(negative_slope=0.2)
           ),
       ])
   ```

3. **Function to Build an MLP Layer:**

   - This function takes the input size and a layer configuration, and constructs an **`nn.Sequential`** model representing the layer.

     ```python
      def build_mlp_layer(in_size: int, layer: MLPlayer) -> nn.Sequential:
         """ Factory that returns nn.Sequential from input size and MLPlayer """
         act_params = layer.act_func_params if layer.act_func_params else {}

         lst: list[nn.Module] = [nn.Linear(in_size, layer.out_size)]
         #lst.append(
         if layer.act_func:
             lst.append(nn.LeakyReLU(**act_params)) # available_activations[layer.act_func](**act_params))
         if layer.dropout:
             lst.append(nn.Dropout(layer.dropout, inplace=True))
         return nn.Sequential(*lst)

     ```

   With these configurations and functions, we can efficiently set up our MLP layers to be integrated into the Neurograph architecture. This structured approach not only ensures clarity in setting up each layer according to specified configurations but also facilitates easy modifications and scalability.

### **Proceeding to Model Configuration:**

Now that we have the fundamental building blocks for our MLP and Graph Attention Network (GAT) layers, let's look at how to integrate these components into a complete model architecture called BrainGAT, which uses both the GAT layers and MLP layers in its structure. This architecture is particularly designed for handling graph-based data in brain connectomics.

### **Integrating GAT and MLP in the BrainGAT Architecture:**

1. **Model Configuration Class:**

   - This class is a blueprint that provides essential information about the model.

   ```python
   @dataclass
   class ModelConfig:
       """ Base class for model config """
       name: str  # see neurograph.models/
       n_classes: int  # must match with loss

       # required for correct init of models
       # see `train.train.init_model`

       data_type: str
   ```

2. **BasicMLP Class:**

- A straightforward MLP implementation, this class constructs a neural network from the provided configuration.

```python
class BasicMLP(nn.Module):
    """ Basic MLP class """
    def __init__(self, in_size: int, out_size: int, config: MLPConfig):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.net = nn.Sequential()
        curr_size = self.in_size
        for layer_conf in config.layers:
            subnet = build_mlp_layer(curr_size, layer_conf)
            self.net.append(subnet)
            curr_size = layer_conf.out_size

        # the last layer
        self.net.append(build_mlp_layer(
            curr_size,
            MLPlayer(
                out_size=self.out_size,
                act_func=config.act_func,
                act_func_params=config.act_func_params,
            ),
        ))

    def forward(self, x):
        return self.net(x)
```

1. **Building GAT Blocks:**

- This function assembles the layers for a GAT block using the PyTorch Geometric library.

```python
def build_gat_block(
    input_dim: int,
    hidden_dim: int,
    proj_dim: Optional[int] = None,
    num_heads: int = 1,
    dropout: float = 0.0,
    use_batchnorm: bool = True,
    use_abs_weight: bool = True,
):
    proj_dim = hidden_dim if proj_dim is None else proj_dim
    return pygSequential(
        'x, edge_index, edge_attr',
        [
            (
                GATConv(
                    input_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                ),
                'x, edge_index, edge_attr -> x'
            ),
            nn.Linear(hidden_dim * num_heads, proj_dim),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Dropout(p=dropout)
        ]
    )
```

1. **Concatenative Pooling Function:**

- This function handles the pooling of node features into a single graph-level representation, essential for tasks involving whole-graph outputs.

```python
def concat_pool(x: torch.Tensor, input_dim: int, num_nodes: int) -> torch.Tensor:
    # NB: x must be a batch of xs
    return x.reshape((x.shape[0] // input_dim, input_dim*x.shape[-1]))
```

With these definitions, we have the necessary components to construct a model that can handle both node-level and graph-level predictions, suitable for complex tasks such as analyzing brain connectivity data in neuroscientific studies. The next step would involve integrating these components into a comprehensive model setup and preparing for training and evaluation, which includes managing data input/output, configuring training loops, and setting up evaluation metrics.

Let's continue by setting up the BrainGAT model, preparing it for training, and running the training process. This model integrates graph attention mechanisms with traditional MLP layers to analyze complex brain connectivity data.

### **Setting Up the BrainGAT Model**

1. **Create the BrainGAT Model Class:**
   - This class constructs a neural network that uses Graph Attention Networks (GAT) along with an MLP for classification.

```python
class BrainGAT(nn.Module):
    def __init__(
        self,
        # determined by dataset
        input_dim: int,
        num_nodes: int,
        num_classes: int
    ):
        """
        Architecture:
            - a list of GATConv blocks (n-1)
            - the last layer GATConv block with diff final embedding size
            - (prepool projection layer)
            - pooling -> graph embeddgin
            - fcn clf
        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.input_dim = input_dim
        self.num_nodes = num_nodes

        num_classes = num_classes
        hidden_dim = 32

        num_layers = 3

        use_batchnorm = True
        use_abs_weight = True
        # mp_type = 'node_concate'
        dropout = 0

        num_heads = 2

        # pack a bunch of convs into a ModuleList
        gat_input_dim = input_dim
        for _ in range(num_layers - 1):
            conv = build_gat_block(
                gat_input_dim,
                hidden_dim,
                proj_dim=None,
                # mp_type=mp_type,
                num_heads=num_heads,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                use_abs_weight=use_abs_weight,
            )
            # update current input_dim
            gat_input_dim = hidden_dim
            self.convs.append(conv)

        # gat block return embeddings of `inter_dim` size
        conv = build_gat_block(
            gat_input_dim,
            hidden_dim,
            proj_dim=64,
            # mp_type=mp_type,
            num_heads=num_heads,
            dropout=dropout,
            use_batchnorm=False,  # batchnorm is applied in prepool layer
            use_abs_weight=use_abs_weight,
        )
        # add extra projection and batchnorm
        self.prepool = nn.Sequential(
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(8) if use_batchnorm else nn.Identity(),
        )

        fcn_dim = 8 * num_nodes

        # add last conv layer
        self.convs.append(conv)

        self.fcn = BasicMLP(in_size=fcn_dim, out_size=num_classes, config=MLPConfig())

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        z = x
        z = z.float()

        # apply conv layers
        for _, conv in enumerate(self.convs):
            z = conv(z, edge_index, edge_attr)

        # prepool dim reduction
        z = self.prepool(z)

        z = nn.BatchNorm1d(num_features=z.shape[-1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)(z)

        # pooling
        z = concat_pool(z, self.input_dim, self.num_nodes)
        # FCN clf on graph embedding
        z = nn.BatchNorm1d(num_features=z.shape[-1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)(z)
        out = self.fcn(z)
        out=nn.Softmax(dim=-1)(out)
        return out
```

1. **Initialize the Model and Set Up the Training Environment:**

- This step involves configuring the model for training, setting up the optimizer, and specifying the device (GPU or CPU).

```python
bgbGAT = BrainGAT(116, 116, 2)
bgbGAT = bgbGAT.to(device)
bgbGAT = bgbGAT.float()
optimizer = torch.optim.AdamW(bgbGAT.parameters(), lr=0.001)
```

1. **Run the Training and Evaluation Process:**

- Start the training process, adjusting hyperparameters as needed, and evaluate the model's performance.

```python
metrics = train_and_evaluate(
    bgbGAT,
    train_loader_cobre_pyg,
    test_loader_cobre_pyg,
    optimizer,
    scheduler=None,
    device=device,
    n_epochs=60
)
```

1. **Enhance Statistical Robustness through Multiple Runs:**

- To ensure the model's performance is robust, run the training multiple times and store the results for analysis.

```python
def train_and_collect_data():
    res_bgbgat_cobre = []
    for _ in range(10):
        bgbGAT = BrainGAT(116, 116, 2)
        bgbGAT = bgbGAT.to(device)
        bgbGAT = bgbGAT.float()
        optimizer = torch.optim.AdamW(bgbGAT.parameters(), lr=0.001)
        res = train_and_evaluate(
            bgbGAT,
            train_loader_cobre_pyg,
            test_loader_cobre_pyg,
            optimizer,
            scheduler=None,
            device=device,
            n_epochs=60,
            verbose=False
        )
        res_bgbgat_cobre.append(res)
    return res_bgbgat_cobre

res_bgbgat_cobre = train_and_collect_data()
```

By setting up the model in this way, we ensure that it is well-prepared to handle the complexities of analyzing brain connectivity data. Additionally, multiple runs help validate the consistency of the model's performance, accounting for variability due to factors like random initialization.

### **Model Performance Overview**

To provide a transparent and detailed view of how our BrainGAT model performs, I've decided to directly share the plots generated during the performance analysis phase. This approach not only saves us from wading through repetitive code but also focuses on the visual outcomes, which are often more intuitive and insightful.

In our ROC curve analysis, the model achieved an AUC (Area Under the Curve) value of 0.74. This indicates a good level of model performance, especially when compared to a baseline AUC of 0.50, which represents random chance. The AUC of 0.50 is what you would expect from a model that makes predictions randomly, with no actual discernment between classes.

The AUC value of 0.74 shows that our model has a significantly better ability to distinguish between patients with schizophrenia and those from other groups (healthy or schizoaffective). This higher than chance performance confirms that the model is learning relevant patterns from the data, which are effective in identifying the schizophrenia phenotype.

![Screenshot 2024-05-08 at 6.20.20 PM.png](Part-IV%20COBRE%20Classification%20Model%20with%20GAT%20b70cf5e2f7d5483f8808f08e03803340/Screenshot_2024-05-08_at_6.20.20_PM.png)

Let's break down each of the four plots provided and evaluate their significance for the blog post:

![Screenshot 2024-05-08 at 6.26.20 PM.png](Part-IV%20COBRE%20Classification%20Model%20with%20GAT%20b70cf5e2f7d5483f8808f08e03803340/Screenshot_2024-05-08_at_6.26.20_PM.png)

### **1. Train vs Test Loss**

This plot shows the model's average training loss decreasing sharply during the initial epochs and then stabilizing, while the average test loss remains relatively constant throughout the training process. The significant gap between the training and test loss suggests that the model might be overfitting to the training data. This indicates good learning on the training set but potentially poor generalization to unseen data, as evidenced by the relatively flat test loss curve.

### **2. Train vs Test Accuracy**

In this graph, the average training accuracy increases rapidly and reaches a high level, which stabilizes close to 95%. In contrast, the test accuracy starts lower and only modestly improves, stabilizing around 70%. The disparity between training and test accuracy further supports the notion of overfitting, where the model performs well on the training data but less so on new, unseen data.

### **3. Train vs Test F1 Score**

The F1 Score plot shows a similar trend to the accuracy plot. The training F1 Score quickly reaches and maintains high performance, indicating strong precision and recall on the training set. However, the test F1 Score is much lower throughout the training process, which might indicate that while the model is capable of identifying training cases well, it struggles to maintain both precision and recall on the test set.

### **4. Final ROC AUC for Each Run**

This bar chart shows the ROC AUC values for multiple runs of the model, demonstrating variability between 0.7 and almost 0.95. The variation across runs might suggest differing initializations or data shuffling that significantly impact the model's ability to generalize. Generally, higher AUC values across different runs show that the model has a good discriminatory ability, capable of distinguishing between classes across varied iterations.

### **Conclusion: Insights and Next Steps**

Through our detailed exploration of the BrainGAT model's performance, we've uncovered several crucial insights about its capabilities and limitations. The provided plots offer a visual testament to how the model learns and behaves across different stages of training and evaluation.

1. **Overfitting Concerns:**
   The observed disparity between training and test metrics, especially in loss and accuracy, indicates a classic case of overfitting. While the model excels with familiar training data, its performance on unseen test data suggests it struggles to generalize well. This is a common challenge in machine learning, particularly in complex models like those dealing with high-dimensional brain connectivity data.
2. **Performance Variability:**
   The variability in the ROC AUC scores from run to run also highlights the sensitivity of the model to initial conditions and dataset partitions. This suggests that further stability in the model's training process could be achieved through more consistent or advanced initialization techniques, possibly enhancing overall performance consistency.
3. **Model Adjustments:**
   To address these issues, several strategies could be implemented: - **Regularization Techniques:** Introducing dropout or L2 regularization might help reduce overfitting by penalizing overly complex models. - **Data Augmentation:** Increasing the diversity of the training data through augmentation techniques could help improve the model's ability to generalize. - **Hyperparameter Tuning:** Adjusting learning rates, the number of layers, or other architectural features could help find a better balance between training and test performance. - **Cross-Validation:** Implementing k-fold cross-validation might provide a more robust evaluation of the model's performance and its generalizability.
4. **Further Research:**
   Continued research into different architectural enhancements or the inclusion of additional data types might also yield improvements. Exploring alternative models or ensemble methods could provide comparative insights and potentially better performance.

In conclusion, while the current model demonstrates strong potential, the journey to refine and enhance its performance is ongoing. The insights gained from this analysis not only guide our next steps but also contribute to the broader discourse on applying graph neural networks in healthcare diagnostics. We encourage the community to engage with the findings, experiment with our shared codebase, and participate in the evolution of this promising field.

### **Where to Find the Code**

For those interested in the specifics of how these plots are generated or wish to dive deeper into the coding aspect, I've updated the notebook with all the necessary scripts. You can find this notebook in the repository. This updated notebook includes all the modifications and enhancements made to the model evaluation process, ensuring you have access to the most current and effective methods used in our analysis.

### **Next Steps**

After reviewing these plots, you might want to experiment with the model further. Whether it's adjusting hyperparameters, experimenting with different model architectures, or applying the model to different datasets, the provided notebook will serve as a valuable starting point for your explorations.

By focusing on visual outcomes and making the code readily accessible, I aim to make this analysis as user-friendly and informative as possible. Whether you're a seasoned data scientist or a curious enthusiast, these resources are designed to help you gain a deeper understanding of graph neural networks and their application in real-world scenarios.
