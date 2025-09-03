# HyperTMO: A trusted multi-omics integration framework based on Hypergraph convolutional network for patient classification
![fig1](./data/fig1.png)
**General conceptual framework of HyperTMO.**
(A) The omics data is used to calculate the cosine similarity for each sample, and then the samples are clustered based on KNN algorithm to get the incidence matrix and form the corresponding hypergraph structure.
(B) A hypergraph convolutional neural network (HGCN) is constructed for each omics data type to perform specific learning and output as evidence for multi-omics integration.
(C)  Dirichlet distribution was parameterized based on the above evidence results, and then the confidence and uncertainty parameters were calculated for each omics data type. Finally, cross-omics information integration was performed based on the D-S evidence theory, and output the classification results with trusted multi-omics integration.

## Dependencies
- Compatible with PyTorch 1.8.0 and Python 3.x.
- For data (and/or splits) not used in the paper, please consider tuning hyperparameters such as hidden size, learning rate, seed, etc. on validation data.
## Run
    python train.py -fd BRCA -nc 5
- `-fd` The dataset file folder.
- `-nc` Number of classes.
- `-kn` Number of vertices in hyperedge.
## Citation
- This article has been accepted by bioinformatics
```
@article{10.1093/bioinformatics/btae159,
    author = {Wang, Haohua and Lin, Kai and Zhang, Qiang and Shi, Jinlong and Song, Xinyu and Wu, Jue and Zhao, Chenghui and He, Kunlun},
    title = "{HyperTMO: a trusted multi-omics integration framework based on hypergraph convolutional network for patient classification}",
    journal = {Bioinformatics},
    pages = {btae159},
    year = {2024},
    month = {03},
    abstract = "{The rapid development of high-throughput biomedical technologies can provide researchers with detailed multi-omics data. The multi-omics integrated analysis approach based on machine learning contributes a more comprehensive perspective to human disease research. However, there are still significant challenges in representing single-omics data and integrating multi-omics information.This paper presents HyperTMO, a Trusted Multi-Omics integration framework based on Hypergraph convolutional network for patient classification. HyperTMO constructs hypergraph structures to represent the association between samples in single-omics data, then evidence extraction is performed by hypergraph convolutional network, and multi-omics information is integrated at an evidence level. Lastly, we experimentally demonstrate that HyperTMO outperforms other state-of-the-art methods in breast cancer subtype classification and Alzheimer’s disease classification tasks using multi-omics data from TCGA (BRCA) and ROSMAP datasets. Importantly, HyperTMO is the first attempt to integrate hypergraph structure, evidence theory, and multi-omics integration for patient classification. Its accurate and robust properties bring great potential for applications in clinical diagnosis.HyperTMO and datasets are publicly available at https://github.com/ippousyuga/HyperTMO.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btae159},
    url = {https://doi.org/10.1093/bioinformatics/btae159},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btae159/57094594/btae159.pdf},
}
```
