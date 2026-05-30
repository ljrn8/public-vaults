[[GNNs]] [[xAI]] [[gnn-xai-exec-summary]]


for [[gnn-xai-exec-summary]]:
- bagel benchmakr (unpublished) and graphXAI benchmark
- all explainers exhibit fragility (low satbility), evaluation gaps, dataset errors(trivial or ambigious ground truths), scalabilty issues.

- Promising directions include learning _consistent_ or _ensemble_ explanations

**perterbation based**
- pge, gnne, subgraphX,
- also Zorro, rate distortion, reportidly more valid, stapse and stable than other explainers
- graphMask

**surrogate based**
- PGM-explainer: bayesian network, grow shrink
- GraphLIME: N hop local NONLINEAR
- GraphSVX : local linear regresson
- recent methods:
	- distill n'explain (DnX) - distilliation, simpler surroage GNN vai knowledge distillation, then runs a convex optimsation to extracte importance. **runs orders faster and outperforms state of the art (read**)**. also FastDnX
	- Local Inteperable Surrogates (ILS)

**Benchmark studies**
explaining the explainers paper:
- **perturbation-based explainers achieved the highest faithfulness/fidelity on graph classification tasks**,
- gradient is better for feature masks
- PGExplainer yields the most stable explanations
- all methods struggled with large complex explanation subgraph, and 56% more faithful on homophilic graphs than heterophilic 

**BAGEL observations**
- purbatoin yield concise suibgraphs and high fidelity at the cost of computational time and fragility.surrogates DnX and ILS outperform this fidelity and are faster, though are often feature based,

**Gaps/limitations**
- extreme fragility for all popular explainers (zhoa et al 2024), to tiny graph perturbatoins. found also with GES.
	- include adversial test in benchmarks
	- and variance between muiltiple models
	- task variety (link prediction, graphs, node)
	- mixture fof ysnthetic and real world datasets
- bad scalability to large graphs
- copmaring explanations to human readability
- reproducability due to explanation configuration and task specification/topology
- 