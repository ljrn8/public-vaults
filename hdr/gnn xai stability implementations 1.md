
> explaining wrong predictions
## 1. MixupExplainer

### Authors / Venue / Citation
Jiaxing Zhang, Dongsheng Luo, Hua Wei. **KDD 2023** (29th ACM SIGKDD, Long Beach CA). arXiv:2307.07832. DOI: 10.1145/3580305.3599435.

### Method Summary
MixupExplainer is a wrapper built on top of GIB-based explainers (GNNExplainer, PGExplainer). The GIB (Graph Information Bottleneck) framework — an information-theoretic approach that finds an explanation subgraph G\* maximising label mutual information while minimising size — uses f(G\*) for prediction. Since f is trained on full graphs, evaluating it on the small G\* is OOD.

MixupExplainer replaces the prediction target: it **mixes** the explanation subgraph G\* with a label-independent (non-explanatory) subgraph sampled from another random graph, producing a mixed graph G^(mix). The GNN predicts on G^(mix) instead of G\* alone, which is closer to the training distribution because it has similar size and topology to full graphs. The objective becomes: minimise I(G, G\*) subject to maximising label agreement between Y and Y^(mix) = f(G^(mix)).

### Drawbacks
- Assumes the explanatory and non-explanatory parts of graphs are **independent** — does not hold in many real-world graphs where these parts interact structurally.
- Mixing uses a randomly sampled graph, so the quality of augmentation is noisy and uncontrolled.
- No explicit generative model for the non-explanation portion; distribution matching is approximate.
- Only a wrapper over GIB-based explainers, not a standalone method.

### Experimental Settings
- **Datasets:** BA-Shapes, BA-Community (node classification); Tree-Circles, Tree-Grid (node classification); BA-2motifs, MUTAG (graph classification)
- **Model:** 3-layer GCN, Adam optimizer, weight decay 5e-4, 80/20 train/test split
- **Baseline explainers:** GRAD, ATT, SubgraphX, MetaGNN, RG-Explainer — results copied from original papers; GNNExplainer and PGExplainer — re-implemented with same configuration
- **Metric:** AUC-ROC on edges (primary); cosine similarity and Euclidean distance used for distribution shift visualisation

### Results: AUC-ROC (Table 1)

| Method           | BA-Shapes | BA-Community | Tree-Circles | Tree-Grid | BA-2motifs | MUTAG     |
| ---------------- | --------- | ------------ | ------------ | --------- | ---------- | --------- |
| GRAD             | 0.882     | 0.750        | 0.905        | 0.612     | 0.717      | 0.783     |
| ATT              | 0.815     | 0.739        | 0.824        | 0.667     | 0.667      | 0.765     |
| SubgraphX        | 0.548     | 0.473        | 0.617        | 0.516     | 0.610      | 0.529     |
| MetaGNN          | 0.851     | 0.688        | 0.523        | 0.628     | 0.500      | 0.680     |
| RG-Explainer     | 0.985     | 0.919        | 0.787        | 0.927     | 0.657      | 0.873     |
| GNNExplainer     | 0.884     | 0.682        | 0.683        | 0.379     | 0.660      | 0.539     |
| **MixUp-GNNExp** | **0.890** | **0.788**    | **0.690**    | **0.501** | **0.869**  | **0.612** |
| PGExplainer      | 0.999     | 0.829        | 0.762        | 0.679     | 0.679      | 0.843     |
| **MixUp-PGExp**  | **0.999** | **0.955**    | **0.774**    | **0.712** | **0.920**  | **0.871** |

MixUp-GNNExp improves over GNNExplainer by 12.3% on average (node classification) and 22.6% (graph classification). MixUp-PGExplainer improves over PGExplainer by 5.41% and 19.4% respectively.

---

## 2. EdgeDropExplainer

### Authors / Venue / Citation
Zhuomin Chen, Hojat Allah Salehi, Esteban Schafir, Xu Zheng, Jiaxing Zhang, Hua Wei, Jingchao Ni, Farhad Shirani, Dongsheng Luo. **IEEE TPAMI 2026** (accepted, DOI: 10.1109/TPAMI.2026.3690304). This is the extended journal version of the ProxyExplainer ICML 2024 paper, in which EdgeDropExplainer is introduced as a simpler variant.

### Method Summary
EdgeDropExplainer addresses the OOD problem with a simple non-parametric approach. When evaluating explanation subgraph G_exp, rather than using G_exp alone (too small, OOD) or generating a learned proxy, it produces an augmented graph by taking G_exp and **randomly dropping edges** from the complement G \ G_exp using Bernoulli(1-p), where p is a drop ratio hyperparameter. The retained edges form a perturbed complement G^Δ, and the proxy is G̃ = G_exp ∪ G^Δ. This creates an augmented graph that is smaller than the full graph and structurally variable, bringing evaluation closer to the training distribution without any generative model.

It uses the same GIB-based explainer objective as PGExplainer but with this stochastic perturbation replacing direct complement evaluation.

### Drawbacks
- Edge dropping is not learned — the distribution of G^Δ is not explicitly matched to training data; it may drop structurally important edges or preserve irrelevant ones.
- No guarantee the resulting proxy is actually in-distribution.
- Consistently underperforms ProxyExplainer; best treated as an ablation of the full ProxyExplainer framework.
- Still requires the same GIB-style training overhead as PGExplainer.

### Experimental Settings
Same as ProxyExplainer (see Section 3). Results reported alongside ProxyExplainer in the TPAMI 2026 paper.

### Results: AUC-ROC (Table III, TPAMI 2026) and Fidelity (Table IV, TPAMI 2026)

**AUC-ROC:**

| Method                | MUTAG     | Benzene   | Alkane-Carbonyl | Fluoride-Carbonyl | BA-2motifs | BA-3motifs |
| --------------------- | --------- | --------- | --------------- | ----------------- | ---------- | ---------- |
| GradCAM               | 0.727     | 0.740     | 0.448           | 0.694             | 0.714      | 0.709      |
| GNNExplainer          | 0.682     | 0.485     | 0.551           | 0.574             | 0.644      | 0.511      |
| PGExplainer           | 0.832     | 0.793     | 0.660           | 0.702             | 0.734      | 0.796      |
| ReFine                | 0.612     | 0.606     | 0.768           | 0.571             | 0.698      | 0.629      |
| MixupExplainer        | 0.863     | 0.611     | 0.811           | 0.706             | 0.906      | 0.859      |
| **EdgeDropExplainer** | 0.864     | 0.842     | **0.946**       | 0.609             | 0.672      | 0.623      |
| **ProxyExplainer**    | **0.977** | **0.845** | 0.934           | **0.758**         | **0.935**  | **0.960**  |

**Fidelity** (Fid_{α₁,+}↑ / Fid_{α₂,-}↓, α₁=0.1, α₂=0.9, on MUTAG and Benzene):

| Method             | MUTAG Fid+ | MUTAG Fid- | Benzene Fid+ | Benzene Fid- |
| ------------------ | ---------- | ---------- | ------------ | ------------ |
| GradCAM            | 0.004      | 0.162      | 0.103        | 0.066        |
| GNNExplainer       | 0.031      | 0.148      | 0.060        | 0.113        |
| PGExplainer        | 0.034      | 0.148      | 0.128        | 0.051        |
| ReFine             | 0.003      | 0.160      | 0.077        | 0.088        |
| MixupExplainer     | 0.037      | 0.146      | 0.085        | 0.078        |
| EdgeDropExplainer  | 0.034      | 0.146      | 0.131        | 0.047        |
| **ProxyExplainer** | **0.040**  | **0.145**  | **0.131**    | **0.046**    |

---

## 3. ProxyExplainer

### Authors / Venue / Citation
Zhuomin Chen, Jiaxing Zhang, Jingchao Ni, Xiaoting Li, Yuchen Bian, Md Mezbahul Islam, Ananda Mohan Mondal, Hua Wei, Dongsheng Luo. **ICML 2024** (41st International Conference on Machine Learning, PMLR 235). arXiv:2402.02036. Extended to TPAMI 2026 (DOI: 10.1109/TPAMI.2026.3690304).

### Method Summary
ProxyExplainer generates **in-distribution proxy graphs** for explanation evaluation, addressing the core OOD problem more rigorously than MixupExplainer or EdgeDropExplainer.

When evaluating explanation subgraph G_exp, a proxy graph G̃ = G_exp + perturbation(G \ G_exp) is constructed. The perturbation is produced by a **Variational Graph Autoencoder (VGAE)**: the non-explanation subgraph G^Δ = G \ G_exp is encoded into a Gaussian latent space (mean μ^Δ, variance σ^Δ), a latent vector Z^Δ is sampled, and a decoder reconstructs an adjacency matrix Ã^Δ. The final proxy adjacency is Ã = Ã_exp + Ã^Δ. The training objective L_proxy = L_dist + λL_KL jointly optimises explanation quality (L_dist: cross-entropy between G̃ and G predictions) and distribution adherence (L_KL: KL divergence from Gaussian prior). The explainer and VGAE are trained alternately: M steps on the VGAE, then 1 step on the explainer.

### Drawbacks
- VGAE adds O(|E|+n²) time and O(n²) space complexity, significantly more than PGExplainer's O(|E|).
- Alternating training is harder to tune and more sensitive to hyperparameters (M, λ).
- Assumes the non-explanation subgraph has a unimodal Gaussian latent structure — may fail on highly irregular graphs.
- Assumes independence between explanation and non-explanation subgraphs during generation.
- Performance on Fluoride-Carbonyl (0.758) is notably below EdgeDropExplainer on Alkane-Carbonyl (0.946), suggesting dataset-specific weaknesses.

### Experimental Settings
- **Datasets (ICML):** MUTAG, Benzene, Alkane-Carbonyl, Fluoride-Carbonyl (real-world molecular); BA-2motifs, BA-3motifs (synthetic)
- **Datasets (TPAMI extension):** Above + D&D, OGBG-MolHIV, IMDB-BINARY, REDDIT-BINARY
- **Model:** 3-layer GCN (primary); GIN tested in appendix
- **Baselines:** GradCAM, GNNExplainer, PGExplainer, ReFine, MixupExplainer — all re-implemented with same settings and architecture
- **Metrics:** AUC-ROC (primary), Average Precision (AP), Fidelity (Fid_{α₁,+}↑ / Fid_{α₂,-}↓)

Results tables are shared with EdgeDropExplainer above (Tables III and IV from TPAMI 2026).

---

## 4. HINT-G

### Authors / Venue / Citation
Heesoo Jung, Chanyong Kim, Geonhee Han, Hogun Park. **KDD 2025** (31st ACM SIGKDD, August 3–7, 2025, Toronto, Canada). DOI: 10.1145/3711896.3736995. Code: https://github.com/cycy-kim/HINT-G.

### Method Summary
HINT-G (Harnessing INfluence function for Task-irrelevant explanation on Graph neural networks) uses **influence functions** — a classical technique that estimates how much the loss would change if a training data point (or perturbation) were up-weighted, by computing the product of the gradient and the inverse Hessian of the model parameters.

HINT-G adapts this to edges: the influence of an edge (i,j) on the GNN's loss is estimated as ∂θ̂*(ε,δ)/∂ε|_{ε=0}, where ε is the perturbation strength and θ̂* are the model parameters. The inverse Hessian is approximated with LiSSA or Arnoldi iteration, reducing exact cubic cost. Because the influence score only requires a differentiable loss (not class labels), HINT-G is **task-irrelevant**: it works for supervised and unsupervised GNNs alike.

A key novelty is identifying **negative edges** — edges not in the original graph but whose addition would change the prediction. This expands the explanation beyond the given graph structure. Two variants: **HINT-G_Edge** computes edge influence directly (requires O(|E|) Hessian passes); **HINT-G_Node** approximates edge influence from node influences (O(|V|) passes), substantially reducing computation.

### Drawbacks
- Even with approximation, computing the inverse Hessian for HINT-G_Edge is expensive for large graphs (|Ẽ| >> |E| for negative edge search).
- HINT-G_Node's approximation of edge influence via node influence loses some precision — visible in Table 1 where HINT-G_Node underperforms HINT-G_Edge on several datasets.
- Does not scale to very large graphs; MovieLens-1M is tested but positive edge detection is not possible (no positive edges by construction).
- Post-hoc and instance-specific; does not generalise across graphs at inference time like PGExplainer.

### Experimental Settings
- **Datasets:** Tree-Cycles, Tree-Grid (node classification); BA-2motifs (graph classification); MovieLens-1M (link prediction/recommendation, used for unsupervised evaluation)
- **Model:** 3-layer GCN (supervised); unsupervised GCN for MovieLens
- **Baselines:** Grad, GNNExplainer, PGExplainer, MixupExplainer, ProxyExplainer, TAGE — all re-implemented with the same GCN architecture for fair comparison
- **Metric:** AUC-ROC, reported as harmonic mean of positive edge detection AUC and negative edge detection AUC
- Note: supervised baselines (GNNExplainer, PGExplainer, MixupExplainer, ProxyExplainer) are not applicable in the unsupervised setting

### Results: AUC-ROC (Tables 1 & 2)

**Table 1 — Harmonic mean of positive and negative edge AUC-ROC:**

| Method          | Tree-Cycles (Sup) | Tree-Grid (Sup) | BA-2motifs (Sup) | Tree-Cycles (Unsup) | Tree-Grid (Unsup) | MovieLens-1M (Unsup) |
| --------------- | ----------------- | --------------- | ---------------- | ------------------- | ----------------- | -------------------- |
| Grad            | 0.613             | 0.585           | 0.772            | 0.731               | 0.489             | 0.500                |
| GNNExplainer    | 0.568             | 0.513           | 0.562            | —                   | —                 | —                    |
| PGExplainer     | 0.639             | 0.606           | 0.595            | —                   | —                 | —                    |
| MixupExplainer  | 0.641             | 0.606           | 0.644            | —                   | —                 | —                    |
| ProxyExplainer  | 0.637             | 0.620           | 0.652            | —                   | —                 | —                    |
| TAGE            | 0.594             | 0.595           | 0.566            | 0.840               | 0.579             | 0.500                |
| **HINT-G_Edge** | **0.927**         | **0.819**       | **0.913**        | **0.751**           | **0.784**         | **0.865**            |
| **HINT-G_Node** | 0.801             | **0.880**       | 0.881            | **0.826**           | **0.800**         | 0.802                |

Improvement over best baseline: +44.6% (Tree-Cycles, sup), +41.9% (Tree-Grid, sup), +18.3% (BA-2motifs, sup); +13.0% (Tree-Cycles, unsup), +38.5% (Tree-Grid, unsup), +73.0% (MovieLens, unsup).

---

## 5. OAR / OAR+ (ISSUE: not ORExplainer, this is OR Resistant Evals)

### Authors / Venue / Citation
Junfeng Fang, Hao Wu, An Zhang, Tianlong Chen, Kun Wang, Yuxuan Liang, Roger Zimmermann, Xiang Wang. **IEEE TPAMI Special Issue on Graphs in Vision and Pattern Analysis, 2026** (DOI: 10.1109/TPAMI.2026.3664091). Code: https://github.com/MangoKiller/OAR.

### Method Summary
OAR (OOD-resistant Adversarial Robustness) is not an explainer — it is an **evaluation framework** for assessing the quality of GNN explanation methods while being resistant to the OOD problems that afflict standard evaluation metrics (Fidelity, Recall, Precision).

Standard metrics evaluate explanations by removing subgraph features, which creates OOD inputs that cause unreliable GNN predictions and biased scores. OAR reframes evaluation as adversarial robustness: it applies adversarial perturbations to the graph (randomly removing edges from the complement or fixing the explanation subgraph) and measures how robust the explanation is to these perturbations. A robustness score δ̂ is computed by measuring prediction change under perturbation. To correct for OOD bias in the perturbed subgraph, OAR includes an **OOD reweighting mechanism** using a VGAE: subgraphs with lower VGAE reconstruction loss (i.e., more in-distribution) receive higher weight in the final score.

**OAR+** extends this with (1) a **counterfactual attack module** that generates adversarial perturbations targeting the specific explanation rather than random deletion, and (2) a **conditional graph diffusion model** that reconstructs perturbed subgraphs toward the training distribution before evaluation. A standardised benchmarking framework is also proposed to enable cross-study comparison.

### Drawbacks
- OAR is an evaluation metric, not a method that improves explanations themselves.
- Requires training a separate VGAE per dataset for OOD reweighting — computationally expensive.
- The diffusion model in OAR+ adds further training overhead.
- The adversarial perturbation framework may not capture all failure modes of explanation methods.
- VGAE reweighting may fail on unusual graph distributions where the VGAE underfits.
- The standardised framework is currently limited to the tested explainer and dataset combinations.

### Experimental Settings
- **Datasets:** BA3 (BA-3motifs), TR3 (Tree-3), MNIST-sp (superpixel MNIST), MUTAG
- **Post-hoc explainers evaluated:** SA (Sensitivity Analysis), GradCAM, GNNExplainer, PGExplainer, CXPlain, ReFine
- **Intrinsic explainers evaluated:** GIN, DIR, IB-subgraph, GSAT, GREA, AIA
- **Evaluation metrics compared:** Recall, RM (removal metric), DSE (distribution-shifted evaluation), OAR, OAR+
- **Key metric:** Kendall's τ — rank correlation between the metric's ordering of explainers and the ground-truth ordering. The primary claim is that OAR/OAR+ achieves τ=1.00 (perfect) while existing metrics do not.

### Results: Ranking correlation τ (Table 1 and Table 2)

The paper reports rankings and τ rather than raw per-explainer AUC-ROC. OAR/OAR+ consistently achieves τ=1.00 across all four datasets for both post-hoc and intrinsic explainers, versus:

| Metric | τ on BA3 | τ on TR3 | τ on MNIST-sp | τ on MUTAG |
|---|---|---|---|---|
| Recall | — | — | 0.33 | — |
| RM | 0.73 | 0.67 | 0.20 | 0.40 |
| DSE | 0.73 | 0.73 | 0.27 | 0.67 |
| **OAR** | **1.00** | **0.93** | **0.93** | **1.00** |
| **OAR+** | **1.00** | **0.93** | **1.00** | **1.00** |

(τ values from Table 1, post-hoc explainers. A dash indicates Recall has no τ reported for that dataset in the same column.)

---

## 6. V-InFoR

### Authors / Venue / Citation
Senzhang Wang, Jun Yin, Chaozhuo Li, Xing Xie, Jianxin Wang. **NeurIPS 2023** (37th Conference on Neural Information Processing Systems). Code: https://anonymous.4open.science/r/V-InFoR-EF88.

### Method Summary
V-InFoR (Variational Inference for Robustness) addresses GNN explanation on **structurally corrupted graphs** — graphs with noisy or adversarially perturbed edges that degrade both raw features and learned representations.

Two-module approach: (1) A **robust graph representation extractor** using a Variational Autoencoder (VAE): rather than using the corrupted raw graph adjacency directly, the graph is encoded into a Gaussian latent distribution (mean, variance), and multiple samples are drawn and averaged. This sampling procedure denoises the representation by marginalising out minor corruptions. (2) A **GIB-based explanation generator** that takes these robust latent representations as input and formulates explanation search as a GIB optimisation problem, without needing rigid structural constraints on explanation size or connectivity. The GIB constraint adaptively captures regularity/irregularity without hard limits. Joint objective: L_joint = L_VAE + L_GIB.
 **setup:**
	- “two GCN encoders infer the mean and std deviation latent distribtuion of the input graph. variational node representations are sampled from the latent graph, and concatenated to form a 4xn edge matrix (mean,std,mean,std) and fed into the adaptive explanation generator.  this is an MLP wich predicts edge probabilities.
	- GAE = 3 layer GCN as the encoder, inner product as the decoder
	- uses gilbert random graph sampling (uses BCE likge PGExplainer)
	- 

### Drawbacks
- VAE component adds training complexity; the latent Gaussian assumption may not hold for highly heterogeneous graphs.
- On clean graphs (noise ratio = 0), V-InFoR still leads, but margins are smaller — the method's advantage is specifically in the corrupted setting.
- Not tested on very large-scale graphs.
- Evaluates with P_S / P_N / F_NS metrics rather than AUC-ROC, making direct comparison to other papers in this collection non-trivial.
- The adversarial corruption simulation (random noise and adversarial attack) may not fully represent real-world corruption patterns.
**drawbacks**
- **structurally corrupted graphs only**
	- such that minor corruption retains the prediction
	- servere corruption is such that the prediction of the GNN has changed (additiona or deletion)
	-  did not evaulate plausability, only fidelity under varying strucutral changes. no feature  wise changes or label changes. ONly 1 model, shallow dataset variation. only one type of classification probelm used it seems
- also uses the “GRABNEL” attack algorithm
- 

### Experimental Settings
- **Datasets:** BA-3Motifs (synthetic); Mutag, Ogbg-molhiv, Ogbg-ppa (real-world molecular/biological)
- **Corruptions:** 20% random structural noise (random edge flips), and adversarial attacks at varying budgets (0–30% edge perturbation)
- **Model:** GNN trained independently as oracle predictor (architecture not specified beyond multi-layer GNN)
- **Baselines:** GradCAM, IG (Integrated Gradients), GNNExplainer, PGExplainer, PGM-Explainer, ReFine
- **Metrics:**
  - P_S (probability of sufficient explanation): probability that the explanation alone preserves the model's prediction
  - P_N (probability of necessary explanation): probability that removing the explanation changes the prediction
  - F_NS: harmonic mean of P_S and P_N (analogous to F1)
- Baseline results are re-implemented with the same experimental pipeline

### Results: F_NS under 20% random structural noise (Table 1)

| Dataset | GradCAM | IG | GNNExplainer | PGExplainer | PGM-Explainer | ReFine | **V-InFoR** | Improvement |
|---|---|---|---|---|---|---|---|---|
| BA-3Motifs | 0.4012 | 0.4222 | 0.3758 | 0.3362 | 0.3540 | 0.3989 | **0.4610** | +9.19% |
| Mutag | 0.1789 | 0.1907 | 0.1080 | 0.2199 | 0.1830 | 0.2207 | **0.2852** | +29.23% |
| Ogbg-molhiv | 0.1267 | 0.0767 | 0.1701 | 0.2198 | 0.1765 | 0.1834 | **0.2542** | +15.65% |
| Ogbg-ppa | 0.4522 | 0.5139 | 0.4561 | 0.3922 | 0.4694 | 0.5200 | **0.5680** | +9.23% |

V-InFoR ranks 1st on all four datasets under noise. Improvement is defined as ([V-InFoR] − [Best Baseline]) / [Best Baseline].

---

## Cross-Paper Notes

- **AUC-ROC is not directly comparable across papers** because different papers use different datasets, GNN architectures, and training configurations.
- **MixupExplainer, EdgeDropExplainer, and ProxyExplainer** all address the same OOD problem in explanation evaluation but with increasing sophistication (random mixing → random dropping → learned VGAE proxy). ProxyExplainer supersedes the others in the TPAMI 2026 paper.
- **HINT-G** is the only method that considers negative edges (edges outside the original graph), making it qualitatively different from all others.
- **OAR/OAR+** is an evaluation framework, not a competing explainer — it is used to *assess* methods like the above.
- **V-InFoR** addresses a different threat model (graph corruption/noise) rather than the OOD evaluation problem addressed by the others.
