# Long Context Question Answering via Supervised Contrastive Learning

This repository contains the accompanying code for the paper:

**"Long Context Question Answering via Supervised Contrastive Learning
."** Avi Caciularu, Arman Cohan, Ido Dagan, Jacob Goldberger  and Arman Cohan. *In NAACL, 2022*.
[[PDF]](https://arxiv.org/pdf/2112.08777.pdf)

Clone one of the tasks' repositories that we used, and edit the model file using the instructions below.
* For multi-hop question answering (*HotpotQA*), use [this link](https://github.com/armancohan/longformer/tree/hotpotqa).
  * The model file appears [here](https://github.com/armancohan/longformer/tree/hotpotqa).
* For long-document question answering (*QAsper*), use [this link](https://github.com/allenai/qasper-led-baseline).
  * The model file appears [here](https://github.com/armancohan/longformer/blob/hotpotqa/examples/hotpotqa.py).

After adding the arguments that appear in `const_loss_compute.py` to your code, insert the following code to the model `__init__` function:

```python
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
if hasattr(self.args, 'add_contrastive') and args.add_contrastive:
    if hasattr(self.args, 'add_dropout') and args.add_dropout:
        self.dropout_layer = nn.Dropout(p=0.1)
    else:
        self.dropout_layer = nn.Dropout(p=0.0)
    self.qa_type_matrix_q = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, projection_dim, bias=False), self.dropout_layer) for _ in range(n_qa_types)])
    self.qa_type_matrix_a = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, projection_dim, bias=False), self.dropout_layer) for _ in range(n_qa_types)])
    self.qa_type_matrix_q.apply(init_weights)
    self.qa_type_matrix_a.apply(init_weights)
```

where you should set the `n_qa_types` according to each benchmark, and `projection_dim` according to the appendix in our paper.

Then, add the code from `const_loss_compute.py` to the `forward` function, after the computing the activations (encoded input). Make sure to add (+) `cont_loss` to the final loss, multiplied by the proper coefficient.

---
## Citation:
If you find our work useful, please cite the paper as:

```bibtex
@article{caciularu2022Long,
  title={Long Context Question Answering via Supervised Contrastive Learning},
  author={Caciularu, Avi and Dagan, Ido and Goldberger, Jacob and Cohan, Arman},
  journal={The North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2022)},
  year={2021}
}
```