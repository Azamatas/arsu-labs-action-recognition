# IF-ConvTransformer Reproduction Results

## Experimental Setup

The official implementation of IF-ConvTransformer was reproduced using the original source code and datasets provided by the authors. Experiments were conducted on the **HAPT** and **Opportunity** datasets using the default training configuration.

## Results

| Dataset | Paper Macro F1 (%) | Reproduced Macro F1 (%) | Difference |
|---------|-------------------:|------------------------:|-----------:|
| HAPT | 85.58 | **86.76 ± 8.90** | +1.18 |
| Opportunity | 47.41 | **46.35 ± 5.93** | -1.06 |

The reproduced weighted/overall accuracy values are:

| Dataset | Reproduced Accuracy (%) |
|---------|------------------------:|
| HAPT | **95.25 ± 4.02** |
| Opportunity | **88.16 ± 1.38** |

## Discussion

The reproduced implementation achieved performance very close to the results reported in the original paper.

- On the **HAPT** dataset, the reproduced model obtained a Macro F1 score of **86.76%**, exceeding the published result (**85.58%**) by **1.18 percentage points**.
- On the **Opportunity** dataset, the reproduced Macro F1 score was **46.35%**, which is only **1.06 percentage points** lower than the reported **47.41%**.

Such small differences are expected in deep learning experiments due to factors including random weight initialization, dataset preprocessing, hardware differences, CUDA/cuDNN implementations, and software library versions.

## Conclusion

The reproduction can be considered **successful**. The obtained performance closely matches the original publication on both datasets, with deviations of approximately **±1%**, demonstrating that the reproduced implementation behaves consistently with the results reported by the authors.