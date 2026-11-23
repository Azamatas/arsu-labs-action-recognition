"""Model configuration. Every size of the model is a field here; nothing is set by changing module globals."""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64           # filters per convolution branch
    kernel: int = 61          # convolution length (samples)
    emb: int = 64             # embedding size per axis (the FIN's dense layer)
    head: int = 128           # hidden units of the classification head
    circular: bool = False    # wrap-around convolutions: the FIN is then exactly ReLU(L |FFT|^2 + c)
    features: str = "acc40"   # statistical feature set (bifin.features.SETS)
    pooling: str = "concat"   # axis embeddings into the head: concat | mean | meanmax (per sensor)
    dropout: float = 0.5      # after the hidden layer of the head
    pretrained: bool = True   # pretrain the FIN to imitate the targets and freeze it; else train it with the head

    def name(self):
        """Short id used in result paths, e.g. w32-k31-e64-h64-circ-acc7."""
        s = f"w{self.width}-k{self.kernel}-e{self.emb}-h{self.head}" + ("-circ" if self.circular else "")
        s += f"-{self.features}"
        if self.pooling != "concat":
            s += f"-{self.pooling}"
        if not self.pretrained:
            s += "-random"
        return s

    def as_dict(self):
        return asdict(self)


BIFIN = ModelConfig()  # the paper's model: 64 filters, kernel 61 (10 s windows), head 128, 40 features
BITFIN = ModelConfig(width=32, kernel=31, emb=64, head=64, circular=True)  # smallest; runs as FFT + one matrix
PRESETS = {"bifin": BIFIN, "bitfin": BITFIN}


def preset(name, **changes):
    return replace(PRESETS[name], **{k: v for k, v in changes.items() if v is not None})
