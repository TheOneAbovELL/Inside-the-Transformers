import pytest
import torch

from transformer_lab.config import ModelConfig
from transformer_lab.models import Transformer


def make_model(**overrides) -> Transformer:
    kw = dict(vocab_size=23, n_ctx=16, d_model=32, n_layers=2, n_heads=4, d_mlp=64, seed=0)
    kw.update(overrides)
    return Transformer(ModelConfig(**kw))


@pytest.fixture
def model() -> Transformer:
    torch.manual_seed(0)
    return make_model()


@pytest.fixture
def tokens() -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 23, (3, 12), generator=g)


@pytest.fixture(autouse=True)
def _isolated_artifacts(tmp_path, monkeypatch):
    """Tests never write into the real artifacts directory."""
    monkeypatch.setenv("ITL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
