"""Configured data sources for the Quasar pretraining run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataSource:
    name: str
    repo_id: str
    url: str
    category: str
    stage: str
    split: str = "train"
    subsets: tuple[str, ...] = ()
    estimated_tokens: int | None = None
    license: str | None = None
    priority: int = 100
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "url": self.url,
            "category": self.category,
            "stage": self.stage,
            "split": self.split,
            "subsets": list(self.subsets),
            "estimated_tokens": self.estimated_tokens,
            "license": self.license,
            "priority": int(self.priority),
            "notes": self.notes,
        }


class DataSourceRegistry:
    def __init__(self, sources: list[DataSource]) -> None:
        self._sources = {source.name: source for source in sources}

    def __iter__(self):
        return iter(self._sources.values())

    def get(self, name: str) -> DataSource:
        return self._sources[name]

    def names(self, *, stage: str | None = None, category: str | None = None) -> list[str]:
        out: list[str] = []
        for source in self._sources.values():
            if stage is not None and source.stage != stage:
                continue
            if category is not None and source.category != category:
                continue
            out.append(source.name)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"sources": [source.to_dict() for source in self._sources.values()]}


DATA_SOURCES: tuple[DataSource, ...] = (
    DataSource(
        name="automathtext_v2",
        repo_id="OpenSQZ/AutoMathText-V2",
        url="https://huggingface.co/datasets/OpenSQZ/AutoMathText-V2",
        category="math",
        stage="pretrain",
        estimated_tokens=2_460_000_000_000,
        priority=10,
        notes="Math-heavy source; user-provided target size 2.46T tokens.",
    ),
    DataSource(
        name="ultra_fineweb_l3",
        repo_id="openbmb/Ultra-FineWeb-L3",
        url="https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3",
        category="synthetic_web",
        stage="pretrain",
        subsets=(
            "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
            "Ultra-FineWeb-L3-en-QA-Synthetic",
            "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
            "Ultra-FineWeb-L3-zh-QA-Synthetic",
        ),
        license="apache-2.0",
        priority=20,
    ),
    DataSource(
        name="ultra_fineweb_en",
        repo_id="openbmb/Ultra-FineWeb",
        url="https://huggingface.co/datasets/openbmb/Ultra-FineWeb",
        category="web",
        stage="pretrain",
        split="en",
        license="apache-2.0",
        priority=30,
    ),
    DataSource(
        name="ultra_fineweb_zh",
        repo_id="openbmb/Ultra-FineWeb",
        url="https://huggingface.co/datasets/openbmb/Ultra-FineWeb",
        category="web",
        stage="pretrain",
        split="zh",
        license="apache-2.0",
        priority=31,
    ),
    DataSource(
        name="ultradata_math",
        repo_id="openbmb/UltraData-Math",
        url="https://huggingface.co/datasets/openbmb/UltraData-Math",
        category="math",
        stage="pretrain",
        subsets=(
            "UltraData-Math-L3-Conversation-Synthetic",
            "UltraData-Math-L3-Multi-Style-Synthetic",
            "UltraData-Math-L3-QA-Synthetic",
            "UltraData-Math-L3-Textbook-Exercise-Synthetic",
            "UltraData-Math-L2-preview",
            "UltraData-Math-L1",
        ),
        license="apache-2.0",
        priority=15,
    ),
    DataSource(
        name="fineweb",
        repo_id="HuggingFaceFW/fineweb",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb",
        category="general_knowledge",
        stage="pretrain",
        license="odc-by",
        priority=40,
    ),
    DataSource(
        name="nemotron_specialized_v1_2",
        repo_id="nvidia/Nemotron-Pretraining-Specialized-v1.2",
        url="https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.2",
        category="specialized",
        stage="midtrain",
        subsets=(
            "Nemotron-Pretraining-Fact-Seeking",
            "Nemotron-Pretraining-Generative",
            "Nemotron-Pretraining-Moral-Scenarios",
            "Nemotron-Pretraining-Multiple-Choice",
        ),
        estimated_tokens=51_000_000_000,
        priority=50,
        notes="Mid-training source; user-provided target size 51B tokens.",
    ),
    DataSource(
        name="nemotron_specialized_v1_1",
        repo_id="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        url="https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.1",
        category="specialized",
        stage="midtrain",
        license="cc-by-4.0",
        priority=60,
    ),
)


def default_registry() -> DataSourceRegistry:
    return DataSourceRegistry(list(DATA_SOURCES))
