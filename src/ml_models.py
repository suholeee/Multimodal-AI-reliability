"""Small interpretable PyTorch models for polymer-based multimodal classification."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from image_feature_layers import normalize_image_batch


IMAGE_MODEL_NAMES = (
    "baseline_cnn",
    "flat_mlp",
    "global_pool_cnn",
    "hybrid",
)

BEST_IMAGE_MODEL_NAMES = (
    "global_pool_cnn",
    "hybrid",
)


def _coerce_shape(input_shape: Sequence[int]) -> Tuple[int, ...]:
    """Convert an input shape into a plain tuple of ints."""
    return tuple(int(dim) for dim in input_shape)


class ConvEncoder2d(nn.Module):
    """A very small CNN encoder for matrix-like modalities."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: Tuple[int, int] = (8, 16),
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        first_channels, second_channels = hidden_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, first_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second_channels * 4 * 4, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a 2D modality into a compact embedding."""
        return self.projection(self.features(inputs))


class FlattenEncoder(nn.Module):
    """A compact MLP encoder for low-resolution smooth images."""

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_dims: Tuple[int, int] = (128, 48),
        embedding_dim: int = 32,
        input_normalization: str = "none",
    ) -> None:
        super().__init__()
        normalized_shape = _coerce_shape(input_shape)
        input_dim = int(torch.tensor(normalized_shape).prod().item())
        first_hidden, second_hidden = hidden_dims
        self.input_normalization = input_normalization
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a smooth image via a flattened global-intensity pathway."""
        normalized = normalize_image_batch(inputs, mode=self.input_normalization)
        return self.network(normalized)


class GlobalPoolingConvEncoder(nn.Module):
    """A shallow average-pooled CNN tuned for smooth spatial density fields."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: Tuple[int, int, int] = (12, 24, 24),
        embedding_dim: int = 32,
        input_normalization: str = "standardize",
    ) -> None:
        super().__init__()
        first_channels, second_channels, third_channels = hidden_channels
        self.input_normalization = input_normalization
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, first_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(second_channels, third_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(third_channels, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode low-frequency texture with global pooling."""
        normalized = normalize_image_batch(inputs, mode=self.input_normalization)
        return self.projection(self.features(normalized))


class VectorEncoder(nn.Module):
    """A small MLP encoder for vector-like modalities."""

    def __init__(self, input_dim: int, embedding_dim: int = 32) -> None:
        super().__init__()
        hidden_dim = max(16, min(128, input_dim // 4))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a vector modality into a compact embedding."""
        return self.network(inputs)


def build_genomic_encoder(
    input_shape: Sequence[int],
    embedding_dim: int = 32,
) -> nn.Module:
    """Create a genomic encoder that matches the representation shape."""
    normalized_shape = _coerce_shape(input_shape)
    if len(normalized_shape) == 1:
        return VectorEncoder(input_dim=normalized_shape[0], embedding_dim=embedding_dim)
    if len(normalized_shape) == 3:
        return ConvEncoder2d(
            in_channels=normalized_shape[0],
            hidden_channels=(8, 16),
            embedding_dim=embedding_dim,
        )
    raise ValueError(f"Unsupported genomic input shape: {normalized_shape!r}")


class ImageClassifier(nn.Module):
    """The original small CNN image-only classifier."""

    def __init__(self, embedding_dim: int = 32) -> None:
        super().__init__()
        self.encoder = ConvEncoder2d(in_channels=1, hidden_channels=(8, 16), embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return the image embedding used for classification."""
        return self.encoder(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict binary logits from grayscale images."""
        return self.classifier(self.encode(image))


class FlattenedImageClassifier(nn.Module):
    """A small MLP baseline for globally smooth optical images."""

    def __init__(
        self,
        image_shape: Sequence[int],
        embedding_dim: int = 32,
        input_normalization: str = "none",
    ) -> None:
        super().__init__()
        self.encoder = FlattenEncoder(
            input_shape=image_shape,
            hidden_dims=(128, 48),
            embedding_dim=embedding_dim,
            input_normalization=input_normalization,
        )
        self.classifier = nn.Linear(embedding_dim, 2)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return the flattened-image embedding used for classification."""
        return self.encoder(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict binary logits from flattened grayscale images."""
        return self.classifier(self.encode(image))


class GlobalPoolingImageClassifier(nn.Module):
    """A shallow CNN that emphasizes broad density and texture patterns."""

    def __init__(
        self,
        embedding_dim: int = 32,
        input_normalization: str = "standardize",
    ) -> None:
        super().__init__()
        self.encoder = GlobalPoolingConvEncoder(
            in_channels=1,
            hidden_channels=(12, 24, 24),
            embedding_dim=embedding_dim,
            input_normalization=input_normalization,
        )
        self.classifier = nn.Linear(embedding_dim, 2)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return the globally pooled image embedding used for classification."""
        return self.encoder(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict binary logits from shallow CNN image embeddings."""
        return self.classifier(self.encode(image))


class HybridImageEncoder(nn.Module):
    """Encode optical images using both learned features and image summaries."""

    def __init__(
        self,
        image_shape: Sequence[int],
        feature_dim: int,
        image_encoder: str = "global_pool_cnn",
        image_embedding_dim: int = 24,
        feature_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        normalized_shape = _coerce_shape(image_shape)
        if image_encoder == "flat_mlp":
            self.image_encoder = FlattenEncoder(
                input_shape=normalized_shape,
                hidden_dims=(128, 48),
                embedding_dim=image_embedding_dim,
                input_normalization="none",
            )
        elif image_encoder == "global_pool_cnn":
            self.image_encoder = GlobalPoolingConvEncoder(
                in_channels=normalized_shape[0],
                hidden_channels=(12, 24, 24),
                embedding_dim=image_embedding_dim,
                input_normalization="standardize",
            )
        else:
            raise ValueError("image_encoder must be 'flat_mlp' or 'global_pool_cnn'")

        self.feature_encoder = nn.Sequential(
            nn.Linear(int(feature_dim), feature_embedding_dim),
            nn.ReLU(),
        )
        self.output_dim = image_embedding_dim + feature_embedding_dim

    def forward(self, image: torch.Tensor, image_feature: torch.Tensor | None = None) -> torch.Tensor:
        """Combine learned optical embeddings with standardized summary features."""
        if image_feature is None:
            raise ValueError("image_feature is required for the hybrid image encoder")
        image_embedding = self.image_encoder(image)
        feature_embedding = self.feature_encoder(image_feature)
        return torch.cat((image_embedding, feature_embedding), dim=1)


class HybridImageClassifier(nn.Module):
    """Combine a learned optical embedding with summary image statistics."""

    def __init__(
        self,
        image_shape: Sequence[int],
        feature_dim: int,
        image_encoder: str = "global_pool_cnn",
        image_embedding_dim: int = 24,
        feature_embedding_dim: int = 8,
        hidden_dim: int = 24,
    ) -> None:
        super().__init__()
        self.encoder = HybridImageEncoder(
            image_shape=image_shape,
            feature_dim=feature_dim,
            image_encoder=image_encoder,
            image_embedding_dim=image_embedding_dim,
            feature_embedding_dim=feature_embedding_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, image: torch.Tensor, image_feature: torch.Tensor) -> torch.Tensor:
        """Encode learned image structure and hand-crafted image summaries."""
        return self.encoder(image, image_feature)

    def forward(self, image: torch.Tensor, image_feature: torch.Tensor) -> torch.Tensor:
        """Predict binary logits from optical image and summary features."""
        return self.classifier(self.encode(image, image_feature))


class GenomicClassifier(nn.Module):
    """A minimal genomic-only classifier for contact-map inputs."""

    def __init__(self, input_shape: Sequence[int], embedding_dim: int = 32) -> None:
        super().__init__()
        self.encoder = build_genomic_encoder(input_shape=input_shape, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)

    def encode(self, genomic: torch.Tensor) -> torch.Tensor:
        """Return the genomic embedding used for classification."""
        return self.encoder(genomic)

    def forward(self, genomic: torch.Tensor) -> torch.Tensor:
        """Predict binary logits from genomic features."""
        return self.classifier(self.encode(genomic))


class FusionClassifier(nn.Module):
    """A small late-fusion classifier for paired image and genomic inputs."""

    def __init__(
        self,
        genomic_input_shape: Sequence[int],
        image_shape: Sequence[int],
        image_model: str = "hybrid",
        image_feature_dim: int | None = None,
        image_embedding_dim: int = 24,
        genomic_embedding_dim: int = 24,
        fusion_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        normalized_image_shape = _coerce_shape(image_shape)
        self.image_model = image_model
        self.image_requires_feature = image_model == "hybrid"
        if image_model == "baseline_cnn":
            self.image_encoder = ConvEncoder2d(
                in_channels=normalized_image_shape[0],
                hidden_channels=(8, 16),
                embedding_dim=image_embedding_dim,
            )
            image_output_dim = image_embedding_dim
        elif image_model == "flat_mlp":
            self.image_encoder = FlattenEncoder(
                input_shape=normalized_image_shape,
                hidden_dims=(128, 48),
                embedding_dim=image_embedding_dim,
                input_normalization="none",
            )
            image_output_dim = image_embedding_dim
        elif image_model == "global_pool_cnn":
            self.image_encoder = GlobalPoolingConvEncoder(
                in_channels=normalized_image_shape[0],
                hidden_channels=(12, 24, 24),
                embedding_dim=image_embedding_dim,
                input_normalization="standardize",
            )
            image_output_dim = image_embedding_dim
        elif image_model == "hybrid":
            if image_feature_dim is None:
                raise ValueError("image_feature_dim is required when fusion uses the hybrid image model")
            self.image_encoder = HybridImageEncoder(
                image_shape=normalized_image_shape,
                feature_dim=image_feature_dim,
                image_encoder="global_pool_cnn",
                image_embedding_dim=image_embedding_dim,
                feature_embedding_dim=8,
            )
            image_output_dim = self.image_encoder.output_dim
        else:
            raise ValueError(f"Unsupported fusion image model: {image_model!r}")
        self.image_output_dim = int(image_output_dim)
        self.genomic_encoder = build_genomic_encoder(
            input_shape=genomic_input_shape,
            embedding_dim=genomic_embedding_dim,
        )
        self.genomic_output_dim = int(genomic_embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(image_output_dim + genomic_embedding_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 2),
        )

    def encode_modalities(
        self,
        image: torch.Tensor,
        genomic: torch.Tensor,
        image_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each modality independently before fusion."""
        if self.image_requires_feature:
            image_embedding = self.image_encoder(image, image_feature)
        else:
            image_embedding = self.image_encoder(image)
        return image_embedding, self.genomic_encoder(genomic)

    def classify_embeddings(
        self,
        image_embedding: torch.Tensor,
        genomic_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict logits from already-encoded modality embeddings."""
        fused_embedding = torch.cat((image_embedding, genomic_embedding), dim=1)
        return self.classifier(fused_embedding)

    def forward(
        self,
        image: torch.Tensor,
        genomic: torch.Tensor,
        image_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict binary logits from concatenated modality embeddings."""
        image_embedding, genomic_embedding = self.encode_modalities(image, genomic, image_feature=image_feature)
        return self.classify_embeddings(image_embedding, genomic_embedding)


def build_image_classifier(
    name: str,
    image_shape: Sequence[int],
    feature_dim: int | None = None,
) -> nn.Module:
    """Build one of the supported image-only classifier variants."""
    normalized_shape = _coerce_shape(image_shape)
    if name == "baseline_cnn":
        return ImageClassifier()
    if name == "flat_mlp":
        return FlattenedImageClassifier(image_shape=normalized_shape, input_normalization="none")
    if name == "global_pool_cnn":
        return GlobalPoolingImageClassifier(input_normalization="standardize")
    if name == "hybrid":
        if feature_dim is None:
            raise ValueError("feature_dim is required for the hybrid image classifier")
        return HybridImageClassifier(
            image_shape=normalized_shape,
            feature_dim=int(feature_dim),
            image_encoder="global_pool_cnn",
        )
    raise ValueError(f"Unknown image model: {name!r}")


def build_fusion_classifier(
    genomic_input_shape: Sequence[int],
    image_shape: Sequence[int],
    image_model: str = "hybrid",
    image_feature_dim: int | None = None,
) -> nn.Module:
    """Build a fusion classifier that reuses the chosen optical encoder family."""
    return FusionClassifier(
        genomic_input_shape=genomic_input_shape,
        image_shape=image_shape,
        image_model=image_model,
        image_feature_dim=image_feature_dim,
    )
