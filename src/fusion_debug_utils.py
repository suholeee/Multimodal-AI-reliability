"""Focused helpers for diagnosing fusion instability in the sandbox pipeline."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from fusion_safety import (
    compute_safety_signals,
    disagreement_error_curve,
    image_model_kind,
    prepare_multimodal_dataset,
)
from ml_models import GenomicClassifier, build_fusion_classifier, build_image_classifier
from ml_train_eval import (
    evaluate_model,
    evaluate_probability_predictions,
    reset_model_parameters,
    resolve_device,
    set_torch_seed,
    train_model,
)


DEFAULT_ABLATION_SEEDS = (21, 22, 23, 24, 25)
LEARNED_FUSION_VARIANTS = (
    "fusion_scratch",
    "fusion_pretrained_full",
    "fusion_frozen",
    "fusion_partial",
)
ALL_FUSION_VARIANTS = (
    "late_average",
    "late_weighted",
    "late_route",
    *LEARNED_FUSION_VARIANTS,
)


def _evaluate_all_splits(
    model: nn.Module,
    splits: Dict[str, Dict[str, torch.Tensor]],
    model_kind: str,
    device: str,
    high_confidence_threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Evaluate one model on train/val/test splits."""
    return {
        split_name: evaluate_model(
            model=model,
            split=split,
            model_kind=model_kind,
            device=device,
            high_confidence_threshold=high_confidence_threshold,
        )
        for split_name, split in splits.items()
    }


def _build_model_shapes(dataset: Dict[str, object], image_model: str) -> Dict[str, object]:
    """Read the modality shapes required to rebuild sandbox models."""
    splits = dataset["splits"]
    image_shape = tuple(splits["train"]["image"].shape[1:])
    genomic_shape = tuple(splits["train"]["genomic"].shape[1:])
    image_feature_dim = int(splits["train"]["image_feature"].shape[1]) if image_model == "hybrid" else None
    return {
        "image_shape": image_shape,
        "genomic_shape": genomic_shape,
        "image_feature_dim": image_feature_dim,
    }


def train_unimodal_controls(
    dataset: Dict[str, object],
    image_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    device: str,
    seed: int,
    high_confidence_threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Train the fixed image-only and genomic-only control models once per seed."""
    shapes = _build_model_shapes(dataset, image_model=image_model)
    splits = dataset["splits"]
    controls: Dict[str, Dict[str, object]] = {}

    for offset, control_name in enumerate(("image", "genomic")):
        control_seed = int(seed + offset)
        set_torch_seed(control_seed)
        if control_name == "image":
            spec = {
                "model": build_image_classifier(
                    name=image_model,
                    image_shape=shapes["image_shape"],
                    feature_dim=shapes["image_feature_dim"],
                ),
                "model_kind": image_model_kind(image_model),
            }
        else:
            spec = {
                "model": GenomicClassifier(input_shape=shapes["genomic_shape"]),
                "model_kind": "genomic",
            }

        history = train_model(
            model=spec["model"],
            train_split=splits["train"],
            val_split=splits["val"],
            model_kind=spec["model_kind"],
            n_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            seed=control_seed,
            verbose=False,
            early_stopping_patience=patience,
            early_stopping_metric="accuracy",
            min_epochs=min_epochs,
        )
        split_metrics = _evaluate_all_splits(
            model=spec["model"],
            splits=splits,
            model_kind=spec["model_kind"],
            device=device,
            high_confidence_threshold=high_confidence_threshold,
        )
        spec["history"] = history
        spec["split_metrics"] = split_metrics
        controls[control_name] = spec

    return controls


def _weighted_average_probability(
    image_probabilities: np.ndarray,
    genomic_probabilities: np.ndarray,
    image_weight: float,
) -> np.ndarray:
    """Blend two binary probability tables with one scalar image weight."""
    weight = float(image_weight)
    return weight * np.asarray(image_probabilities, dtype=float) + (1.0 - weight) * np.asarray(
        genomic_probabilities,
        dtype=float,
    )


def _max_confidence_probability(
    image_probabilities: np.ndarray,
    genomic_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Route each sample to the unimodal branch with higher confidence."""
    image_array = np.asarray(image_probabilities, dtype=float)
    genomic_array = np.asarray(genomic_probabilities, dtype=float)
    use_image = image_array.max(axis=1) >= genomic_array.max(axis=1)
    routed = np.where(use_image[:, None], image_array, genomic_array)
    return routed, use_image


def _select_best_weight(
    image_eval: Dict[str, object],
    genomic_eval: Dict[str, object],
    high_confidence_threshold: float,
    weight_grid: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Tune one validation-only blending weight for late fusion."""
    candidate_weights = np.linspace(0.0, 1.0, 21) if weight_grid is None else np.asarray(weight_grid, dtype=float)
    best_weight = 0.5
    best_accuracy = -np.inf
    best_loss = np.inf

    for candidate in candidate_weights:
        fused_probabilities = _weighted_average_probability(
            image_eval["probabilities"],
            genomic_eval["probabilities"],
            image_weight=float(candidate),
        )
        metrics = evaluate_probability_predictions(
            probabilities=fused_probabilities,
            labels=image_eval["labels"],
            high_confidence_threshold=high_confidence_threshold,
        )
        candidate_accuracy = float(metrics["accuracy"])
        candidate_loss = float(metrics["loss"])
        if (
            candidate_accuracy > best_accuracy + 1e-8
            or (
                abs(candidate_accuracy - best_accuracy) <= 1e-8
                and candidate_loss < best_loss - 1e-8
            )
        ):
            best_weight = float(candidate)
            best_accuracy = candidate_accuracy
            best_loss = candidate_loss

    return {
        "image_weight": float(best_weight),
        "val_accuracy": float(best_accuracy),
        "val_loss": float(best_loss),
    }


def _evaluate_late_fusion_variant(
    image_metrics_by_split: Dict[str, Dict[str, object]],
    genomic_metrics_by_split: Dict[str, Dict[str, object]],
    variant_name: str,
    high_confidence_threshold: float,
    tuned_image_weight: float = 0.5,
) -> Dict[str, object]:
    """Evaluate one probability-level late-fusion baseline on every split."""
    split_metrics: Dict[str, Dict[str, object]] = {}
    route_image_fraction = []

    for split_name in ("train", "val", "test"):
        image_eval = image_metrics_by_split[split_name]
        genomic_eval = genomic_metrics_by_split[split_name]
        labels = image_eval["labels"]

        if variant_name == "late_average":
            probabilities = _weighted_average_probability(
                image_eval["probabilities"],
                genomic_eval["probabilities"],
                image_weight=0.5,
            )
        elif variant_name == "late_weighted":
            probabilities = _weighted_average_probability(
                image_eval["probabilities"],
                genomic_eval["probabilities"],
                image_weight=tuned_image_weight,
            )
        elif variant_name == "late_route":
            probabilities, use_image = _max_confidence_probability(
                image_eval["probabilities"],
                genomic_eval["probabilities"],
            )
            route_image_fraction.append(float(np.asarray(use_image, dtype=float).mean()))
        else:
            raise ValueError(f"Unknown late-fusion variant: {variant_name!r}")

        split_metrics[split_name] = evaluate_probability_predictions(
            probabilities=probabilities,
            labels=labels,
            high_confidence_threshold=high_confidence_threshold,
        )

    metadata: Dict[str, object] = {}
    if variant_name == "late_weighted":
        metadata["image_weight"] = float(tuned_image_weight)
    if route_image_fraction:
        metadata["route_to_image_fraction_test"] = float(route_image_fraction[-1])
    return {
        "split_metrics": split_metrics,
        "history": None,
        "initial_split_metrics": None,
        "metadata": metadata,
    }


def evaluate_late_fusion_variants(
    controls: Dict[str, Dict[str, object]],
    high_confidence_threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Evaluate non-learned late-fusion baselines from trained unimodal models."""
    image_metrics = controls["image"]["split_metrics"]
    genomic_metrics = controls["genomic"]["split_metrics"]
    tuned = _select_best_weight(
        image_eval=image_metrics["val"],
        genomic_eval=genomic_metrics["val"],
        high_confidence_threshold=high_confidence_threshold,
    )

    return {
        "late_average": _evaluate_late_fusion_variant(
            image_metrics_by_split=image_metrics,
            genomic_metrics_by_split=genomic_metrics,
            variant_name="late_average",
            high_confidence_threshold=high_confidence_threshold,
        ),
        "late_weighted": _evaluate_late_fusion_variant(
            image_metrics_by_split=image_metrics,
            genomic_metrics_by_split=genomic_metrics,
            variant_name="late_weighted",
            tuned_image_weight=float(tuned["image_weight"]),
            high_confidence_threshold=high_confidence_threshold,
        ),
        "late_route": _evaluate_late_fusion_variant(
            image_metrics_by_split=image_metrics,
            genomic_metrics_by_split=genomic_metrics,
            variant_name="late_route",
            high_confidence_threshold=high_confidence_threshold,
        ),
    }


def _set_module_trainable(module: nn.Module, is_trainable: bool) -> None:
    """Switch a whole module between trainable and frozen states."""
    for parameter in module.parameters():
        parameter.requires_grad = is_trainable


def _set_last_layer_trainable(module: nn.Module) -> None:
    """Expose only the final projection block of one encoder."""
    if hasattr(module, "projection"):
        _set_module_trainable(module.projection, True)
    if hasattr(module, "feature_encoder"):
        _set_module_trainable(module.feature_encoder, True)
    if hasattr(module, "image_encoder"):
        _set_last_layer_trainable(module.image_encoder)
    if hasattr(module, "network"):
        trainable_linear = None
        for child in module.network:
            if isinstance(child, nn.Linear):
                trainable_linear = child
        if trainable_linear is not None:
            for parameter in trainable_linear.parameters():
                parameter.requires_grad = True


def _copy_overlapping_state(target_module: nn.Module, source_module: nn.Module) -> Dict[str, float]:
    """Copy the overlapping portion of one state dict into another."""
    target_state = target_module.state_dict()
    source_state = source_module.state_dict()
    copied_names = 0
    copied_elements = 0
    total_elements = 0

    for name, target_tensor in target_state.items():
        total_elements += int(target_tensor.numel())
        if name not in source_state:
            continue
        source_tensor = source_state[name].detach().cpu()
        target_cpu = target_tensor.detach().cpu().clone()
        if source_tensor.ndim != target_cpu.ndim:
            continue
        overlap_shape = tuple(min(int(src), int(dst)) for src, dst in zip(source_tensor.shape, target_cpu.shape))
        if not overlap_shape:
            continue
        overlap_slices = tuple(slice(0, size) for size in overlap_shape)
        target_cpu[overlap_slices] = source_tensor[overlap_slices].to(dtype=target_cpu.dtype)
        copied_elements += int(np.prod(overlap_shape))
        copied_names += 1
        target_state[name] = target_cpu

    target_module.load_state_dict(target_state)
    return {
        "copied_parameter_tensors": float(copied_names),
        "copied_parameter_fraction": float(copied_names / max(len(target_state), 1)),
        "copied_element_fraction": float(copied_elements / max(total_elements, 1)),
    }


def _initialize_fusion_from_controls(
    fusion_model: nn.Module,
    controls: Dict[str, Dict[str, object]],
) -> Dict[str, float]:
    """Copy trained unimodal encoder weights into one fusion model."""
    image_copy_stats = _copy_overlapping_state(
        target_module=fusion_model.image_encoder,
        source_module=controls["image"]["model"].encoder,
    )
    genomic_copy_stats = _copy_overlapping_state(
        target_module=fusion_model.genomic_encoder,
        source_module=controls["genomic"]["model"].encoder,
    )
    return {
        "image_copied_parameter_fraction": float(image_copy_stats["copied_parameter_fraction"]),
        "image_copied_element_fraction": float(image_copy_stats["copied_element_fraction"]),
        "genomic_copied_parameter_fraction": float(genomic_copy_stats["copied_parameter_fraction"]),
        "genomic_copied_element_fraction": float(genomic_copy_stats["copied_element_fraction"]),
    }


def _configure_fusion_variant_trainability(model: nn.Module, variant_name: str) -> None:
    """Freeze or unfreeze the pretrained fusion model according to the ablation."""
    if variant_name == "fusion_scratch":
        return
    if variant_name == "fusion_pretrained_full":
        return
    if variant_name == "fusion_frozen":
        _set_module_trainable(model.image_encoder, False)
        _set_module_trainable(model.genomic_encoder, False)
        _set_module_trainable(model.classifier, True)
        return
    if variant_name == "fusion_partial":
        _set_module_trainable(model.image_encoder, False)
        _set_module_trainable(model.genomic_encoder, False)
        _set_module_trainable(model.classifier, True)
        _set_last_layer_trainable(model.image_encoder)
        _set_last_layer_trainable(model.genomic_encoder)
        return
    raise ValueError(f"Unknown learned fusion variant: {variant_name!r}")


def summarize_output_dominance(
    fusion_eval: Dict[str, object],
    image_eval: Dict[str, object],
    genomic_eval: Dict[str, object],
) -> Dict[str, float]:
    """Summarize whether the fused output tracks one unimodal branch more closely."""
    fusion_probability = np.asarray(fusion_eval["positive_probability"], dtype=float)
    image_probability = np.asarray(image_eval["positive_probability"], dtype=float)
    genomic_probability = np.asarray(genomic_eval["positive_probability"], dtype=float)
    image_confidence = np.asarray(image_eval["confidence"], dtype=float)
    genomic_confidence = np.asarray(genomic_eval["confidence"], dtype=float)
    fusion_prediction = np.asarray(fusion_eval["predictions"], dtype=int)
    image_prediction = np.asarray(image_eval["predictions"], dtype=int)
    genomic_prediction = np.asarray(genomic_eval["predictions"], dtype=int)

    image_gap = np.abs(fusion_probability - image_probability)
    genomic_gap = np.abs(fusion_probability - genomic_probability)
    more_confident_is_image = image_confidence >= genomic_confidence
    more_confident_prediction = np.where(more_confident_is_image, image_prediction, genomic_prediction)

    return {
        "fusion_probability_gap_to_image": float(image_gap.mean()),
        "fusion_probability_gap_to_genomic": float(genomic_gap.mean()),
        "fusion_closer_to_image_fraction": float((image_gap <= genomic_gap).mean()),
        "fusion_matches_more_confident_branch_fraction": float(
            (fusion_prediction == more_confident_prediction).mean()
        ),
        "more_confident_branch_is_image_fraction": float(more_confident_is_image.mean()),
    }


def compute_fusion_branch_diagnostics(
    model: nn.Module,
    split: Dict[str, torch.Tensor],
    image_eval: Dict[str, object],
    genomic_eval: Dict[str, object],
    fusion_eval: Dict[str, object],
    device: str,
) -> Dict[str, float]:
    """Measure branch scale and whether the learned head is ignoring one modality."""
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    with torch.no_grad():
        batch = {
            key: value.to(resolved_device) if isinstance(value, torch.Tensor) else value
            for key, value in split.items()
        }
        image_embedding, genomic_embedding = model.encode_modalities(
            batch["image"],
            batch["genomic"],
            batch.get("image_feature"),
        )
        fused_logits = model.classify_embeddings(image_embedding, genomic_embedding)
        no_image_logits = model.classify_embeddings(torch.zeros_like(image_embedding), genomic_embedding)
        no_genomic_logits = model.classify_embeddings(image_embedding, torch.zeros_like(genomic_embedding))

    fused_probability = torch.softmax(fused_logits, dim=1).cpu().numpy()[:, 1]
    no_image_probability = torch.softmax(no_image_logits, dim=1).cpu().numpy()[:, 1]
    no_genomic_probability = torch.softmax(no_genomic_logits, dim=1).cpu().numpy()[:, 1]
    image_norm = torch.norm(image_embedding.detach(), dim=1).cpu().numpy()
    genomic_norm = torch.norm(genomic_embedding.detach(), dim=1).cpu().numpy()

    diagnostics = summarize_output_dominance(
        fusion_eval=fusion_eval,
        image_eval=image_eval,
        genomic_eval=genomic_eval,
    )
    diagnostics.update(
        {
            "image_embedding_norm_mean": float(image_norm.mean()),
            "image_embedding_norm_std": float(image_norm.std(ddof=0)),
            "genomic_embedding_norm_mean": float(genomic_norm.mean()),
            "genomic_embedding_norm_std": float(genomic_norm.std(ddof=0)),
            "image_to_genomic_embedding_norm_ratio": float(image_norm.mean() / (genomic_norm.mean() + 1e-8)),
            "image_ablation_probability_shift": float(np.abs(fused_probability - no_image_probability).mean()),
            "genomic_ablation_probability_shift": float(np.abs(fused_probability - no_genomic_probability).mean()),
        }
    )
    return diagnostics


def summarize_training_history(history: Dict[str, object]) -> Dict[str, float]:
    """Reduce per-epoch curves into a few interpretable stability summaries."""
    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)
    train_accuracy = np.asarray(history["train_accuracy"], dtype=float)
    val_accuracy = np.asarray(history["val_accuracy"], dtype=float)
    image_grad = np.asarray(history.get("grad_norm_image", []), dtype=float)
    genomic_grad = np.asarray(history.get("grad_norm_genomic", []), dtype=float)
    fusion_grad = np.asarray(history.get("grad_norm_fusion_head", []), dtype=float)

    best_epoch = int(history["best_epoch"])
    best_index = max(0, best_epoch - 1)
    summary = {
        "epochs_trained": float(history["epochs_trained"]),
        "best_epoch": float(best_epoch),
        "best_val_accuracy": float(history["best_val_accuracy"]),
        "best_val_loss": float(history["best_val_loss"]),
        "train_loss_first": float(train_loss[0]),
        "train_loss_last": float(train_loss[-1]),
        "val_loss_first": float(val_loss[0]),
        "val_loss_last": float(val_loss[-1]),
        "train_accuracy_first": float(train_accuracy[0]),
        "train_accuracy_last": float(train_accuracy[-1]),
        "val_accuracy_first": float(val_accuracy[0]),
        "val_accuracy_last": float(val_accuracy[-1]),
        "train_val_gap_at_best": float(train_accuracy[best_index] - val_accuracy[best_index]),
        "val_accuracy_range": float(val_accuracy.max() - val_accuracy.min()),
    }
    if image_grad.size > 0:
        summary["mean_image_grad_norm"] = float(image_grad.mean())
    if genomic_grad.size > 0:
        summary["mean_genomic_grad_norm"] = float(genomic_grad.mean())
    if fusion_grad.size > 0:
        summary["mean_fusion_head_grad_norm"] = float(fusion_grad.mean())
    if image_grad.size > 0 and genomic_grad.size > 0:
        summary["image_to_genomic_grad_ratio"] = float(
            image_grad.mean() / (genomic_grad.mean() + 1e-8)
        )
    return summary


def _count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters for one model."""
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total_parameters": total, "trainable_parameters": trainable}


def _clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone one module state dict onto CPU for exact comparisons."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _state_dict_difference_summary(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> Dict[str, float | bool | int]:
    """Summarize whether two state dicts are identical."""
    if set(state_a) != set(state_b):
        return {
            "identical": False,
            "key_count_a": len(state_a),
            "key_count_b": len(state_b),
            "different_tensors": -1,
            "max_abs_diff": float("inf"),
        }

    different_tensors = 0
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    total_elements = 0
    for name in state_a:
        tensor_a = state_a[name]
        tensor_b = state_b[name]
        if tensor_a.shape != tensor_b.shape:
            different_tensors += 1
            max_abs_diff = float("inf")
            continue
        abs_diff = torch.abs(tensor_a.to(dtype=torch.float64) - tensor_b.to(dtype=torch.float64))
        local_max = float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0
        if local_max > 0.0:
            different_tensors += 1
        max_abs_diff = max(max_abs_diff, local_max)
        total_abs_diff += float(abs_diff.sum().item())
        total_elements += int(abs_diff.numel())

    return {
        "identical": bool(different_tensors == 0),
        "different_tensors": int(different_tensors),
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(total_abs_diff / max(total_elements, 1)),
        "key_count_a": len(state_a),
        "key_count_b": len(state_b),
    }


def _build_effective_initial_fusion_model(
    dataset: Dict[str, object],
    controls: Dict[str, Dict[str, object]],
    image_model: str,
    variant_name: str,
    seed: int,
) -> tuple[nn.Module, Dict[str, float]]:
    """Recreate the exact initial model state seen at the first training step."""
    shapes = _build_model_shapes(dataset, image_model=image_model)
    set_torch_seed(seed)
    model = build_fusion_classifier(
        genomic_input_shape=shapes["genomic_shape"],
        image_shape=shapes["image_shape"],
        image_model=image_model,
        image_feature_dim=shapes["image_feature_dim"],
    )
    metadata: Dict[str, float] = {}
    if variant_name != "fusion_scratch":
        metadata.update(_initialize_fusion_from_controls(model, controls))
    _configure_fusion_variant_trainability(model, variant_name=variant_name)
    if variant_name == "fusion_scratch":
        set_torch_seed(seed)
        reset_model_parameters(model)
    return model, metadata


def train_learned_fusion_variant(
    dataset: Dict[str, object],
    controls: Dict[str, Dict[str, object]],
    image_model: str,
    variant_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    device: str,
    seed: int,
    high_confidence_threshold: float,
) -> Dict[str, object]:
    """Train one learned fusion variant and attach diagnostics."""
    if variant_name not in LEARNED_FUSION_VARIANTS:
        raise ValueError(f"Unknown learned fusion variant: {variant_name!r}")

    shapes = _build_model_shapes(dataset, image_model=image_model)
    set_torch_seed(seed)
    fusion_model = build_fusion_classifier(
        genomic_input_shape=shapes["genomic_shape"],
        image_shape=shapes["image_shape"],
        image_model=image_model,
        image_feature_dim=shapes["image_feature_dim"],
    )

    metadata: Dict[str, float] = {}
    if variant_name != "fusion_scratch":
        metadata.update(_initialize_fusion_from_controls(fusion_model, controls))
    _configure_fusion_variant_trainability(fusion_model, variant_name=variant_name)
    metadata.update(_count_parameters(fusion_model))

    initial_split_metrics = _evaluate_all_splits(
        model=fusion_model,
        splits=dataset["splits"],
        model_kind="fusion",
        device=device,
        high_confidence_threshold=high_confidence_threshold,
    )
    history = train_model(
        model=fusion_model,
        train_split=dataset["splits"]["train"],
        val_split=dataset["splits"]["val"],
        model_kind="fusion",
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
        verbose=False,
        early_stopping_patience=patience,
        early_stopping_metric="accuracy",
        min_epochs=min_epochs,
        reset_parameters_on_start=variant_name == "fusion_scratch",
        gradient_monitor_prefixes={
            "image": ("image_encoder",),
            "genomic": ("genomic_encoder",),
            "fusion_head": ("classifier",),
        },
    )
    split_metrics = _evaluate_all_splits(
        model=fusion_model,
        splits=dataset["splits"],
        model_kind="fusion",
        device=device,
        high_confidence_threshold=high_confidence_threshold,
    )
    branch_diagnostics = compute_fusion_branch_diagnostics(
        model=fusion_model,
        split=dataset["splits"]["test"],
        image_eval=controls["image"]["split_metrics"]["test"],
        genomic_eval=controls["genomic"]["split_metrics"]["test"],
        fusion_eval=split_metrics["test"],
        device=device,
    )

    return {
        "model": fusion_model,
        "split_metrics": split_metrics,
        "initial_split_metrics": initial_split_metrics,
        "history": history,
        "training_summary": summarize_training_history(history),
        "branch_diagnostics": branch_diagnostics,
        "metadata": metadata,
    }


def attach_variant_safety_diagnostics(
    variant_result: Dict[str, object],
    image_metrics: Dict[str, Dict[str, object]],
    genomic_metrics: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Attach disagreement-driven oversight diagnostics to one fusion result."""
    test_signals = compute_safety_signals(
        image_eval=image_metrics["test"],
        genomic_eval=genomic_metrics["test"],
        fusion_eval=variant_result["split_metrics"]["test"],
    )
    disagreement_curve = disagreement_error_curve(
        disagreement_score=test_signals["disagreement_score"],
        correct_mask=variant_result["split_metrics"]["test"]["correct_mask"],
    )
    variant_result["signals"] = test_signals
    variant_result["disagreement_curve"] = disagreement_curve
    if "branch_diagnostics" not in variant_result:
        variant_result["branch_diagnostics"] = summarize_output_dominance(
            fusion_eval=variant_result["split_metrics"]["test"],
            image_eval=image_metrics["test"],
            genomic_eval=genomic_metrics["test"],
        )
    else:
        variant_result["branch_diagnostics"].update(
            summarize_output_dominance(
                fusion_eval=variant_result["split_metrics"]["test"],
                image_eval=image_metrics["test"],
                genomic_eval=genomic_metrics["test"],
            )
        )
    return variant_result


def run_fusion_ablation_for_seed(
    seed: int,
    mode: str,
    image_size: int,
    genomic_representation: str,
    image_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    device: str,
    high_confidence_threshold: float,
) -> Dict[str, object]:
    """Run the full fusion-ablation suite for one random seed."""
    dataset = prepare_multimodal_dataset(
        mode=mode,
        image_size=image_size,
        genomic_representation=genomic_representation,
        seed=seed,
        image_model=image_model,
    )
    controls = train_unimodal_controls(
        dataset=dataset,
        image_model=image_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_epochs=min_epochs,
        device=device,
        seed=seed,
        high_confidence_threshold=high_confidence_threshold,
    )
    variants = evaluate_late_fusion_variants(
        controls=controls,
        high_confidence_threshold=high_confidence_threshold,
    )
    for offset, variant_name in enumerate(LEARNED_FUSION_VARIANTS):
        variants[variant_name] = train_learned_fusion_variant(
            dataset=dataset,
            controls=controls,
            image_model=image_model,
            variant_name=variant_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            min_epochs=min_epochs,
            device=device,
            seed=seed + 20 + offset,
            high_confidence_threshold=high_confidence_threshold,
        )

    image_metrics = controls["image"]["split_metrics"]
    genomic_metrics = controls["genomic"]["split_metrics"]
    for variant_name in list(variants):
        variants[variant_name] = attach_variant_safety_diagnostics(
            variant_result=variants[variant_name],
            image_metrics=image_metrics,
            genomic_metrics=genomic_metrics,
        )

    return {
        "dataset_meta": {
            "mode": dataset["mode"],
            "image_size": dataset["image_size"],
            "genomic_representation": dataset["genomic_representation"],
            "image_model": image_model,
            "n_monomers": dataset["n_monomers"],
        },
        "controls": controls,
        "variants": variants,
    }


def debug_scratch_vs_pretrained_for_seed(
    seed: int,
    mode: str,
    image_size: int,
    genomic_representation: str,
    image_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    device: str,
    high_confidence_threshold: float,
) -> Dict[str, object]:
    """Compare scratch and pretrained fusion behavior for one seed."""
    dataset = prepare_multimodal_dataset(
        mode=mode,
        image_size=image_size,
        genomic_representation=genomic_representation,
        seed=seed,
        image_model=image_model,
    )
    controls = train_unimodal_controls(
        dataset=dataset,
        image_model=image_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_epochs=min_epochs,
        device=device,
        seed=seed,
        high_confidence_threshold=high_confidence_threshold,
    )

    scratch_train_seed = seed + 20
    pretrained_train_seed = seed + 21
    scratch_initial_model, scratch_initial_metadata = _build_effective_initial_fusion_model(
        dataset=dataset,
        controls=controls,
        image_model=image_model,
        variant_name="fusion_scratch",
        seed=scratch_train_seed,
    )
    pretrained_initial_model, pretrained_initial_metadata = _build_effective_initial_fusion_model(
        dataset=dataset,
        controls=controls,
        image_model=image_model,
        variant_name="fusion_pretrained_full",
        seed=pretrained_train_seed,
    )

    scratch_initial_image = _clone_state_dict(scratch_initial_model.image_encoder)
    scratch_initial_genomic = _clone_state_dict(scratch_initial_model.genomic_encoder)
    pretrained_initial_image = _clone_state_dict(pretrained_initial_model.image_encoder)
    pretrained_initial_genomic = _clone_state_dict(pretrained_initial_model.genomic_encoder)
    control_image = _clone_state_dict(controls["image"]["model"].encoder)
    control_genomic = _clone_state_dict(controls["genomic"]["model"].encoder)

    scratch_result = train_learned_fusion_variant(
        dataset=dataset,
        controls=controls,
        image_model=image_model,
        variant_name="fusion_scratch",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_epochs=min_epochs,
        device=device,
        seed=scratch_train_seed,
        high_confidence_threshold=high_confidence_threshold,
    )
    pretrained_result = train_learned_fusion_variant(
        dataset=dataset,
        controls=controls,
        image_model=image_model,
        variant_name="fusion_pretrained_full",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_epochs=min_epochs,
        device=device,
        seed=pretrained_train_seed,
        high_confidence_threshold=high_confidence_threshold,
    )

    variants = {
        "fusion_scratch": scratch_result,
        "fusion_pretrained_full": pretrained_result,
    }
    image_metrics = controls["image"]["split_metrics"]
    genomic_metrics = controls["genomic"]["split_metrics"]
    for variant_name in list(variants):
        variants[variant_name] = attach_variant_safety_diagnostics(
            variant_result=variants[variant_name],
            image_metrics=image_metrics,
            genomic_metrics=genomic_metrics,
        )

    scratch_final_state = _clone_state_dict(scratch_result["model"])
    pretrained_final_state = _clone_state_dict(pretrained_result["model"])

    return {
        "seed": seed,
        "train_seeds": {
            "fusion_scratch": scratch_train_seed,
            "fusion_pretrained_full": pretrained_train_seed,
        },
        "accuracies": {
            "fusion_scratch": float(scratch_result["split_metrics"]["test"]["accuracy"]),
            "fusion_pretrained_full": float(pretrained_result["split_metrics"]["test"]["accuracy"]),
        },
        "initial_comparisons": {
            "image_encoder": _state_dict_difference_summary(
                scratch_initial_image,
                pretrained_initial_image,
            ),
            "genomic_encoder": _state_dict_difference_summary(
                scratch_initial_genomic,
                pretrained_initial_genomic,
            ),
        },
        "pretrained_load_checks": {
            "metadata_from_manual_init": dict(pretrained_initial_metadata),
            "metadata_from_training_path": dict(pretrained_result["metadata"]),
            "pretrained_image_matches_control": _state_dict_difference_summary(
                pretrained_initial_image,
                control_image,
            ),
            "pretrained_genomic_matches_control": _state_dict_difference_summary(
                pretrained_initial_genomic,
                control_genomic,
            ),
            "scratch_image_matches_control": _state_dict_difference_summary(
                scratch_initial_image,
                control_image,
            ),
            "scratch_genomic_matches_control": _state_dict_difference_summary(
                scratch_initial_genomic,
                control_genomic,
            ),
            "scratch_manual_metadata": dict(scratch_initial_metadata),
        },
        "final_weight_comparison": _state_dict_difference_summary(
            scratch_final_state,
            pretrained_final_state,
        ),
        "storage_checks": {
            "different_result_dict_objects": variants["fusion_scratch"] is not variants["fusion_pretrained_full"],
            "different_split_metric_objects": (
                variants["fusion_scratch"]["split_metrics"] is not variants["fusion_pretrained_full"]["split_metrics"]
            ),
            "different_signal_objects": variants["fusion_scratch"]["signals"] is not variants["fusion_pretrained_full"]["signals"],
            "stored_keys": list(variants.keys()),
        },
        "variants": variants,
    }
