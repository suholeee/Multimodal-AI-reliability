"""Minimal training, evaluation, and oversight utilities for sandbox ML experiments."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Choose a torch device with a simple auto fallback."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_torch_seed(seed: int) -> None:
    """Seed torch for reproducible training runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Compute normalized categorical entropy for each prediction."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / np.log(clipped.shape[1])


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor values in a batch dictionary onto one device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _slice_batch(
    split: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract one mini-batch by index."""
    batch = {
        key: value[indices]
        for key, value in split.items()
        if isinstance(value, torch.Tensor)
    }
    return _batch_to_device(batch, device=device)


def _forward_model(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    model_kind: str,
    input_key: Optional[str] = None,
) -> torch.Tensor:
    """Run the correct forward pass for one model kind."""
    if model_kind == "image":
        return model(batch["image"])
    if model_kind == "image_hybrid":
        return model(batch["image"], batch["image_feature"])
    if model_kind == "genomic":
        return model(batch["genomic"])
    if model_kind == "fusion":
        return model(batch["image"], batch["genomic"], batch.get("image_feature"))
    if model_kind == "feature":
        if input_key is None:
            raise ValueError("input_key is required for feature models")
        return model(batch[input_key])
    raise ValueError(f"Unknown model_kind: {model_kind!r}")


def _safe_mean(values: np.ndarray) -> float:
    """Average an array when it is non-empty."""
    return float(values.mean()) if values.size > 0 else float("nan")


def _confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, int]:
    """Return binary confusion matrix counts."""
    return {
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
    }


def _cross_entropy_from_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Estimate cross-entropy directly from predicted probabilities."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
    label_array = np.asarray(labels, dtype=np.int64)
    losses = -np.log(clipped[np.arange(len(label_array)), label_array])
    return float(losses.mean())


def reset_model_parameters(model: nn.Module) -> None:
    """Reset module parameters after seeding so initialization is reproducible."""
    for module in model.modules():
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()


def _selects_better_checkpoint(
    metric_name: str,
    metric_value: float,
    aux_loss: float,
    aux_accuracy: float,
    best_metric_value: float,
    best_aux_loss: float,
    best_aux_accuracy: float,
    atol: float = 1e-8,
) -> bool:
    """Decide whether the current epoch is a better checkpoint candidate."""
    if metric_name == "accuracy":
        if metric_value > best_metric_value + atol:
            return True
        if abs(metric_value - best_metric_value) <= atol and aux_loss < best_aux_loss - atol:
            return True
        return False
    if metric_name == "loss":
        if metric_value < best_metric_value - atol:
            return True
        if abs(metric_value - best_metric_value) <= atol and aux_accuracy > best_aux_accuracy + atol:
            return True
        return False
    raise ValueError(f"Unknown checkpoint metric: {metric_name!r}")


def evaluate_probability_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    high_confidence_threshold: float = 0.80,
    loss: Optional[float] = None,
) -> Dict[str, object]:
    """Evaluate already-computed probabilities with the same summary as model evaluation."""
    probability_array = np.asarray(probabilities, dtype=float)
    label_array = np.asarray(labels, dtype=np.int64)

    predictions = probability_array.argmax(axis=1)
    positive_probability = probability_array[:, 1]
    confidence = probability_array.max(axis=1)
    entropy = compute_entropy(probability_array)
    correct_mask = predictions == label_array
    high_confidence_mask = confidence >= high_confidence_threshold
    resolved_loss = (
        float(loss)
        if loss is not None
        else _cross_entropy_from_probabilities(probability_array, label_array)
    )

    return {
        "loss": resolved_loss,
        "labels": label_array,
        "predictions": predictions,
        "probabilities": probability_array,
        "positive_probability": positive_probability,
        "confidence": confidence,
        "entropy": entropy,
        "correct_mask": correct_mask,
        "accuracy": float(correct_mask.mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": _safe_mean(confidence[correct_mask]),
        "mean_confidence_incorrect": _safe_mean(confidence[~correct_mask]),
        "high_confidence_threshold": float(high_confidence_threshold),
        "high_confidence_fraction": float(high_confidence_mask.mean()),
        "high_confidence_error_rate": _safe_mean((~correct_mask[high_confidence_mask]).astype(float)),
        "confusion_matrix": _confusion_counts(label_array, predictions),
    }


def evaluate_model(
    model: nn.Module,
    split: Dict[str, torch.Tensor],
    model_kind: str,
    device: Optional[str] = None,
    high_confidence_threshold: float = 0.80,
    input_key: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate a model on one split and collect confidence diagnostics."""
    resolved_device = resolve_device(device)
    criterion = nn.CrossEntropyLoss()
    model = model.to(resolved_device)
    model.eval()

    with torch.no_grad():
        batch = _batch_to_device(split, resolved_device)
        logits = _forward_model(model, batch, model_kind=model_kind, input_key=input_key)
        loss = criterion(logits, batch["label"]).item()
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        labels = batch["label"].cpu().numpy()
    return evaluate_probability_predictions(
        probabilities=probabilities,
        labels=labels,
        high_confidence_threshold=high_confidence_threshold,
        loss=float(loss),
    )


def train_model(
    model: nn.Module,
    train_split: Dict[str, torch.Tensor],
    val_split: Optional[Dict[str, torch.Tensor]],
    model_kind: str,
    n_epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
    input_key: Optional[str] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_metric: str = "accuracy",
    min_epochs: int = 1,
    reset_parameters_on_start: bool = True,
    gradient_monitor_prefixes: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, object]:
    """Train one small model with a simple mini-batch loop."""
    if early_stopping_metric not in {"accuracy", "loss"}:
        raise ValueError("early_stopping_metric must be 'accuracy' or 'loss'")
    if min_epochs < 1:
        raise ValueError("min_epochs must be at least 1")

    set_torch_seed(seed)
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    if reset_parameters_on_start:
        reset_model_parameters(model)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters")

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    n_train = len(train_split["label"])
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(seed)
    named_parameters = list(model.named_parameters())

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    if gradient_monitor_prefixes is not None:
        for group_name in gradient_monitor_prefixes:
            history[f"grad_norm_{group_name}"] = []
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    if early_stopping_metric == "accuracy":
        best_monitor_value = -np.inf
        best_aux_accuracy = -np.inf
        best_aux_loss = np.inf
    else:
        best_monitor_value = np.inf
        best_aux_accuracy = -np.inf
        best_aux_loss = np.inf
    stopped_early = False

    for epoch_idx in range(1, n_epochs + 1):
        model.train()
        permutation = torch.randperm(n_train, generator=permutation_generator)

        total_loss = 0.0
        total_correct = 0
        gradient_accumulator = None
        gradient_steps = 0
        if gradient_monitor_prefixes is not None:
            gradient_accumulator = {
                group_name: 0.0
                for group_name in gradient_monitor_prefixes
            }
        for start_idx in range(0, n_train, batch_size):
            batch_indices = permutation[start_idx : start_idx + batch_size]
            batch = _slice_batch(train_split, batch_indices, device=resolved_device)

            optimizer.zero_grad()
            logits = _forward_model(model, batch, model_kind=model_kind, input_key=input_key)
            loss = criterion(logits, batch["label"])
            loss.backward()
            if gradient_monitor_prefixes is not None and gradient_accumulator is not None:
                for group_name, prefixes in gradient_monitor_prefixes.items():
                    squared_norm = 0.0
                    for parameter_name, parameter in named_parameters:
                        if parameter.grad is None:
                            continue
                        if any(parameter_name.startswith(prefix) for prefix in prefixes):
                            squared_norm += float(torch.sum(parameter.grad.detach() ** 2).item())
                    gradient_accumulator[group_name] += float(np.sqrt(squared_norm))
                gradient_steps += 1
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_indices)
            total_correct += int((logits.argmax(dim=1) == batch["label"]).sum().item())

        train_loss = total_loss / n_train
        train_accuracy = total_correct / n_train
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_accuracy))
        if gradient_monitor_prefixes is not None and gradient_accumulator is not None:
            denom = max(gradient_steps, 1)
            for group_name, total_norm in gradient_accumulator.items():
                history[f"grad_norm_{group_name}"].append(float(total_norm / denom))

        if val_split is None:
            val_metrics = {"loss": train_loss, "accuracy": train_accuracy}
        else:
            val_metrics = evaluate_model(
                model,
                val_split,
                model_kind=model_kind,
                device=resolved_device,
                input_key=input_key,
            )
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))

        monitor_value = float(val_metrics[early_stopping_metric])
        aux_loss = float(val_metrics["loss"])
        aux_accuracy = float(val_metrics["accuracy"])
        improved = _selects_better_checkpoint(
            metric_name=early_stopping_metric,
            metric_value=monitor_value,
            aux_loss=aux_loss,
            aux_accuracy=aux_accuracy,
            best_metric_value=float(best_monitor_value),
            best_aux_loss=float(best_aux_loss),
            best_aux_accuracy=float(best_aux_accuracy),
        )

        if improved:
            best_monitor_value = monitor_value
            best_aux_loss = aux_loss
            best_aux_accuracy = aux_accuracy
            best_epoch = epoch_idx
            epochs_without_improvement = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if verbose and (epoch_idx == 1 or epoch_idx == n_epochs or epoch_idx % max(1, n_epochs // 5) == 0):
            print(
                f"  epoch {epoch_idx:>3d}/{n_epochs}  "
                f"train_loss={train_loss:.3f}  train_acc={train_accuracy:.3f}  "
                f"val_acc={float(val_metrics['accuracy']):.3f}"
            )

        if (
            early_stopping_patience is not None
            and epoch_idx >= max(min_epochs, 1)
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            if verbose:
                print(
                    f"  early_stop epoch={epoch_idx:d}  "
                    f"best_epoch={best_epoch:d}  "
                    f"best_val_{early_stopping_metric}={best_monitor_value:.3f}"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = int(best_epoch)
    history["best_val_metric"] = float(best_monitor_value)
    history["best_val_loss"] = float(best_aux_loss)
    history["best_val_accuracy"] = float(best_aux_accuracy)
    history["early_stopped"] = bool(stopped_early)
    history["epochs_trained"] = len(history["train_loss"])
    return history


def oversight_filter(
    image_eval: Dict[str, object],
    genomic_eval: Dict[str, object],
    fusion_eval: Dict[str, object],
    disagreement_threshold: float = 0.35,
    entropy_threshold: float = 0.65,
    route_strategy: str = "lower_entropy",
) -> Dict[str, object]:
    """Flag unreliable fusion predictions using simple output-only rules."""
    image_prob = np.asarray(image_eval["positive_probability"], dtype=float)
    genomic_prob = np.asarray(genomic_eval["positive_probability"], dtype=float)
    image_pred = np.asarray(image_eval["predictions"], dtype=int)
    genomic_pred = np.asarray(genomic_eval["predictions"], dtype=int)
    fusion_pred = np.asarray(fusion_eval["predictions"], dtype=int)
    fusion_confidence = np.asarray(fusion_eval["confidence"], dtype=float)
    labels = np.asarray(fusion_eval["labels"], dtype=int)

    image_entropy = np.asarray(image_eval["entropy"], dtype=float)
    genomic_entropy = np.asarray(genomic_eval["entropy"], dtype=float)
    fusion_entropy = np.asarray(fusion_eval["entropy"], dtype=float)

    prediction_disagreement = image_pred != genomic_pred
    disagreement_score = np.abs(image_prob - genomic_prob)
    flags = (prediction_disagreement & (disagreement_score >= disagreement_threshold)) | (
        fusion_entropy >= entropy_threshold
    )
    unflagged = ~flags

    if route_strategy not in {"lower_entropy", "abstain"}:
        raise ValueError("route_strategy must be 'lower_entropy' or 'abstain'")

    routed_predictions = fusion_pred.copy()
    routed_confidence = fusion_confidence.copy()
    routed_positive_probability = np.asarray(fusion_eval["positive_probability"], dtype=float).copy()

    if route_strategy == "lower_entropy":
        use_image = image_entropy <= genomic_entropy
        routed_predictions[flags] = np.where(use_image[flags], image_pred[flags], genomic_pred[flags])
        routed_confidence[flags] = np.where(
            use_image[flags],
            np.asarray(image_eval["confidence"], dtype=float)[flags],
            np.asarray(genomic_eval["confidence"], dtype=float)[flags],
        )
        routed_positive_probability[flags] = np.where(use_image[flags], image_prob[flags], genomic_prob[flags])

    routed_correct = routed_predictions == labels
    fusion_correct = fusion_pred == labels

    return {
        "flags": flags,
        "flagged_fraction": float(flags.mean()),
        "retained_fraction": float(unflagged.mean()),
        "prediction_disagreement": prediction_disagreement,
        "disagreement_score": disagreement_score,
        "fusion_entropy": fusion_entropy,
        "error_rate_flagged": _safe_mean((~fusion_correct[flags]).astype(float)),
        "accuracy_unflagged": _safe_mean(fusion_correct[unflagged].astype(float)),
        "selective_accuracy": _safe_mean(fusion_correct[unflagged].astype(float)),
        "routed_predictions": routed_predictions,
        "routed_confidence": routed_confidence,
        "routed_positive_probability": routed_positive_probability,
        "routed_accuracy": float(routed_correct.mean()),
        "route_strategy": route_strategy,
        "entropy_threshold": float(entropy_threshold),
        "disagreement_threshold": float(disagreement_threshold),
    }


def reliability_metrics(
    image_eval: Dict[str, object],
    genomic_eval: Dict[str, object],
    fusion_eval: Dict[str, object],
    disagreement_threshold: float = 0.35,
    entropy_threshold: float = 0.65,
    route_strategy: str = "lower_entropy",
) -> Dict[str, object]:
    """Collect the core oversight statistics requested by the demo."""
    oversight = oversight_filter(
        image_eval=image_eval,
        genomic_eval=genomic_eval,
        fusion_eval=fusion_eval,
        disagreement_threshold=disagreement_threshold,
        entropy_threshold=entropy_threshold,
        route_strategy=route_strategy,
    )
    return {
        "image_probability": np.asarray(image_eval["positive_probability"], dtype=float),
        "genomic_probability": np.asarray(genomic_eval["positive_probability"], dtype=float),
        "fusion_probability": np.asarray(fusion_eval["positive_probability"], dtype=float),
        "image_entropy": np.asarray(image_eval["entropy"], dtype=float),
        "genomic_entropy": np.asarray(genomic_eval["entropy"], dtype=float),
        "fusion_entropy": np.asarray(fusion_eval["entropy"], dtype=float),
        "flags": oversight["flags"],
        "flagged_fraction": oversight["flagged_fraction"],
        "error_rate_flagged": oversight["error_rate_flagged"],
        "accuracy_unflagged": oversight["accuracy_unflagged"],
        "selective_accuracy": oversight["selective_accuracy"],
        "routed_accuracy": oversight["routed_accuracy"],
        "disagreement_score": oversight["disagreement_score"],
        "prediction_disagreement": oversight["prediction_disagreement"],
        "routed_predictions": oversight["routed_predictions"],
        "routed_confidence": oversight["routed_confidence"],
        "entropy_threshold": oversight["entropy_threshold"],
        "disagreement_threshold": oversight["disagreement_threshold"],
    }
