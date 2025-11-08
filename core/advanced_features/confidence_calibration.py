"""
Confidence Calibration Dinámica - Sistema de calibración automática de confianza
Implementa calibración dinámica de scores de confianza con ajuste automático basado en feedback
y métricas de rendimiento histórico
"""

import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, deque
import statistics


class CalibrationMethod(Enum):
    PLATT_SCALING = "platt_scaling"          # Calibración de Platt
    ISOTONIC_REGRESSION = "isotonic"         # Regresión isotónica
    TEMPERATURE_SCALING = "temperature"      # Temperature scaling
    HISTOGRAM_BINNING = "histogram"          # Histogram binning
    BAYESIAN_BINNING = "bayesian"           # Bayesian binning
    DYNAMIC_THRESHOLD = "dynamic_threshold"  # Umbralización dinámica


class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class ConfidenceScore:
    raw_score: float
    calibrated_score: float
    confidence_level: ConfidenceLevel
    calibration_method: CalibrationMethod
    uncertainty_estimate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationFeedback:
    predicted_confidence: float
    actual_correctness: bool
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationMetrics:
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float   # MCE
    brier_score: float
    log_loss: float
    reliability_score: float
    sharpness_score: float
    calibration_bins: Dict[str, Any]
    sample_count: int


class PlattCalibrator:
    """Implementa calibración de Platt usando regresión sigmoide"""

    def __init__(self):
        self.A = 1.0  # Parámetro de pendiente
        self.B = 0.0  # Parámetro de sesgo
        self.is_fitted = False

    def fit(self, scores: List[float], labels: List[bool]) -> None:
        """Ajusta los parámetros A y B usando maximum likelihood"""
        if len(scores) != len(labels) or len(scores) < 10:
            raise ValueError("Insufficient data for Platt calibration")

        scores = np.array(scores)
        labels = np.array(labels, dtype=float)

        # Inicialización de parámetros
        prior1 = np.sum(labels)
        prior0 = len(labels) - prior1

        # Método de Newton-Raphson simplificado
        # En implementación completa se usaría optimización más robusta
        self.A, self.B = self._newton_raphson_fit(scores, labels, prior0, prior1)
        self.is_fitted = True

    def _newton_raphson_fit(self, scores: np.ndarray, labels: np.ndarray,
                           prior0: float, prior1: float) -> Tuple[float, float]:
        """Ajuste usando Newton-Raphson (simplificado)"""
        # Parámetros iniciales
        A = 1.0
        B = math.log((prior0 + 1.0) / (prior1 + 1.0))

        # Iteraciones de optimización
        for _ in range(100):
            # Calcular probabilidades actuales
            linear = A * scores + B
            pp = 1.0 / (1.0 + np.exp(-linear))

            # Gradiente
            gradient_A = np.sum((labels - pp) * scores)
            gradient_B = np.sum(labels - pp)

            # Hessiano (aproximado)
            hessian_AA = -np.sum(pp * (1 - pp) * scores * scores)
            hessian_AB = -np.sum(pp * (1 - pp) * scores)
            hessian_BB = -np.sum(pp * (1 - pp))

            # Actualización
            det = hessian_AA * hessian_BB - hessian_AB * hessian_AB
            if abs(det) < 1e-12:
                break

            dA = -(hessian_BB * gradient_A - hessian_AB * gradient_B) / det
            dB = -(hessian_AA * gradient_B - hessian_AB * gradient_A) / det

            A += dA
            B += dB

            # Convergencia
            if abs(dA) < 1e-7 and abs(dB) < 1e-7:
                break

        return A, B

    def calibrate(self, score: float) -> float:
        """Aplica calibración de Platt a un score"""
        if not self.is_fitted:
            return score

        linear = self.A * score + self.B
        return 1.0 / (1.0 + math.exp(-linear))


class TemperatureScaler:
    """Implementa temperature scaling para calibración"""

    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    def fit(self, scores: List[float], labels: List[bool]) -> None:
        """Encuentra la temperatura óptima minimizando log-loss"""
        if len(scores) != len(labels) or len(scores) < 5:
            raise ValueError("Insufficient data for temperature scaling")

        scores = np.array(scores)
        labels = np.array(labels)

        # Búsqueda de temperatura óptima usando búsqueda de línea
        best_temp = 1.0
        best_loss = float('inf')

        for temp in np.linspace(0.1, 5.0, 50):
            calibrated = self._apply_temperature(scores, temp)
            loss = self._log_loss(labels, calibrated)

            if loss < best_loss:
                best_loss = loss
                best_temp = temp

        self.temperature = best_temp
        self.is_fitted = True

    def _apply_temperature(self, scores: np.ndarray, temperature: float) -> np.ndarray:
        """Aplica temperature scaling a los scores"""
        # Para scores que no son logits, aplicamos transformación
        logits = np.log(scores / (1 - scores + 1e-8))
        return 1 / (1 + np.exp(-logits / temperature))

    def _log_loss(self, labels: np.ndarray, probs: np.ndarray) -> float:
        """Calcula log loss"""
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    def calibrate(self, score: float) -> float:
        """Aplica temperature scaling a un score"""
        if not self.is_fitted:
            return score

        if score <= 0 or score >= 1:
            return score

        logit = math.log(score / (1 - score))
        return 1 / (1 + math.exp(-logit / self.temperature))


class HistogramCalibrator:
    """Implementa calibración por binning histograma"""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_boundaries = None
        self.bin_true_probs = None
        self.is_fitted = False

    def fit(self, scores: List[float], labels: List[bool]) -> None:
        """Construye bins y calcula probabilidades verdaderas"""
        if len(scores) != len(labels) or len(scores) < self.n_bins:
            raise ValueError("Insufficient data for histogram calibration")

        scores = np.array(scores)
        labels = np.array(labels)

        # Crear bins con igual número de muestras
        sorted_indices = np.argsort(scores)
        bin_size = len(scores) // self.n_bins

        self.bin_boundaries = []
        self.bin_true_probs = []

        for i in range(self.n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < self.n_bins - 1 else len(scores)

            bin_indices = sorted_indices[start_idx:end_idx]
            bin_labels = labels[bin_indices]
            bin_scores = scores[bin_indices]

            # Límites del bin
            bin_min = float(bin_scores.min())
            bin_max = float(bin_scores.max())
            self.bin_boundaries.append((bin_min, bin_max))

            # Probabilidad verdadera del bin
            true_prob = float(bin_labels.mean()) if len(bin_labels) > 0 else 0.5
            self.bin_true_probs.append(true_prob)

        self.is_fitted = True

    def calibrate(self, score: float) -> float:
        """Aplica calibración por histogram binning"""
        if not self.is_fitted:
            return score

        # Encontrar el bin correspondiente
        for i, (bin_min, bin_max) in enumerate(self.bin_boundaries):
            if bin_min <= score <= bin_max:
                return self.bin_true_probs[i]

        # Si no está en ningún bin, usar el bin más cercano
        distances = []
        for bin_min, bin_max in self.bin_boundaries:
            bin_center = (bin_min + bin_max) / 2
            distances.append(abs(score - bin_center))

        closest_bin = int(np.argmin(distances))
        return self.bin_true_probs[closest_bin]


class DynamicConfidenceCalibrator:
    """Sistema principal de calibración dinámica de confianza"""

    def __init__(self,
                 methods: List[CalibrationMethod] = None,
                 window_size: int = 1000,
                 min_samples_for_recalibration: int = 50):

        self.methods = methods or [
            CalibrationMethod.PLATT_SCALING,
            CalibrationMethod.TEMPERATURE_SCALING,
            CalibrationMethod.HISTOGRAM_BINNING
        ]

        self.window_size = window_size
        self.min_samples_for_recalibration = min_samples_for_recalibration

        # Calibradores por método
        self.calibrators = {
            CalibrationMethod.PLATT_SCALING: PlattCalibrator(),
            CalibrationMethod.TEMPERATURE_SCALING: TemperatureScaler(),
            CalibrationMethod.HISTOGRAM_BINNING: HistogramCalibrator()
        }

        # Historia de feedback
        self.feedback_history = deque(maxlen=window_size)
        self.method_performance = defaultdict(list)
        self.current_best_method = CalibrationMethod.PLATT_SCALING

        # Métricas y estadísticas
        self.calibration_stats = {}
        self.last_recalibration = 0
        self.recalibration_frequency = 100  # Cada N nuevos feedbacks

    def add_feedback(self,
                    predicted_confidence: float,
                    actual_correctness: bool,
                    context: Dict[str, Any] = None) -> None:
        """Añade feedback de rendimiento real"""

        feedback = CalibrationFeedback(
            predicted_confidence=predicted_confidence,
            actual_correctness=actual_correctness,
            timestamp=time.time(),
            context=context or {}
        )

        self.feedback_history.append(feedback)

        # Recalibrar si es necesario
        if (len(self.feedback_history) - self.last_recalibration >=
            self.recalibration_frequency):
            self._recalibrate_all_methods()

    def _recalibrate_all_methods(self) -> None:
        """Recalibra todos los métodos con datos recientes"""

        if len(self.feedback_history) < self.min_samples_for_recalibration:
            return

        # Extraer scores y labels
        scores = [f.predicted_confidence for f in self.feedback_history]
        labels = [f.actual_correctness for f in self.feedback_history]

        # Recalibrar cada método
        method_errors = {}

        for method in self.methods:
            try:
                calibrator = self.calibrators[method]
                calibrator.fit(scores, labels)

                # Evaluar performance en validation split
                val_error = self._evaluate_calibrator(calibrator, scores, labels)
                method_errors[method] = val_error

            except Exception as e:
                print(f"Error calibrating method {method}: {e}")
                method_errors[method] = float('inf')

        # Seleccionar mejor método
        if method_errors:
            self.current_best_method = min(method_errors.items(), key=lambda x: x[1])[0]

        self.last_recalibration = len(self.feedback_history)

    def _evaluate_calibrator(self,
                           calibrator,
                           scores: List[float],
                           labels: List[bool]) -> float:
        """Evalúa performance de un calibrador usando cross-validation simple"""

        # Split simple 80/20
        split_idx = int(len(scores) * 0.8)

        train_scores = scores[:split_idx]
        train_labels = labels[:split_idx]
        val_scores = scores[split_idx:]
        val_labels = labels[split_idx:]

        if len(val_scores) < 10:
            return float('inf')

        try:
            # Re-entrenar en subset
            calibrator.fit(train_scores, train_labels)

            # Evaluar en validation
            calibrated_scores = [calibrator.calibrate(s) for s in val_scores]
            ece = self._calculate_ece(calibrated_scores, val_labels)

            return ece

        except Exception:
            return float('inf')

    def calibrate_confidence(self,
                           raw_score: float,
                           method: CalibrationMethod = None,
                           context: Dict[str, Any] = None) -> ConfidenceScore:
        """Calibra un score de confianza usando el mejor método disponible"""

        # Usar método especificado o el mejor actual
        active_method = method or self.current_best_method

        # Obtener calibrador
        calibrator = self.calibrators.get(active_method)
        if not calibrator:
            # Fallback a score raw
            calibrated_score = raw_score
            active_method = CalibrationMethod.DYNAMIC_THRESHOLD
        else:
            try:
                calibrated_score = calibrator.calibrate(raw_score)
            except Exception:
                calibrated_score = raw_score

        # Calcular incertidumbre estimada
        uncertainty = self._estimate_uncertainty(raw_score, calibrated_score, context)

        # Determinar nivel de confianza
        confidence_level = self._determine_confidence_level(calibrated_score)

        return ConfidenceScore(
            raw_score=raw_score,
            calibrated_score=calibrated_score,
            confidence_level=confidence_level,
            calibration_method=active_method,
            uncertainty_estimate=uncertainty,
            metadata={
                'context': context or {},
                'feedback_samples': len(self.feedback_history),
                'last_recalibration': self.last_recalibration
            }
        )

    def _estimate_uncertainty(self,
                            raw_score: float,
                            calibrated_score: float,
                            context: Dict[str, Any] = None) -> float:
        """Estima la incertidumbre del score calibrado"""

        # Factores de incertidumbre
        uncertainty_factors = []

        # 1. Diferencia entre raw y calibrated (indica cuánta corrección se aplicó)
        calibration_gap = abs(raw_score - calibrated_score)
        uncertainty_factors.append(calibration_gap)

        # 2. Proximidad a los extremos (scores extremos son menos confiables)
        extreme_proximity = 2 * min(calibrated_score, 1 - calibrated_score)
        uncertainty_factors.append(1 - extreme_proximity)

        # 3. Cantidad de datos de calibración disponibles
        if len(self.feedback_history) < 100:
            data_uncertainty = 1 - (len(self.feedback_history) / 100)
            uncertainty_factors.append(data_uncertainty)

        # 4. Variabilidad histórica del método actual
        if self.current_best_method in self.method_performance:
            method_variance = statistics.stdev(self.method_performance[self.current_best_method][-50:])
            uncertainty_factors.append(method_variance)

        # Combinar factores
        if uncertainty_factors:
            uncertainty = np.mean(uncertainty_factors)
        else:
            uncertainty = 0.5  # Default

        return min(max(uncertainty, 0.0), 1.0)

    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determina el nivel de confianza categórico"""
        if score < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.4:
            return ConfidenceLevel.LOW
        elif score < 0.6:
            return ConfidenceLevel.MEDIUM
        elif score < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def calculate_calibration_metrics(self,
                                    n_bins: int = 10) -> CalibrationMetrics:
        """Calcula métricas de calibración comprehensivas"""

        if len(self.feedback_history) < n_bins:
            return CalibrationMetrics(0, 0, 0, 0, 0, 0, {}, len(self.feedback_history))

        predictions = [f.predicted_confidence for f in self.feedback_history]
        actuals = [f.actual_correctness for f in self.feedback_history]

        # Recalibrar predicciones con método actual
        calibrator = self.calibrators[self.current_best_method]
        try:
            calibrated_predictions = [calibrator.calibrate(p) for p in predictions]
        except:
            calibrated_predictions = predictions

        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(calibrated_predictions, actuals, n_bins)

        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(calibrated_predictions, actuals, n_bins)

        # Brier Score
        brier = self._calculate_brier_score(calibrated_predictions, actuals)

        # Log Loss
        log_loss = self._calculate_log_loss(calibrated_predictions, actuals)

        # Reliability (how close predicted probabilities are to actual frequencies)
        reliability = 1 - ece  # Simplified

        # Sharpness (spread of predicted probabilities)
        sharpness = float(np.std(calibrated_predictions))

        # Bins information
        bins_info = self._create_calibration_bins(calibrated_predictions, actuals, n_bins)

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier,
            log_loss=log_loss,
            reliability_score=reliability,
            sharpness_score=sharpness,
            calibration_bins=bins_info,
            sample_count=len(self.feedback_history)
        )

    def _calculate_ece(self,
                      predictions: List[float],
                      actuals: List[bool],
                      n_bins: int = 10) -> float:
        """Calcula Expected Calibration Error"""

        predictions = np.array(predictions)
        actuals = np.array(actuals, dtype=float)

        # Crear bins uniformes
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        total_samples = len(predictions)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Muestras en este bin
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = actuals[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()

                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    def _calculate_mce(self,
                      predictions: List[float],
                      actuals: List[bool],
                      n_bins: int = 10) -> float:
        """Calcula Maximum Calibration Error"""

        predictions = np.array(predictions)
        actuals = np.array(actuals, dtype=float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_error = 0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = actuals[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()

                error = abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)

        return float(max_error)

    def _calculate_brier_score(self,
                              predictions: List[float],
                              actuals: List[bool]) -> float:
        """Calcula Brier Score"""
        predictions = np.array(predictions)
        actuals = np.array(actuals, dtype=float)

        return float(np.mean((predictions - actuals) ** 2))

    def _calculate_log_loss(self,
                           predictions: List[float],
                           actuals: List[bool]) -> float:
        """Calcula Log Loss"""
        predictions = np.array(predictions)
        actuals = np.array(actuals, dtype=float)

        # Clip para evitar log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)

        return float(-np.mean(actuals * np.log(predictions) +
                             (1 - actuals) * np.log(1 - predictions)))

    def _create_calibration_bins(self,
                                predictions: List[float],
                                actuals: List[bool],
                                n_bins: int = 10) -> Dict[str, Any]:
        """Crea información detallada de bins de calibración"""

        predictions = np.array(predictions)
        actuals = np.array(actuals, dtype=float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins_info = {
            'bin_boundaries': bin_boundaries.tolist(),
            'bins': []
        }

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)

            bin_info = {
                'bin_id': i,
                'lower_bound': float(bin_lower),
                'upper_bound': float(bin_upper),
                'count': int(in_bin.sum()),
                'proportion': float(in_bin.mean())
            }

            if in_bin.sum() > 0:
                bin_info.update({
                    'avg_confidence': float(predictions[in_bin].mean()),
                    'accuracy': float(actuals[in_bin].mean()),
                    'calibration_error': float(abs(predictions[in_bin].mean() - actuals[in_bin].mean()))
                })
            else:
                bin_info.update({
                    'avg_confidence': 0.0,
                    'accuracy': 0.0,
                    'calibration_error': 0.0
                })

            bins_info['bins'].append(bin_info)

        return bins_info

    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema de calibración"""

        status = {
            'current_best_method': self.current_best_method.value,
            'available_methods': [m.value for m in self.methods],
            'feedback_samples': len(self.feedback_history),
            'last_recalibration': self.last_recalibration,
            'recalibration_frequency': self.recalibration_frequency,
            'calibrators_fitted': {}
        }

        # Estado de cada calibrador
        for method, calibrator in self.calibrators.items():
            status['calibrators_fitted'][method.value] = getattr(calibrator, 'is_fitted', False)

        # Métricas recientes si hay suficientes datos
        if len(self.feedback_history) >= 20:
            recent_metrics = self.calculate_calibration_metrics()
            status['recent_metrics'] = {
                'ece': recent_metrics.expected_calibration_error,
                'mce': recent_metrics.maximum_calibration_error,
                'brier_score': recent_metrics.brier_score,
                'reliability': recent_metrics.reliability_score
            }

        return status


# Funciones de utilidad
def create_calibrator(methods: List[str] = None,
                     window_size: int = 1000) -> DynamicConfidenceCalibrator:
    """Crea instancia de calibrador con configuración por defecto"""

    if methods:
        method_enums = [CalibrationMethod(m) for m in methods]
    else:
        method_enums = [
            CalibrationMethod.PLATT_SCALING,
            CalibrationMethod.TEMPERATURE_SCALING,
            CalibrationMethod.HISTOGRAM_BINNING
        ]

    return DynamicConfidenceCalibrator(method_enums, window_size)


if __name__ == "__main__":
    # Ejemplo completo de uso
    print("🎯 Dynamic Confidence Calibration - Demo")
    print("=" * 60)

    # Crear calibrador
    calibrator = create_calibrator()

    # Simular datos de entrenamiento inicial
    print("\n📚 Generando datos de calibración inicial...")
    np.random.seed(42)

    # Generar scores y resultados sintéticos con bias conocido
    n_samples = 500
    true_probs = np.random.beta(2, 2, n_samples)  # Distribución beta
    raw_scores = true_probs + 0.1 * np.random.normal(0, 1, n_samples)  # Añadir ruido
    raw_scores = np.clip(raw_scores, 0.01, 0.99)  # Mantener en rango válido

    # Generar resultados verdaderos basados en probabilidades
    actual_results = np.random.random(n_samples) < true_probs

    print(f"   Generadas {n_samples} muestras sintéticas")
    print(f"   Score promedio: {np.mean(raw_scores):.3f}")
    print(f"   Tasa de éxito real: {np.mean(actual_results):.3f}")

    # Añadir feedback inicial
    print("\n🔄 Añadiendo feedback para calibración inicial...")
    for i, (score, result) in enumerate(zip(raw_scores, actual_results)):
        calibrator.add_feedback(
            predicted_confidence=float(score),
            actual_correctness=bool(result),
            context={'sample_id': i, 'synthetic': True}
        )

        # Mostrar progreso cada 100 muestras
        if (i + 1) % 100 == 0:
            print(f"   Procesadas {i + 1}/{n_samples} muestras")

    print(f"✅ Feedback inicial completado")

    # Mostrar estado del sistema
    print("\n📊 Estado del sistema después de calibración:")
    status = calibrator.get_system_status()
    print(f"   Método actual: {status['current_best_method']}")
    print(f"   Muestras de feedback: {status['feedback_samples']}")
    print(f"   Calibradores entrenados: {status['calibrators_fitted']}")

    if 'recent_metrics' in status:
        metrics = status['recent_metrics']
        print(f"   ECE: {metrics['ece']:.4f}")
        print(f"   Brier Score: {metrics['brier_score']:.4f}")
        print(f"   Reliability: {metrics['reliability']:.4f}")

    # Probar calibración en nuevos scores
    print("\n🧪 Probando calibración en nuevos scores:")
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]

    for raw_score in test_scores:
        calibrated = calibrator.calibrate_confidence(
            raw_score=raw_score,
            context={'test': True}
        )

        print(f"   Raw: {raw_score:.2f} -> Calibrated: {calibrated.calibrated_score:.3f}")
        print(f"      Level: {calibrated.confidence_level.value}")
        print(f"      Method: {calibrated.calibration_method.value}")
        print(f"      Uncertainty: {calibrated.uncertainty_estimate:.3f}")

    # Calcular métricas detalladas
    print("\n📈 Métricas de calibración detalladas:")
    detailed_metrics = calibrator.calculate_calibration_metrics()

    print(f"   Expected Calibration Error (ECE): {detailed_metrics.expected_calibration_error:.4f}")
    print(f"   Maximum Calibration Error (MCE): {detailed_metrics.maximum_calibration_error:.4f}")
    print(f"   Brier Score: {detailed_metrics.brier_score:.4f}")
    print(f"   Log Loss: {detailed_metrics.log_loss:.4f}")
    print(f"   Reliability Score: {detailed_metrics.reliability_score:.4f}")
    print(f"   Sharpness Score: {detailed_metrics.sharpness_score:.4f}")

    # Mostrar información de bins de calibración
    print(f"\n🗂️  Información de bins de calibración:")
    for bin_info in detailed_metrics.calibration_bins['bins'][:5]:  # Primeros 5 bins
        print(f"   Bin [{bin_info['lower_bound']:.1f}-{bin_info['upper_bound']:.1f}]: "
              f"{bin_info['count']} muestras, "
              f"Confianza: {bin_info['avg_confidence']:.3f}, "
              f"Precisión: {bin_info['accuracy']:.3f}, "
              f"Error: {bin_info['calibration_error']:.3f}")

    # Simular feedback adicional en tiempo real
    print(f"\n⏱️  Simulando feedback en tiempo real...")
    new_test_data = [
        (0.2, False), (0.8, True), (0.6, True), (0.9, True), (0.3, False),
        (0.7, True), (0.4, False), (0.5, True), (0.8, False), (0.9, True)
    ]

    for i, (score, result) in enumerate(new_test_data):
        # Calibrar score antes de añadir feedback
        pre_feedback = calibrator.calibrate_confidence(score)

        # Añadir feedback
        calibrator.add_feedback(score, result, {'real_time': True, 'batch': i})

        # Calibrar después del feedback
        post_feedback = calibrator.calibrate_confidence(score)

        print(f"   Score {score:.1f} -> Pre: {pre_feedback.calibrated_score:.3f}, "
              f"Post: {post_feedback.calibrated_score:.3f} "
              f"({'✓' if result else '✗'})")

    # Comparar métrica antes y después
    print(f"\n📊 Comparación de métricas después de feedback adicional:")
    final_metrics = calibrator.calculate_calibration_metrics()

    print(f"   ECE: {detailed_metrics.expected_calibration_error:.4f} -> "
          f"{final_metrics.expected_calibration_error:.4f} "
          f"({'↓' if final_metrics.expected_calibration_error < detailed_metrics.expected_calibration_error else '↑'})")

    print(f"   Brier: {detailed_metrics.brier_score:.4f} -> "
          f"{final_metrics.brier_score:.4f} "
          f"({'↓' if final_metrics.brier_score < detailed_metrics.brier_score else '↑'})")

    # Prueba de diferentes métodos de calibración
    print(f"\n🔬 Comparando métodos de calibración:")
    test_score = 0.6

    methods_to_test = [
        CalibrationMethod.PLATT_SCALING,
        CalibrationMethod.TEMPERATURE_SCALING,
        CalibrationMethod.HISTOGRAM_BINNING
    ]

    for method in methods_to_test:
        try:
            result = calibrator.calibrate_confidence(
                test_score,
                method=method,
                context={'method_test': True}
            )
            print(f"   {method.value}: {test_score:.2f} -> {result.calibrated_score:.3f} "
                  f"(uncertainty: {result.uncertainty_estimate:.3f})")
        except Exception as e:
            print(f"   {method.value}: Error - {e}")

    # Estado final del sistema
    print(f"\n🎯 Estado final del sistema:")
    final_status = calibrator.get_system_status()
    print(f"   Método óptimo: {final_status['current_best_method']}")
    print(f"   Total de feedback: {final_status['feedback_samples']}")
    print(f"   Recalibraciones: {final_status['last_recalibration'] // calibrator.recalibration_frequency}")

    print(f"\n🎉 Dynamic Confidence Calibration Demo Completado!")
    print(f"="*60)
