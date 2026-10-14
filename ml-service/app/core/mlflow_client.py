"""
app/core/mlflow_client.py
─────────────────────────
Cliente resiliente para interactuar con el Servidor MLflow Tracking y Model Registry.
Maneja la verificación de conexión, registro de experimentos, consulta de versiones y
promoción de modelos entre etapas (Candidate -> Production).
"""

from __future__ import annotations

import os
from typing import Any
import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import requests


class MLflowManager:
    """Administrador centralizado de la conexión y operaciones con MLflow."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        self._client: MlflowClient | None = None

    @property
    def client(self) -> MlflowClient:
        """Instancia lazy de MlflowClient."""
        if self._client is None:
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    def check_connection(self, timeout_sec: float = 3.0) -> bool:
        """
        Verifica si el servidor MLflow está accesible.
        Retorna True si responde HTTP 200, False si no está disponible.
        """
        try:
            url = self.tracking_uri.rstrip("/")
            # Si es URL HTTP/HTTPS
            if url.startswith("http://") or url.startswith("https://"):
                resp = requests.get(f"{url}/health", timeout=timeout_sec)
                if resp.status_code == 200:
                    return True
                # Algunos servidores MLflow responden en la raíz
                resp = requests.get(url, timeout=timeout_sec)
                return resp.status_code == 200
            # Si es conexión directa a base de datos
            self.client.search_experiments(max_results=1)
            return True
        except Exception as e:
            print(f"[MLflowManager] Servidor no accesible en {self.tracking_uri}: {e}")
            return False

    def get_or_create_experiment(self, experiment_name: str) -> str:
        """
        Obtiene el experiment_id existente o crea uno nuevo si no existe.
        """
        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is not None:
            return exp.experiment_id
        try:
            return self.client.create_experiment(experiment_name)
        except MlflowException:
            # En caso de concurrencia
            exp = self.client.get_experiment_by_name(experiment_name)
            if exp:
                return exp.experiment_id
            raise

    def get_production_version(self, model_name: str) -> ModelVersion | None:
        """
        Obtiene la versión del modelo que actualmente se encuentra en stage 'Production'
        o que posee el alias 'champion'.
        """
        try:
            # 1. Intentar por alias moderno (MLflow 2.8+)
            try:
                model_ver = self.client.get_model_version_by_alias(model_name, "champion")
                if model_ver:
                    return model_ver
            except Exception:
                pass

            # 2. Buscar por stage clásico 'Production'
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return versions[0]
            return None
        except Exception as e:
            print(f"[MLflowManager] Error consultando versión de producción de '{model_name}': {e}")
            return None

    def get_latest_candidate_versions(self, model_name: str, limit: int = 5) -> list[ModelVersion]:
        """
        Retorna las versiones candidatas activas (excluyendo 'Production' y 'Archived')
        para evaluación y benchmarking.
        """
        try:
            # Buscar versiones del modelo registradas ordenadas descendentemente
            filter_str = f"name = '{model_name}'"
            all_versions = self.client.search_model_versions(
                filter_string=filter_str,
                order_by=["version_number DESC"],
                max_results=max(limit * 3, 20),
            )
            # Filtrar estrictamente las candidatas activas (no Production ni Archived)
            candidates = [
                v for v in all_versions
                if v.current_stage not in ("Production", "Archived")
            ]
            return candidates[:limit]
        except Exception as e:
            print(f"[MLflowManager] Error buscando versiones candidatas de '{model_name}': {e}")
            return []

    def get_model_version(self, model_name: str, version: str | int) -> ModelVersion | None:
        """
        Obtiene una versión específica de un modelo en Model Registry.
        """
        try:
            v_str = str(version).strip()
            import re
            m = re.search(r"\d+", v_str)
            if m:
                v_str = m.group(0)
            return self.client.get_model_version(model_name, v_str)
        except Exception as e:
            print(f"[MLflowManager] Error consultando versión {version} de '{model_name}': {e}")
            return None

    def get_latest_archived_version(self, model_name: str) -> ModelVersion | None:
        """
        Retorna la versión archivada más reciente para operaciones de Rollback automático.
        """
        try:
            filter_str = f"name = '{model_name}'"
            all_versions = self.client.search_model_versions(
                filter_string=filter_str,
                order_by=["version_number DESC"],
            )
            for v in all_versions:
                if v.current_stage == "Archived":
                    return v
            return None
        except Exception as e:
            print(f"[MLflowManager] Error buscando versión archivada de '{model_name}': {e}")
            return None

    def get_run_metrics_and_params(self, run_id: str) -> dict[str, Any]:
        """
        Obtiene las métricas, parámetros y tags de una corrida específica.
        """
        run = self.client.get_run(run_id)
        return {
            "run_id": run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "status": run.info.status,
        }

    def promote_version_to_production(
        self,
        model_name: str,
        version: int | str,
        archive_existing: bool = True,
        motivo: str = "",
    ) -> ModelVersion:
        """
        Promueve una versión candidata a 'Production', asigna el alias 'champion',
        archiva la versión previa y registra los tags de auditoría.
        """
        version_str = str(version).strip()
        import re
        m = re.search(r"\d+", version_str)
        if m:
            version_str = m.group(0)

        print(f"[MLflowManager] Promoviendo {model_name} v{version_str} a Production...")

        # 1. Asignar stage Production en Model Registry
        promoted = self.client.transition_model_version_stage(
            name=model_name,
            version=version_str,
            stage="Production",
            archive_existing_versions=archive_existing,
        )

        # 2. Asignar alias champion
        try:
            self.client.set_registered_model_alias(model_name, "champion", version_str)
        except Exception:
            pass

        # 3. Registrar tag de auditoría con el motivo
        if motivo:
            self.client.set_model_version_tag(
                name=model_name,
                version=version_str,
                key="promotion_reason",
                value=motivo,
            )

        print(f"[MLflowManager] [OK] {model_name} v{version_str} promovido exitosamente a Production.")
        return promoted


# Instancia singleton del cliente de MLflow
mlflow_manager = MLflowManager()
