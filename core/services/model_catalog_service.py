import logging

from core.domain.entities import ModelInfo
from core.domain.exceptions import ModelNotFoundError
from core.ports.inbound import AuthUseCase, ModelCatalogUseCase
from core.ports.outbound import UpstreamModelPort

logger = logging.getLogger(__name__)


class ModelCatalogService(ModelCatalogUseCase):
    """Queries the upstream catalog using the current authenticated session."""

    def __init__(
        self,
        auth_use_case: AuthUseCase,
        upstream_model: UpstreamModelPort,
    ):
        self._auth = auth_use_case
        self._upstream = upstream_model

    def list_models(self) -> tuple[ModelInfo, ...]:
        token = self._auth.ensure_authenticated()
        return tuple(self._upstream.fetch_models(token))

    def get_model(self, model_id: str) -> ModelInfo:
        for model in self.list_models():
            if model.id == model_id:
                return model
        raise ModelNotFoundError(model_id)
