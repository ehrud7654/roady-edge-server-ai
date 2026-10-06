from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Callable

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from .analyzer import ServerDamageAnalyzer
from .config import AppSettings
from .schemas import (
    AnalysisInputMetadata,
    DamageAnalysisResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .service import (
    AnalysisInputError,
    DamageAnalysisService,
    InputImage,
    read_and_verify_sha256,
)


log = logging.getLogger(__name__)
AnalyzerFactory = Callable[[str], ServerDamageAnalyzer]


def create_app(
    settings: AppSettings | None = None,
    *,
    analyzer_factory: AnalyzerFactory = ServerDamageAnalyzer,
) -> FastAPI:
    app_settings = settings or AppSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        model_sha256 = read_and_verify_sha256(
            app_settings.model_path,
            app_settings.model_sha256_path,
            verify=app_settings.verify_model_hash,
        )
        analyzer = analyzer_factory(str(app_settings.model_path))
        service = DamageAnalysisService(
            analyzer,
            model_path=app_settings.model_path,
            model_sha256=model_sha256,
            imgsz=app_settings.imgsz,
            device=app_settings.device,
            score_minor_max_pixels=app_settings.score_minor_max_pixels,
            score_moderate_max_pixels=app_settings.score_moderate_max_pixels,
            score_max_pixels=app_settings.score_max_pixels,
        )
        if app_settings.warmup_enabled:
            await run_in_threadpool(service.warmup)
        app.state.service = service
        app.state.inference_semaphore = asyncio.Semaphore(
            app_settings.inference_concurrency
        )
        app.state.ready = True
        log.info(
            "AI model is ready. model=%s device=%s imgsz=%s sha256=%s",
            app_settings.model_path.name,
            app_settings.device,
            app_settings.imgsz,
            model_sha256,
        )
        try:
            yield
        finally:
            app.state.ready = False

    app = FastAPI(
        title="ROADY AI Server",
        version="2.0.0",
        lifespan=lifespan,
    )

    @app.get("/health/live", response_model=HealthResponse)
    async def health_live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/health/ready", response_model=HealthResponse)
    async def health_ready() -> HealthResponse:
        if not getattr(app.state, "ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI model is not ready.",
            )
        return HealthResponse(status="ready")

    @app.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        if not getattr(app.state, "ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI model is not ready.",
            )
        service: DamageAnalysisService = app.state.service
        analyzer = service.analyzer
        class_names = getattr(analyzer, "class_names", {})
        classes = (
            {str(key): str(value) for key, value in class_names.items()}
            if isinstance(class_names, dict)
            else {str(index): str(value) for index, value in enumerate(class_names)}
        )
        return ModelInfoResponse(
            name=(
                service.model_path.stem
                if classes
                else "yolo26s-seg-server-v1"
            ),
            weights=service.model_path.name,
            weights_sha256=service.model_sha256,
            classes=classes,
            device=service.device,
            imgsz=service.imgsz,
        )

    @app.post("/analyze", response_model=DamageAnalysisResponse)
    async def analyze(
        damage_id: Annotated[str, Form(alias="damageId")],
        images: Annotated[list[UploadFile], File()],
        latitude: Annotated[str | None, Form()] = None,
        longitude: Annotated[str | None, Form()] = None,
        captured_at: Annotated[str | None, Form(alias="capturedAt")] = None,
        analysis_metadata: Annotated[
            str | None,
            Form(alias="analysisMetadata"),
        ] = None,
    ) -> DamageAnalysisResponse:
        del latitude, longitude, captured_at
        try:
            parsed_damage_id = int(damage_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="damageId must be a positive integer.",
            ) from exc
        if parsed_damage_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="damageId must be a positive integer.",
            )
        if not 1 <= len(images) <= app_settings.max_images:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"images must contain between 1 and {app_settings.max_images} files.",
            )

        decoded_images = await _decode_uploads(images, app_settings)
        decoded_images = _attach_input_metadata(decoded_images, analysis_metadata)
        service: DamageAnalysisService = app.state.service
        try:
            async with app.state.inference_semaphore:
                return await run_in_threadpool(service.analyze_images, decoded_images)
        except AnalysisInputError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            log.exception("AI inference failed. damageId=%s", parsed_damage_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI inference failed.",
            ) from exc

    return app


async def _decode_uploads(
    uploads: list[UploadFile], settings: AppSettings
) -> list[InputImage]:
    decoded: list[InputImage] = []
    total_bytes = 0
    for index, upload in enumerate(uploads):
        try:
            data = await upload.read(settings.max_image_bytes + 1)
        finally:
            await upload.close()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Image index {index} is empty.",
            )
        if len(data) > settings.max_image_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image index {index} exceeds the size limit.",
            )
        total_bytes += len(data)
        if total_bytes > settings.max_request_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="The image request exceeds the total size limit.",
            )

        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Image index {index} cannot be decoded.",
            )
        decoded.append(
            InputImage(
                filename=upload.filename or f"image-{index}",
                image=image,
            )
        )
    return decoded


def _attach_input_metadata(
    images: list[InputImage],
    raw_metadata: str | None,
) -> list[InputImage]:
    if raw_metadata is None or not raw_metadata.strip():
        metadata_items: list[dict] = [{} for _ in images]
    else:
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="analysisMetadata must be valid JSON.",
            ) from exc
        if isinstance(parsed, dict):
            metadata_items = [parsed]
        elif isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            metadata_items = parsed
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="analysisMetadata must be a JSON object or an array of objects.",
            )

    if len(metadata_items) != len(images):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="analysisMetadata count must match images count.",
        )

    enriched: list[InputImage] = []
    for image, raw_item in zip(images, metadata_items):
        try:
            metadata = AnalysisInputMetadata.model_validate(raw_item)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="analysisMetadata contains invalid fields.",
            ) from exc
        values = metadata.model_dump(exclude_none=True)
        values.setdefault("analysis_roi", image.filename)
        enriched.append(
            InputImage(
                filename=image.filename,
                image=image.image,
                input_metadata=values,
            )
        )
    return enriched


app = create_app()
