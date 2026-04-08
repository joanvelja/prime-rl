import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.orchestrator import setup_rollout_inference_pool


def test_setup_rollout_inference_pool_uses_plain_client_for_external_teacher_rollout():
    async def run() -> None:
        config = SimpleNamespace(
            teacher_rollout_model=SimpleNamespace(),
            model=SimpleNamespace(renderer="auto", name="student-model"),
        )
        rollout_client_config = SimpleNamespace(base_url=["https://api.pinference.ai/api/v1"])
        logger = MagicMock()
        inference_pool = object()

        with (
            patch(
                "prime_rl.orchestrator.orchestrator.setup_inference_pool", new=AsyncMock(return_value=inference_pool)
            ),
            patch("prime_rl.orchestrator.orchestrator.create_renderer") as create_renderer_mock,
            patch("prime_rl.orchestrator.orchestrator.RenderingProxy") as proxy_mock,
            patch("prime_rl.orchestrator.orchestrator.ProxiedInferencePool") as proxied_pool_mock,
            patch("prime_rl.orchestrator.orchestrator.asyncio.create_task") as create_task_mock,
        ):
            renderer, returned_pool, proxy, proxy_server, proxy_task = await setup_rollout_inference_pool(
                config=config,
                rollout_client_config=rollout_client_config,
                rollout_model_name="teacher-model",
                tokenizer=object(),
                processor=None,
                logger=logger,
            )

        assert renderer is None
        assert returned_pool is inference_pool
        assert proxy is None
        assert proxy_server is None
        assert proxy_task is None
        create_renderer_mock.assert_not_called()
        proxy_mock.assert_not_called()
        proxied_pool_mock.assert_not_called()
        create_task_mock.assert_not_called()

    asyncio.run(run())


def test_setup_rollout_inference_pool_wraps_local_vllm_with_renderer_proxy():
    async def run() -> None:
        config = SimpleNamespace(
            teacher_rollout_model=None,
            model=SimpleNamespace(renderer="auto", name="student-model"),
        )
        rollout_client_config = SimpleNamespace(base_url=["http://localhost:8000/v1"])
        logger = MagicMock()
        renderer = object()
        upstream_pool = object()
        proxy = SimpleNamespace(app=object())
        proxy_server = SimpleNamespace(serve=MagicMock(return_value=object()))
        proxy_task = object()
        inference_pool = object()

        with (
            patch("prime_rl.orchestrator.orchestrator.create_renderer", return_value=renderer) as create_renderer_mock,
            patch("prime_rl.orchestrator.orchestrator.RenderingProxy", return_value=proxy) as proxy_cls_mock,
            patch("prime_rl.orchestrator.orchestrator.uvicorn.Config", return_value=object()) as uvicorn_config_mock,
            patch(
                "prime_rl.orchestrator.orchestrator.uvicorn.Server", return_value=proxy_server
            ) as uvicorn_server_mock,
            patch(
                "prime_rl.orchestrator.orchestrator.setup_inference_pool",
                new=AsyncMock(return_value=upstream_pool),
            ) as setup_pool_mock,
            patch(
                "prime_rl.orchestrator.orchestrator.ProxiedInferencePool",
                return_value=inference_pool,
            ) as proxied_pool_mock,
            patch(
                "prime_rl.orchestrator.orchestrator.asyncio.create_task", return_value=proxy_task
            ) as create_task_mock,
        ):
            (
                returned_renderer,
                returned_pool,
                returned_proxy,
                returned_server,
                returned_task,
            ) = await setup_rollout_inference_pool(
                config=config,
                rollout_client_config=rollout_client_config,
                rollout_model_name="student-model",
                tokenizer=object(),
                processor="processor",
                logger=logger,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        assert returned_proxy is proxy
        assert returned_server is proxy_server
        assert returned_task is proxy_task
        create_renderer_mock.assert_called_once()
        proxy_cls_mock.assert_called_once_with(
            renderer, vllm_base_url="http://localhost:8000/v1", processor="processor"
        )
        uvicorn_config_mock.assert_called_once()
        uvicorn_server_mock.assert_called_once()
        setup_pool_mock.assert_awaited_once()
        proxied_pool_mock.assert_called_once()
        create_task_mock.assert_called_once()

    asyncio.run(run())
