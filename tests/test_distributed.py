"""GPU-count based training launch selection tests."""

import pytest

from sgmse.utils.distributed import resolve_training_launch


def _resolve(gpus, device="", environment=None, available=True, count=4):
    return resolve_training_launch(
        gpus,
        device,
        environ={} if environment is None else environment,
        cuda_available=available,
        cuda_device_count=count,
    )


def test_zero_gpus_selects_cpu() -> None:
    launch = _resolve(0)
    assert (launch.mode, launch.device, launch.spawn) == (
        "cpu",
        "cpu",
        False,
    )


def test_one_gpu_selects_single_process_cuda() -> None:
    launch = _resolve(1)
    assert (launch.mode, launch.num_gpus, launch.spawn) == (
        "single_gpu",
        1,
        False,
    )


def test_multiple_gpus_select_internal_ddp_spawn() -> None:
    launch = _resolve(4)
    assert (launch.mode, launch.num_gpus, launch.spawn) == (
        "internal_ddp",
        4,
        True,
    )


def test_torchrun_environment_is_not_spawned_again() -> None:
    launch = _resolve(
        2,
        environment={"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"},
        count=2,
    )
    assert launch.mode == "external_ddp"
    assert launch.spawn is False


def test_cpu_torchrun_remains_supported_without_cuda() -> None:
    launch = _resolve(
        None,
        environment={"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"},
        available=False,
        count=0,
    )
    assert (launch.mode, launch.device, launch.spawn) == (
        "external_ddp",
        "cpu",
        False,
    )


def test_torchrun_gpu_count_must_match_local_world_size() -> None:
    with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE"):
        _resolve(
            4,
            environment={"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2"},
            count=4,
        )


def test_request_cannot_exceed_visible_gpu_count() -> None:
    with pytest.raises(ValueError, match="only 2 are visible"):
        _resolve(3, count=2)


def test_cuda_request_fails_cleanly_without_cuda() -> None:
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _resolve(1, available=False, count=0)


def test_conflicting_device_and_gpu_count_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve(2, device="cpu")
