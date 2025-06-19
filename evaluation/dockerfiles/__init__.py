from c import (
    _DOCKERFILE_BASE_C,
    _DOCKERFILE_INSTANCE_C,
)

_DOCKERFILE_BASE = {
    "c": _DOCKERFILE_BASE_C,
}

_DOCKERFILE_INSTANCE = {
    "c": _DOCKERFILE_INSTANCE_C,
}


def get_dockerfile_base(platform, arch, language, **kwargs):
    if arch == "arm64":
        conda_arch = "aarch64"
    else:
        conda_arch = arch

    return _DOCKERFILE_BASE[language].format(
        platform=platform, conda_arch=conda_arch, **kwargs
    )


def get_dockerfile_instance(platform, language, env_image_name):
    return _DOCKERFILE_INSTANCE[language].format(
        platform=platform, env_image_name=env_image_name
    )


__all__ = [
    "get_dockerfile_base",
    "get_dockerfile_instance",
]
