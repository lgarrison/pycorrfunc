FROM rockylinux:9.3

RUN dnf -y install \
    python3 \
    python3-devel \
    git \
    gcc \
    gcc-c++ \
    clang

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
