FROM rockylinux:9.3

RUN dnf check-update || true
RUN dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
RUN dnf install docker-ce docker-ce-cli containerd.io
RUN systemctl start docker

RUN dnf -y install \
    python3 \
    python3-devel \
    python3-pip \
    git \
    gcc \
    gcc-c++ \
    clang
