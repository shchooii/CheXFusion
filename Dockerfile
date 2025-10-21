# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 continuumio/miniconda3:latest

ARG ENV_NAME=cxr13

# 패킹한 conda env 복사 후 언팩
COPY cxr13.tar.gz /tmp/cxr13.tar.gz
RUN mkdir -p /opt/conda/envs/${ENV_NAME} \
 && tar -xzf /tmp/cxr13.tar.gz -C /opt/conda/envs/${ENV_NAME} \
 && rm /tmp/cxr13.tar.gz \
 && /opt/conda/envs/${ENV_NAME}/bin/conda-unpack

# 이후 명령은 cxr13 환경에서 실행
SHELL ["conda", "run", "-n", "cxr13", "/bin/bash", "-c"]

WORKDIR /workspace
CMD ["python", "-V"]
EOF
