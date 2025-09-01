# 编程环境
记录基础的编程环境遇到的问题。

## linux

- [tmux](https://github.com/tmux/tmux/wiki) 
- [9.4k] [nvtop](https://github.com/Syllo/nvtop) gpu监控

## python

- [uv](https://docs.astral.sh/uv/) 包和环境管理
- [dynaconf](https://www.dynaconf.com/) 配置管理
    - [The Twelve Factors, Config:Store config in the environment](https://12factor.net/config)
- [9.6k] [uvicorn](https://github.com/encode/uvicorn) asgi 异步的web服务框架

## docker

- [docker](https://www.docker.com/)



### 问题
描述：mac电脑上docker容器在host模式下也不能ping通目标主机，能ping通baidu。宿主机能 ping 通目标主机。容器能 ping 通宿主机。
分析：Docker for Mac通过HyperKit虚拟机运行，其网络架构与Linux不同，host模式可能未完全共享宿主机的物理网络接口。
方法：最后绕开了这个问题。在本地用ollama部署一个服务，容器内直接请求这个本地服务。