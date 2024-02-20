# ref doc: https://docs.docker.com/language/golang/build-images/
# WITH SUBFOLDER
# ======================
#  GO FIRST STAGE
# ======================
FROM golang:latest as builder
WORKDIR /usr/src/app
COPY go.mod go.sum ./
ENV GO111MODULE="on" \
  GOARCH="amd64" \
  GOOS="linux" \
  CGO_ENABLED="0" \
  GOPROXY="https://mirrors.aliyun.com/goproxy/,direct"
RUN go mod download
COPY *.go ./
RUN go build -o /main

# ======================
#  GO FINAL STAGE
# ======================

# wtyfft image
FROM scratch
COPY --from=builder /main /main
EXPOSE 8080
ENTRYPOINT ["/main"]

# wtytest image
# FROM ubuntu:18.04
# COPY --from=builder /main /main
# # ref: https://blog.csdn.net/zmzwll1314/article/details/100557519
# RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
# RUN  apt-get clean
# RUN apt-get update \
#   && apt-get install -y \
#   build-essential \
#   wget \
#   curl \
#   ca-certificates \
#   xdg-utils \
#   python3 \
#   g++ \
#   make \
#   vim 
# EXPOSE 8080
# ENTRYPOINT ["/main"]