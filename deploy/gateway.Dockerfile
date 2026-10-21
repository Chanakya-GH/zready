ARG GO_VERSION=1.27
FROM --platform=$BUILDPLATFORM golang:${GO_VERSION} AS build
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src
COPY gateway/go.mod gateway/go.sum ./
RUN go mod download
COPY gateway/*.go ./
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags="-s -w" -o /out/gateway .

FROM scratch
COPY --from=build /out/gateway /gateway
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["/gateway"]
