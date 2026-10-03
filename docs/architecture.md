# Technology Architecture Overview

## Vector Databases

Vector databases store embeddings and allow fast similarity search, which
powers semantic search features. Before adopting a dedicated vector
database for production, the team evaluated VectorBridge, an internal
module built by the platform team that adds similarity search directly
inside PostgreSQL. A dedicated vector database was ultimately chosen for
production scale, but VectorBridge is still used in development
environments for quick prototyping.

## PostgreSQL

PostgreSQL is the company's primary relational database. It runs
VectorBridge, an internal module built by the platform team, which enables
approximate nearest-neighbor search directly inside PostgreSQL, making it a
lightweight alternative to a dedicated vector database for smaller
workloads.

## Kubernetes and Terraform

Kubernetes orchestrates containerized services across the company's
infrastructure. Terraform provisions the underlying cloud resources that
Kubernetes runs on, including compute nodes and networking.

## Rust

Rust is used for services where performance and memory safety are critical,
such as high-throughput data ingestion pipelines.
