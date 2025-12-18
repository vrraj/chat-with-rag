# 🗄️ Qdrant Operations & Management

This document provides a specialized reference for managing the Qdrant vector database within this project. It covers high-level container management, developer utilities, and granular data manipulation using the CLI.

## 🚀 Quick Start (Makefile)

The following make targets simplify the most common database lifecycle tasks:

| Command | Action |
|---------|--------|
| `make start-qdrant` | Starts only the Qdrant container in detached mode. |
| `make stop-qdrant` | Stops and removes the Qdrant container and associated resources. |
| `make seed` | Ingests the default sample data into the active collection. |
| `make qdrant-info` | Displays status, dimensions, and vector counts for a collection (e.g., `make qdrant-info COLLECTION=document_index`). |
| `make qdrant-backup` | Creates a compressed archive (.tar.gz) of the qdrant_storage/ directory. |
| `make qdrant-logs` | Streams live logs from the database container. |

## 🛠️ Qdrant Operations CLI (qdrant_ops.py)

For advanced maintenance and inspection, use the dedicated Python utility. This script is designed to run within the project's virtual environment.

### Configuration

The script defaults to the host, port, and collection name defined in `backend/core/config.py`.

- **Default Host**: localhost
- **Default Port**: 6333
- **Default Collection**: document_index

### Common Commands

#### 1. Data Inspection

**List Document Titles**: Displays unique URLs and their titles currently in the index (limited to 30 chars).

```bash
python qdrant_scripts/qdrant_ops.py list-titles --limit 50
```

**Count Chunks by URL**: Returns the number of individual text fragments indexed for a specific base URL.

```bash
python qdrant_scripts/qdrant_ops.py count-chunks --base-url "https://example.com"
```

**Introspect Schema**: Lists all payload field names available in the first point of the collection.

```bash
python qdrant_scripts/qdrant_ops.py list-fields
```

#### 2. Data Export & Backups

**Export to JSONL**: Saves all points (vectors + payloads) to a JSONL file in the `data/` directory. This is useful for re-seeding other environments.

```bash
python qdrant_scripts/qdrant_ops.py export -f custom-backup.jsonl
```

#### 3. Deletion & Maintenance

**Delete by Payload**: Removes points where a specific field matches a value (e.g., deleting all chunks from one source).

```bash
python qdrant_scripts/qdrant_ops.py delete --field source --value wikipedia
```

**Truncate Collection**: Deletes all points while preserving the collection's configuration (distance metric, vector size, etc.). This requires interactive confirmation.

```bash
python qdrant_scripts/qdrant_ops.py truncate
```

## 🏗️ Storage Architecture

### Persistence

By default, all data is stored in the local `qdrant_storage/` directory. This directory is mapped to the Qdrant container, ensuring that your index survives container restarts or system reboots.

### Indexing Behavior

- **Similarity Metric**: Defaults are managed by the `backend/core/config.py` settings.
- **Payloads**: The system stores rich metadata (source URLs, section headings, titles) alongside the vectors to support traceability and reranking during the chat orchestration phase.

## 🔍 Troubleshooting Tips

- **Check Collection Existence**: Run `make qdrant-collections` to see if your intended collection has been initialized.
- **Check Payload Indexes**: Run `make qdrant-indexes` to verify which fields are currently indexed. This is critical for high-performance filtering during retrieval.
- **Empty Results**: If retrieval returns no results, use `list-fields` to ensure the metadata keys (like `base_url_lower`) match what the retrieval pipeline is searching for.
