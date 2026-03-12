[← Back to Chat with RAG Home](https://vrraj.github.io/chat-with-rag)

# 📄 Data Attribution & Licensing

> **About this document**
>
> This page provides **data attribution and licensing information** for the *Chat-with-RAG* system, including source credits and usage rights for sample data.
>
> **Note:** If you landed here directly (for example from documentation hosting or search), start with the repository **[README](https://github.com/vrraj/chat-with-rag/blob/main/README.md)** to see how to run the system locally and try the interactive demo.

This project utilizes a sample knowledge base to demonstrate its RAG capabilities. We believe in transparent data sourcing and respect for open-content creators.

## 📚 Wikipedia Content
The optional seed dataset (`data/docs-index-seed.jsonl`) contains content derived from [Wikipedia](https://www.wikipedia.org/). 

* **Source:** 55 specific articles across diverse topics.
* **Attribution:** Individual source URLs and author attribution metadata are preserved within the metadata fields of the `.jsonl` file and the Qdrant vector payload.
* **License:** This content is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

## ⚖️ Derivative Work & License Notice
In accordance with the **Share-Alike (SA)** provision of the CC BY-SA 4.0 license:
1.  **Modifications:** The original text has been transformed through cleaning, semantic segmentation (chunking), and conversion into vector embeddings for use within this RAG pipeline.
2.  **Dataset License:** The resulting derivative dataset (`data/docs-index-seed.jsonl`) is hereby released under the same **CC BY-SA 4.0** license.
3.  **Disclaimer:** This project is an independent educational tool and is not affiliated with, sponsored by, or endorsed by the Wikimedia Foundation.

---
*For questions regarding the data processing pipeline, please refer to the [Technical Overview](./technical-overview.md).*