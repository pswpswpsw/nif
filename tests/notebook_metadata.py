"""Remove saved runtime output metadata from a compatibility notebook copy."""


def clear_execution_metadata(notebook):
    notebook["metadata"].pop("widgets", None)
    notebook["metadata"].get("colab", {}).pop("provenance", None)
    for cell in notebook["cells"]:
        metadata = cell.get("metadata", {})
        for key in ("colab", "executionInfo", "outputId"):
            metadata.pop(key, None)
