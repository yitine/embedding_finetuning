"""測試真實 HuggingFace 下載（需連網）"""


def test_nfcorpus_download():
    from datasets import load_dataset

    # 測試下載（會快取）
    ds = load_dataset("mteb/nfcorpus", split="train")
    print(f"NFCorpus train: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")
    print(f"First sample: {ds[0]}")

    assert len(ds) > 0
    assert (
        "query" in ds.column_names
        or "text" in ds.column_names
        or {"query-id", "corpus-id", "score"}.issubset(ds.column_names)
    )


def test_scifact_download():
    from datasets import load_dataset

    ds = load_dataset("mteb/scifact", split="test")
    print(f"SciFact test: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    assert len(ds) > 0


if __name__ == "__main__":
    test_nfcorpus_download()
    test_scifact_download()
    print("所有下載測試通過")
