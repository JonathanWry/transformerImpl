from utils.loader_factory import get_loader_and_dataloaders

def test_dataloader(dataset_name, batch_size=64, max_len=256):
    """
    Tests the DataLoader for the specified dataset.
    Args:
        dataset_name (str): Name of the dataset to test (e.g., 'multi30k_en_de').
        batch_size (int): Batch size for the DataLoader.
        max_len (int): Maximum sequence length.
    """
    print(f"Testing DataLoader for dataset: {dataset_name}")
    try:
        loader, train_loader, valid_loader, test_loader = get_loader_and_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_len=max_len
        )
        print("Sample train batch:")
        print(next(iter(train_loader)))

        # Test Train DataLoader
        for batch in train_loader:
            print("Train Batch Example:")
            print("Input IDs:", batch["input_ids"].shape)
            print("Labels:", batch["labels"].shape)
            break

        # Test Validation DataLoader
        for batch in valid_loader:
            print("Validation Batch Example:")
            print("Input IDs:", batch["input_ids"].shape)
            print("Labels:", batch["labels"].shape)
            break

        # Test Test DataLoader
        for batch in test_loader:
            print("Test Batch Example:")
            print("Input IDs:", batch["input_ids"].shape)
            print("Labels:", batch["labels"].shape)
            break

        print(f"DataLoader for dataset {dataset_name} tested successfully!")

    except Exception as e:
        print(f"Error testing DataLoader for {dataset_name}: {e}")


# Example Usage
if __name__ == "__main__":
    test_dataloader(dataset_name="multi30k_en_de")
    test_dataloader(dataset_name="multi30k_en_fr")
    test_dataloader(dataset_name="wikitext")


