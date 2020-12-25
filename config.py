hp = {
    'net_type': "DenseNet",
    'train_data_path': "cifar-10-batches-py/",
    'output_dir': "./",

    # model settings
    'num_dense_blocks': 4,
    'num_dense_layers': [6, 12, 24, 16],
    'growth_rate': 12,

    # train settings
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "num_workers": 4,
}