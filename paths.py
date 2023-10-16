class Path:
    """
    The Path class provides paths to various files and directories within a specified root directory for a project
    related to E-commerce consumer behavior analysis.

    Attributes:

    root: A string representing the root directory for the project. This directory contains subdirectories and files
    related to the project.

    train_path: A string representing the path to the raw E-commerce consumer behavior dataset file. This dataset is
    located within the raw subdirectory of the project's root directory.

    cleaned_train_path: A string representing the path to the preprocessed and cleaned version of the raw dataset. This
    cleaned dataset is located within the preprocessed subdirectory of the project's root directory.

    models_path: A string representing the path to the directory where machine learning models related to the project
    are stored. This directory is located within the project's root directory.
    """
    root = 'C:/Users/MahmutYAVUZ/Desktop/Software/Python/kaggle/clustering'
    train_path = root + '/data/raw/ECommerce_consumer behaviour.csv'
    cleaned_train_path = root + '/data/preprocessed/cleaned_raw.csv'
    models_path = root + "/models/"
