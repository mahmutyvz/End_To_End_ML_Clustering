import os
from joblib import dump
from sklearn.pipeline import Pipeline
from src.models.metrics import calculate_silhouette_score

class Trainer:
    def __init__(self,data,model,scaler,saved_model_path):
        """
            The Trainer class is designed to facilitate the training and saving of machine learning models for clustering tasks.
             It uses a specified data set, model, scaler, and saved model path to perform training and evaluation.

            Parameters:
            data: A pandas DataFrame or numpy array containing the data for training.

            model: An instance of a scikit-learn clustering model that is used for clustering tasks.

            scaler: A scikit-learn transformer for preprocessing the data, usually used for feature scaling.

            saved_model_path: A string representing the directory path where trained model files will be saved.

            Returns:
                None
        """
        self.data = data
        self.model = model
        self.scaler = scaler
        self.saved_model_path = saved_model_path
    def train(self):
        """
        The train method is responsible for training a clustering model using the provided data, scaler, and saving the
        trained model. It also calculates the Silhouette Score to evaluate the quality of the generated clusters.

        Returns:
            The method returns both the Silhouette Score and the predicted cluster labels.
            The Silhouette Score provides insight into the quality of the clustering, and the labels can be used
            for further analysis.
        """
        directory = os.path.join(self.saved_model_path)
        if not os.path.exists(os.path.join(directory,str(f'{type(self.model).__name__}'))):
            os.makedirs(os.path.join(directory, str(f'{type(self.model).__name__}')), exist_ok=True)
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)
        ])
        labels = pipeline.fit_predict(self.data)
        scores = calculate_silhouette_score(self.data, labels)
        model_name = os.path.join(f'{directory}/{str(type(self.model).__name__)}/{str(type(self.model).__name__)}.gz')
        dump(self.model, model_name, compress=('gzip', 3))
        return scores,labels