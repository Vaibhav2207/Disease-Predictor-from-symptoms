import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from joblib import dump, load


class DiseasePrediction:

    
    def __init__(self, model_name=None):
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")
            # Handle exceptions appropriately

        self.verbose = self.config['verbose']
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        self.model_name = model_name
        self.model_save_path = self.config['model_save_path']

    def _load_train_dataset(self):
        # Load and preprocess training data
        # Example:
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']
        return train_features, train_labels, df_train

    def _load_test_dataset(self):
        # Load and preprocess test data
        # Example:
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']
        return test_features, test_labels, df_test

    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Plot feature correlation heatmap
        # Example:
        corr = data_frame.corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')

    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                         test_size=self.config['dataset']['validation_size'],
                                                         random_state=self.config['random_state'])
        return X_train, y_train, X_val, y_val

    def select_model(self):
        if self.model_name == 'mnb':
            return MultinomialNB()
        elif self.model_name == 'decision_tree':
            return DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            return RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            return GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                              criterion=self.config['model']['gradient_boost']['criterion'])
        else:
            # Handle unrecognized model names
            print("Invalid model name provided")
            return None  # Or handle this case appropriately

    def train_model(self):
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        classifier.fit(X_train, y_train)
        self.evaluate_model(classifier, X_val, y_val)
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))

    def evaluate_model(self, classifier, X_val, y_val):
        confidence = classifier.score(X_val, y_val)
        y_pred = classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        conf_mat = confusion_matrix(y_val, y_pred)
        clf_report = classification_report(y_val, y_pred)
        score = cross_val_score(classifier, X_val, y_val, cv=3)
        print('\nTraining Accuracy: ', confidence)
        print('\nValidation Prediction: ', y_pred)
        print('\nValidation Accuracy: ', accuracy)
        print('\nValidation Confusion Matrix: \n', conf_mat)
        print('\nCross Validation Score: \n', score)
        print('\nClassification Report: \n', clf_report)

    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")
        if test_data is not None:
            result = clf.predict(test_data)
            return result
        else:
            result = clf.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, result)
        clf_report = classification_report(self.test_labels, result)
        return accuracy, clf_report


if __name__ == "__main__":
    current_model_name = 'random_forest'
    dp = DiseasePrediction(model_name=current_model_name)
    dp.train_model()
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    print("Model Test Accuracy: ", test_accuracy)
    print("Test Data Classification Report: \n", classification_report)