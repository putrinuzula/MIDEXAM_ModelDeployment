import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from xgboost import XGBClassifier
import pickle
from sklearn.preprocessing import LabelEncoder


class data_handler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
    
    def dataInfo(self):
        self.data.info()

    def drop_column(self, kolom):
        self.data.drop(columns=kolom, axis=1, inplace=True)
    
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

# ModelHandler Class
class model_handler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def split_data(self, test_size = 0.2, random_state = 42):
        #
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_data, self.output_data, test_size = test_size, random_state = random_state)
        # self.x_train = 
    
    def mean_for_fill(self, kolom):
        return np.mean(self.x_train[kolom])  

    def fill_missval(self,column,number):
        self.x_train[column].fillna(number, inplace=True)
        self.x_test[column].fillna(number, inplace=True)

    def encoding(self, kolom):
      labels = LabelEncoder()
      self.x_train[kolom] = labels.fit_transform(self.x_train[kolom])
      self.x_test[kolom] = labels.fit_transform(self.x_test[kolom])     

    # def fill_missval(self, column, mean):
    #     self.x_train[column] = self.x_train[column].fillna(mean)
    #     self.x_test[column] = self.x_train[column].fillna(mean)

    # def one_hot_encode(self, columns):
    #         self.x_train[columns] = pd.get_dummies(self.x_train, columns=[columns])
    #         self.x_test[columns] = pd.get_dummies(self.x_test, columns=[columns])
    def createModel(self, maxdepth = 4, estimators = 100):
        self.model = XGBClassifier(max_depth=maxdepth, n_estimators = estimators)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['1','2']))

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)
    
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  # Open the file in write-binary mode
            pickle.dump(self.model, file)  # Use pickle to write the model to the file


#data reading
file_path = 'data_D.csv'  
dataHandler = data_handler(file_path)
dataHandler.load_data()

#drop unrelated column
dataHandler.drop_column(['Unnamed: 0', 'id', 'CustomerId','Surname'])
dataHandler.dataInfo()

dataHandler.create_input_output('churn')
input_df = dataHandler.input_df
output_df = dataHandler.output_df


modelHandler = model_handler(input_df, output_df)
modelHandler.split_data()


replace_missval = modelHandler.mean_for_fill('CreditScore')
modelHandler.fill_missval('CreditScore', replace_missval)


modelHandler.encoding('Gender')
modelHandler.encoding('Geography')

print("XGBoost Model")
modelHandler.train_model()
print("Model Accuracy:", modelHandler.evaluate_model())
modelHandler.makePrediction()

modelHandler.createReport()
modelHandler.save_model_to_file('XGB_model.pkl') 