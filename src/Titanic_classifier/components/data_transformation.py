import os
from Titanic_classifier.constant import *
from Titanic_classifier.utils.common import read_yaml,create_directories
from Titanic_classifier import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import urllib.request as request
import zipfile
from Titanic_classifier.entity import DataTransformationConfig
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_data(self,data):

        data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
        data['Title'] = data['Title'].replace(['Capt','Col', 'Master', 'Rev', 'Dr','Countess','Jonkheer','Sir','Major','Don'], 'Honored')
        data['Title'] = data['Title'].replace([ 'Ms','Dona','Mme'], 'Mrs')
        data['Title'] = data['Title'].replace(['Mlle','Lady'], 'Miss')
        
        age_ref = data.groupby('Title').Age.mean()
        data['Age'] = data.apply(lambda r: r.Age if pd.notnull(r.Age) else age_ref[r.Title] , axis=1)
        data['AgeBand'] = pd.cut(data['Age'], 5, labels=range(5)).astype(int)
    
        data['Alone'] = (data['SibSp'] + data['Parch'] == 0).astype(int)
        
        data['Cabin_letter'] = data['Cabin'].str[0]
        label_encoder = LabelEncoder()
        data['Cabin_brand'] = label_encoder.fit_transform(data['Cabin_letter'])

        data['Room'] = (data['Cabin']
                        .str.slice(1,5).str.extract('([0-9]+)', expand=False)
                        .fillna(0)
                        .astype(int))
        data['RoomBand'] = 0
        data.loc[(data.Room > 0) & (data.Room <= 20), 'RoomBand'] = 1
        data.loc[(data.Room > 20) & (data.Room <= 40), 'RoomBand'] = 2
        data.loc[(data.Room > 40) & (data.Room <= 80), 'RoomBand'] = 3
        data.loc[data.Room > 80, 'RoomBand'] = 4
        

        data['Embarked'] = data['Embarked'].fillna('S')
        

        data['FamilySize'] = (data['SibSp'] + data['Parch']).astype(int)
        data['FamilySizeBand'] = 0
        data.loc[(data.FamilySize == 1), 'FamilySizeBand'] = 1
        data.loc[(data.FamilySize == 2), 'FamilySizeBand'] = 2
        data.loc[(data.FamilySize == 3), 'FamilySizeBand'] = 3
        data.loc[data.FamilySize > 3, 'FamilySizeBand'] = 3

        cols = [
            'Pclass',
            'Sex',
            'AgeBand',
            'SibSp',
            'Parch',
            'Fare',
            'Title',
            'Alone',
            'Cabin_brand',
            'RoomBand',
            'FamilySizeBand',
            'Survived'  
        ]

        data_prep = data[cols]
        X = data_prep.drop(columns=['Survived'], axis=1)
        y = data_prep['Survived']

        one_hot_features = [
            # 'Pclass',
            'Sex',
            'AgeBand',
            # 'SibSp',
            # 'Parch',
            # 'Fare',
            'Title',
            'Alone',
            'Cabin_brand',
            'RoomBand',
            'FamilySizeBand',
            # 'Survived'
        ]

        oh_transformer = OneHotEncoder()
        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, one_hot_features),    
            ],
            remainder='passthrough'  
        )
        X_encoded = preprocessor.fit_transform(X)

        encoded_column_names = preprocessor.named_transformers_['OneHotEncoder'].get_feature_names_out(one_hot_features)

        other_columns = [col for col in X.columns if col not in one_hot_features]


        final_column_names = list(encoded_column_names) + other_columns

        X_encoded_df = pd.DataFrame(X_encoded, columns=final_column_names)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    

    def convert(self):
        dataset_titanic = pd.read_csv(self.config.data_path)
        X_train, X_test, y_train, y_test = self.preprocess_data(dataset_titanic)
        X_train.to_csv(os.path.join(self.config.root_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.config.root_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)
