#%%
import pandas as pd
#%%
def load_raw_data(dataset):
    if dataset == "abalone":
        data = pd.read_csv('./data/abalone.data', header=None)
        columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Sex")
        columns.remove("Rings")
        continuous_features = columns
        categorical_features = [
            "Sex",
            "Rings"
        ]
        integer_features = []
        ClfTarget = "Rings"
        
    elif dataset == "anuran":
        data = pd.read_csv('./data/Frogs_MFCCs.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [x for x in data.columns if x.startswith("MFCCs_")]
        categorical_features = [
            'Family',
            'Genus',
            'Species'
        ]
        integer_features = []
        ClfTarget = "Species"
    
    elif dataset == "banknote":
        data = pd.read_csv('./data/data_banknote_authentication.txt', header=None)
        data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            "variance", "skewness", "curtosis", "entropy"
        ]
        categorical_features = [
            'class',
        ]
        integer_features = []
        ClfTarget = "class"
        
    elif dataset == "breast":
        data = pd.read_csv('./data/wdbc.data', header=None)
        data = data.drop(columns=[0]) # drop ID number
        columns = ["Diagnosis"]
        common_cols = [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave points",
            "symmetry",
            "fractal dimension",
        ]
        columns += [f"{x}1" for x in common_cols]
        columns += [f"{x}2" for x in common_cols]
        columns += [f"{x}3" for x in common_cols]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = []
        continuous_features += [f"{x}1" for x in common_cols]
        continuous_features += [f"{x}2" for x in common_cols]
        continuous_features += [f"{x}3" for x in common_cols]
        categorical_features = [
            "Diagnosis"
        ]
        integer_features = []
        ClfTarget = "Diagnosis"
        
    elif dataset == "concrete":
        data = pd.read_csv('./data/Concrete_Data.csv')
        columns = [
            "Cement",
            "Blast Furnace Slag",
            "Fly Ash",
            "Water",
            "Superplasticizer",
            "Coarse Aggregate",
            "Fine Aggregate",
            "Age",
            "Concrete compressive strength"
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Age")
        continuous_features = columns
        categorical_features = [
            "Age",
        ]
        integer_features = []
        ClfTarget = "Age"
        
    elif dataset == "kings":
        data = pd.read_csv('./data/kc_house_data.csv')
        
        continuous_features = [
            'price', 
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'lat',
            'long',
            'sqft_living15',
            'sqft_lot15',
        ]
        categorical_features = [
            'bedrooms',
            'bathrooms',
            'floors',
            'waterfront',
            'view',
            'condition',
            'grade', 
        ]
        integer_features = [
            'price',
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'sqft_living15',
            'sqft_lot15',
        ]
        ClfTarget = "grade"
        
    elif dataset == "letter":
        data = pd.read_csv('./data/letter-recognition.data', header=None)
        columns = [
            "lettr",
            "x-box",
            "y-box",
            "width",
            "high",
            "onpix",
            "x-bar",
            "y-bar",
            "x2bar",
            "y2bar",
            "xybar",
            "x2ybr",
            "xy2br",
            "x-ege",
            "xegvy",
            "y-ege",
            "yegvx",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("lettr")
        continuous_features = columns
        categorical_features = [
            "lettr"
        ]
        integer_features = columns
        ClfTarget = "lettr"
        
    elif dataset == "loan":
        data = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        
        continuous_features = [
            'Age',
            'Experience',
            'Income', 
            'CCAvg',
            'Mortgage',
        ]
        categorical_features = [
            'Family',
            'Personal Loan',
            'Securities Account',
            'CD Account',
            'Online',
            'CreditCard'
        ]
        integer_features = [
            'Age',
            'Experience',
            'Income', 
            'Mortgage'
        ]
        data = data[continuous_features + categorical_features]
        data = data.dropna()
        ClfTarget = "Personal Loan"
        
    elif dataset == "redwine":
        data = pd.read_csv('./data/winequality-red.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    elif dataset == "whitewine":
        data = pd.read_csv('./data/winequality-white.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    elif dataset == "shoppers":
        ### https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
        data = pd.read_csv('./data/online_shoppers_intention.csv')
        
        assert data.isna().sum().sum() == 0

        continuous_features = [
            'Administrative_Duration',   
            'Informational_Duration',      
            'ProductRelated_Duration',     
            'BounceRates',             
            'ExitRates',                   
            'PageValues',                
            'SpecialDay',                
            'Administrative',    
            'Informational',     
            'ProductRelated',      
        ]

        categorical_features = [
            'Month',               
            'VisitorType',         
            'Weekend',          
            'OperatingSystems',    
            'Browser',            
            'Region',           
            'TrafficType',        
            "Revenue"
        ]

        integer_features = [
            'Administrative',    
            'Informational',      
            'ProductRelated',     
        ]

        ClfTarget = "Revenue"

    elif dataset == "default":
        data = pd.read_csv('./data/default.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            'LIMIT_BAL',  
            'AGE', 
            'BILL_AMT1', 
            'BILL_AMT2',
            'BILL_AMT3',
            'BILL_AMT4', 
            'BILL_AMT5', 
            'BILL_AMT6', 
            'PAY_AMT1',
            'PAY_AMT2', 
            'PAY_AMT3', 
            'PAY_AMT4', 
            'PAY_AMT5', 
            'PAY_AMT6',
        ]
        categorical_features = [
            'SEX', 
            'EDUCATION', 
            'MARRIAGE', 
            'PAY_0',
            'PAY_2', 
            'PAY_3', 
            'PAY_4',
            'PAY_5', 
            'PAY_6', 
            'default_payment_next_month'
        ]
        integer_features = [
            'LIMIT_BAL',  
            'AGE', 
        ]
        ClfTarget = "default_payment_next_month"    

    elif dataset == "BAF":
        # https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data
        data = pd.read_csv('./data/BAF.csv')
        
        ### remove missing values
        data = data.loc[data["prev_address_months_count"] != -1]
        data = data.loc[data["current_address_months_count"] != -1]
        data = data.loc[data["intended_balcon_amount"] >= 0]
        data = data.loc[data["bank_months_count"] != -1]
        data = data.loc[data["session_length_in_minutes"] != -1]
        data = data.loc[data["device_distinct_emails_8w"] != -1]
        data = data.reset_index(drop=True)
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            'income', 
            'name_email_similarity',
            'prev_address_months_count', 
            'current_address_months_count',
            'days_since_request', 
            'intended_balcon_amount',
            'zip_count_4w', 
            'velocity_6h', 
            'velocity_24h',
            'velocity_4w', 
            'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 
            'credit_risk_score', 
            'bank_months_count',
            'proposed_credit_limit', 
            'session_length_in_minutes', 
        ]
        categorical_features = [
            'customer_age', 
            'payment_type', 
            'employment_status',
            'email_is_free', 
            'housing_status',
            'phone_home_valid', 
            'phone_mobile_valid', 
            'has_other_cards', 
            'foreign_request', 
            'source',
            'device_os', 
            'keep_alive_session',
            'device_distinct_emails_8w', 
            'month',
            'fraud_bool', 
        ]
        integer_features = [
            'prev_address_months_count', 
            'current_address_months_count',
            'zip_count_4w', 
            'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 
            'credit_risk_score', 
            'bank_months_count',
        ]
        ClfTarget = "fraud_bool"    
                      
    return data, continuous_features, categorical_features, integer_features, ClfTarget