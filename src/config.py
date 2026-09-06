# Import Library
import os

# Menentukan Path Setiap Data
FOLDER_DATA = "data"
FILE_RAW = os.path.join(FOLDER_DATA, "house_prices_dataset.csv")
FILE_TRAIN = os.path.join(FOLDER_DATA, "train_split.csv")
FILE_TEST = os.path.join(FOLDER_DATA, "test_split.csv")
FILE_LIVE_DRIFT = os.path.join(FOLDER_DATA, "live_drift.csv")

# Menentukan Ukuran Test dan Target
UKURAN_TEST = 0.2
TARGET = "SalePrice"

# Menentukan Feature Numerik
FITUR_NUMERIK = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 
    'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'OverallQual', 'OverallCond'
]

# Menentukan Feature Kategorikal
FITUR_KATEGORIKAL = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 
    'MiscFeature', 'SaleType', 'SaleCondition'
]

# Menentukan Total Seluruh Feature
FITUR = FITUR_NUMERIK + FITUR_KATEGORIKAL

# Menentukan Reproducibility (Random State)
RANDOM_STATE = 42

# Menentukan Model Parameters
N_ESTIMATORS = 200
MAX_DEPTH = 12

# Menentukan Quality Gate
AMBANG_MAE = 70_000        # $ 70,000

# Menentukan MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
NAMA_EXPERIMENT = "prediksi-harga-rumah"
NAMA_MODEL = "harga-rumah-kaggle"
ALIAS_PRODUKSI = "champion"
ALIAS_KANDIDAT = "challenger"
ALIAS_SEBELUMNYA = "champion-sebelumnya"

# Menentukan Serving Configuration
FOLDER_MODEL_EKSPOR = "models/champion"
FILE_LOG = "logs/predictions.log"

# Menentukan Monitoring Parameters
AMBANG_PVALUE = 0.05
JENDELA_MONITOR = 200