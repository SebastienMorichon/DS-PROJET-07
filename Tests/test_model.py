import json
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Charger les fichiers nécessaires (imputer, scaler, modèle)
with open("imputer.pkl", "rb") as imputer_file:
    imputer = pickle.load(imputer_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("best_lightgbm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

required_columns = [
    "SK_ID_CURR", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE",
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
    "FLAG_EMAIL", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START", "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "APARTMENTS_AVG",
    "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG",
    "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE",
    "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE",
    "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "TOTALAREA_MODE", "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "CODE_GENDER_F", "CODE_GENDER_M", "NAME_TYPE_SUITE_Children", "NAME_TYPE_SUITE_Family",
    "NAME_TYPE_SUITE_Group of people", "NAME_TYPE_SUITE_Other_A", "NAME_TYPE_SUITE_Other_B",
    "NAME_TYPE_SUITE_Spouse, partner", "NAME_TYPE_SUITE_Unaccompanied", "NAME_INCOME_TYPE_Businessman",
    "NAME_INCOME_TYPE_Commercial associate", "NAME_INCOME_TYPE_Pensioner", "NAME_INCOME_TYPE_State servant",
    "NAME_INCOME_TYPE_Student", "NAME_INCOME_TYPE_Unemployed", "NAME_INCOME_TYPE_Working",
    "NAME_EDUCATION_TYPE_Academic degree", "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Incomplete higher", "NAME_EDUCATION_TYPE_Lower secondary",
    "NAME_EDUCATION_TYPE_Secondary / secondary special", "NAME_FAMILY_STATUS_Civil marriage",
    "NAME_FAMILY_STATUS_Married", "NAME_FAMILY_STATUS_Separated", "NAME_FAMILY_STATUS_Single / not married",
    "NAME_FAMILY_STATUS_Widow", "NAME_HOUSING_TYPE_Co-op apartment", "NAME_HOUSING_TYPE_House / apartment",
    "NAME_HOUSING_TYPE_Municipal apartment", "NAME_HOUSING_TYPE_Office apartment",
    "NAME_HOUSING_TYPE_Rented apartment", "NAME_HOUSING_TYPE_With parents", "OCCUPATION_TYPE_Accountants",
    "OCCUPATION_TYPE_Cleaning staff", "OCCUPATION_TYPE_Cooking staff", "OCCUPATION_TYPE_Core staff",
    "OCCUPATION_TYPE_Drivers", "OCCUPATION_TYPE_HR staff", "OCCUPATION_TYPE_High skill tech staff",
    "OCCUPATION_TYPE_IT staff", "OCCUPATION_TYPE_Laborers", "OCCUPATION_TYPE_Low-skill Laborers",
    "OCCUPATION_TYPE_Managers", "OCCUPATION_TYPE_Medicine staff", "OCCUPATION_TYPE_Private service staff",
    "OCCUPATION_TYPE_Realty agents", "OCCUPATION_TYPE_Sales staff", "OCCUPATION_TYPE_Secretaries",
    "OCCUPATION_TYPE_Security staff", "OCCUPATION_TYPE_Waiters/barmen staff",
    "WEEKDAY_APPR_PROCESS_START_FRIDAY", "WEEKDAY_APPR_PROCESS_START_MONDAY",
    "WEEKDAY_APPR_PROCESS_START_SATURDAY", "WEEKDAY_APPR_PROCESS_START_SUNDAY",
    "WEEKDAY_APPR_PROCESS_START_THURSDAY", "WEEKDAY_APPR_PROCESS_START_TUESDAY",
    "WEEKDAY_APPR_PROCESS_START_WEDNESDAY", "ORGANIZATION_TYPE_Advertising",
    "ORGANIZATION_TYPE_Agriculture", "ORGANIZATION_TYPE_Bank", "ORGANIZATION_TYPE_Business Entity Type 1",
    "ORGANIZATION_TYPE_Business Entity Type 2", "ORGANIZATION_TYPE_Business Entity Type 3",
    "ORGANIZATION_TYPE_Cleaning", "ORGANIZATION_TYPE_Construction", "ORGANIZATION_TYPE_Culture",
    "ORGANIZATION_TYPE_Electricity", "ORGANIZATION_TYPE_Emergency", "ORGANIZATION_TYPE_Government",
    "ORGANIZATION_TYPE_Hotel", "ORGANIZATION_TYPE_Housing", "ORGANIZATION_TYPE_Industry: type 1",
    "ORGANIZATION_TYPE_Industry: type 10", "ORGANIZATION_TYPE_Industry: type 11",
    "ORGANIZATION_TYPE_Industry: type 12", "ORGANIZATION_TYPE_Industry: type 13",
    "ORGANIZATION_TYPE_Industry: type 2", "ORGANIZATION_TYPE_Industry: type 3",
    "ORGANIZATION_TYPE_Industry: type 4", "ORGANIZATION_TYPE_Industry: type 5",
    "ORGANIZATION_TYPE_Industry: type 6", "ORGANIZATION_TYPE_Industry: type 7",
    "ORGANIZATION_TYPE_Industry: type 8", "ORGANIZATION_TYPE_Industry: type 9",
    "ORGANIZATION_TYPE_Insurance", "ORGANIZATION_TYPE_Kindergarten", "ORGANIZATION_TYPE_Legal Services",
    "ORGANIZATION_TYPE_Medicine", "ORGANIZATION_TYPE_Military", "ORGANIZATION_TYPE_Mobile",
    "ORGANIZATION_TYPE_Other", "ORGANIZATION_TYPE_Police", "ORGANIZATION_TYPE_Postal",
    "ORGANIZATION_TYPE_Realtor", "ORGANIZATION_TYPE_Religion", "ORGANIZATION_TYPE_Restaurant",
    "ORGANIZATION_TYPE_School", "ORGANIZATION_TYPE_Security", "ORGANIZATION_TYPE_Security Ministries",
    "ORGANIZATION_TYPE_Self-employed", "ORGANIZATION_TYPE_Services", "ORGANIZATION_TYPE_Telecom",
    "ORGANIZATION_TYPE_Trade: type 1", "ORGANIZATION_TYPE_Trade: type 2", "ORGANIZATION_TYPE_Trade: type 3",
    "ORGANIZATION_TYPE_Trade: type 4", "ORGANIZATION_TYPE_Trade: type 5", "ORGANIZATION_TYPE_Trade: type 6", "ORGANIZATION_TYPE_Trade: type 7",
    "ORGANIZATION_TYPE_Transport: type 1", "ORGANIZATION_TYPE_Transport: type 2",
    "ORGANIZATION_TYPE_Transport: type 3", "ORGANIZATION_TYPE_Transport: type 4",
    "ORGANIZATION_TYPE_University", "ORGANIZATION_TYPE_XNA", "FONDKAPREMONT_MODE_not specified",
    "FONDKAPREMONT_MODE_org spec account", "FONDKAPREMONT_MODE_reg oper account",
    "FONDKAPREMONT_MODE_reg oper spec account", "HOUSETYPE_MODE_block of flats",
    "HOUSETYPE_MODE_specific housing", "HOUSETYPE_MODE_terraced house", "WALLSMATERIAL_MODE_Block",
    "WALLSMATERIAL_MODE_Mixed", "WALLSMATERIAL_MODE_Monolithic", "WALLSMATERIAL_MODE_Others",
    "WALLSMATERIAL_MODE_Panel", "WALLSMATERIAL_MODE_Stone, brick", "WALLSMATERIAL_MODE_Wooden",
    "EMERGENCYSTATE_MODE_No", "EMERGENCYSTATE_MODE_Yes", "TARGET", "DAYS_EMPLOYED_ANOM",
    "CREDIT_INCOME_PERCENT", "ANNUITY_INCOME_PERCENT", "CREDIT_TERM", "DAYS_EMPLOYED_PERCENT"
]

# Charger les données de test depuis le fichier JSON
with open("Tests/test_data.json", "r") as test_file:
    test_data = json.load(test_file)["inputs"]

def validate_features(row, required_columns):
    """
    S'assure que la ligne possède les mêmes colonnes que celles attendues par le modèle.
    Ajoute des valeurs manquantes pour les colonnes absentes et ignore celles en trop.
    """
    validated_row = {feature: np.nan for feature in required_columns}
    validated_row.update({key: value for key, value in row.items() if key in required_columns})
    return [validated_row[feature] for feature in required_columns]

def test_model_predictions():
    X_test = []
    for i, item in enumerate(test_data):
        try:
            # Validation des colonnes et transformation des valeurs
            row = validate_features(item, required_columns)
            row = [float(value) if value not in [None, "", "False", "True"] else np.nan for value in row]
            X_test.append(row)
        except Exception as e:
            print(f"Erreur lors du traitement de la ligne {i}: {e}")
            continue

    # Convertir en tableau numpy
    X_test = np.array(X_test)
    print(f"Shape de X_test après validation : {X_test.shape}")

    # Appliquer l'imputer et le scaler
    try:
        X_imputed = imputer.transform(X_test)
        X_scaled = scaler.transform(X_imputed)
    except Exception as e:
        print(f"Erreur pendant le prétraitement : {e}")
        return False

    # Faire les prédictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Comparer les prédictions aux cibles (si disponibles)
    target_key = "TARGET"
    targets = [item.get(target_key) for item in test_data if target_key in item]
    if targets:
        for i, (pred, prob, target) in enumerate(zip(predictions, probabilities, targets)):
            print(f"Ligne {i}: Prédiction={pred}, Probabilité={prob:.2f}, Cible={target}")
            if int(pred) != int(target):
                print(f"Incohérence dans la prédiction à la ligne {i}")
    else:
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"Ligne {i}: Prédiction={pred}, Probabilité={prob:.2f}")

    return True

# Exécuter le test
if __name__ == "__main__":
    success = test_model_predictions()
    print(f"Test {'réussi' if success else 'échoué'}")
