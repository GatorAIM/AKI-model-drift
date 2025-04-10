import os
import pandas as pd

def load_disease_code_definitions():
    # Chronic Kidney Disease (CKD) Codes
    ckd_codes_2 = [r'^N18.1', r'^N18.2', r'^N18.3',r'^N18.30', r'^N18.31',r'^N18.32', r'^N18.4', r'^N18.5',  # ICD-10
                 r'^585.1', r'^585.2', r'^585.3', r'^585.4', r'^585.5']  # ICD-9

    ckd_codes = [r'^585.1', r'^585.2', r'^585.3', r'^585.5', r'^585.6', r'^585.7', 
                 r'^N01.7', r'^N01.8', r'^N01.9', r'^N03.1', r'^N03.2', r'^N03.3', r'^N03.5', r'^N03.7', r'^N03.8', r'^N03.9',
        r'^N04.0', r'^N04.1', r'^N04.2', r'^N04.3', r'^N04.4', r'^N04.5', r'^N04.7', r'^N04.8', r'^N04.9',
        r'^N05.0', r'^N05.1', r'^N05.2', r'^N05.3', r'^N05.5', r'^N05.6', r'^N05.7', r'^N05.8', r'^N05.9',
        r'^N07.1', r'^N07.8', r'^N07.9',r'^N14.0', r'^N14.1', r'^N14.2', r'^N15.8', r'^N18.0', r'^N18.8', r'^N18.9', r'^N19', r'^N26'] 

    Peripheral_vascular_code_2 =[r'433.9',r'^I73.9']

    # Diabetes Mellitus Codes
    diabetes_codes = [    r'^E10.0', r'^E10.10', r'^E10.11', r'^E10.12', r'^E10.20', r'^E10.21', r'^E10.22', r'^E10.28',
        r'^E10.32', r'^E10.33', r'^E10.35', r'^E10.38', r'^E10.40', r'^E10.41', r'^E10.42', r'^E10.50',
        r'^E10.51', r'^E10.52', r'^E10.60', r'^E10.61', r'^E10.62', r'^E10.63', r'^E10.68', r'^E10.70',
        r'^E10.71', r'^E10.78', r'^E10.9',

        r'^E11.0', r'^E11.10', r'^E11.11', r'^E11.12', r'^E11.2021', r'^E11.22', r'^E11.28',
        r'^E11.30', r'^E11.31', r'^E11.32', r'^E11.33', r'^E11.35', r'^E11.38', r'^E11.40',
        r'^E11.41', r'^E11.42', r'^E11.50', r'^E11.51', r'^E11.52', r'^E11.60', r'^E11.61',
        r'^E11.62', r'^E11.63', r'^E11.68', r'^E11.70', r'^E11.71', r'^E11.78', r'^E11.9',

        r'^E13.0', r'^E13.10', r'^E13.20', r'^E13.22', r'^E13.28', r'^E13.40', r'^E13.41',
        r'^E13.42', r'^E13.50', r'^E13.51', r'^E13.63', r'^E13.70', r'^E13.71', r'^E13.78', r'^E13.9',

        r'^E14.0', r'^E14.10', r'^E14.11', r'^E14.12', r'^E14.20', r'^E14.21', r'^E14.22', r'^E14.28',
        r'^E14.33', r'^E14.40', r'^E14.41', r'^E14.42', r'^E14.50', r'^E14.51', r'^E14.52',
        r'^E14.60', r'^E14.62', r'^E14.63', r'^E14.68', r'^E14.70', r'^E14.71', r'^E14.78', r'^E14.9',  # ICD-10
                      r'^250', r'^249']  

    diabetes_codes_2 = [r'^E08.9', r'^E11.9', r'^250']  

    # Hypertension Codes
    hypertension_essential_codes = [r'^401',r'^I10']

    hypertension_all_codes = [r'^I10', r'^I11',r'^I12',r'^I13', r'^I15',  
                          r'^401',r'^402',r'^403', r'^404', r'405']  

    hypertensive_disease_codes=[r'^401.0', r'^401.1', r'^401.9',
        r'^402.00', r'^402.01', r'^402.10', r'^402.11', r'^402.90', r'^402.91',
        r'^I10', r'^I11.0', r'^I11.9']


    # Liver Diseases Codes 
    hepatitis_other_liver_diseases_codes = [    r'^K70.0', r'^K70.1', r'^K70.2', r'^K70.3', r'^K70.4', r'^K70.9',
        r'^K71.0', r'^K71.1', r'^K71.2', r'^K71.6', r'^K71.8', r'^K71.9',
        r'^K72.1', r'^K72.9',
        r'^K73.0', r'^K73.1', r'^K73.2', r'^K73.8', r'^K73.9',
        r'^K74.0', r'^K74.1', r'^K74.3', r'^K74.4', r'^K74.5', r'^K74.6',
        r'^K75.0', r'^K75.1', r'^K75.2', r'^K75.3', r'^K75.4', r'^K75.8', r'^K75.9',
        r'^K76.0', r'^K76.1', r'^K76.3', r'^K76.5', r'^K76.6', r'^K76.7', r'^K76.8', r'^K76.9',  # ICD-10
        r'^571',r'^572',r'^573' ]  

    chronic_liver_disease_codes_2=[r'^571', r'^571.1', r'^571.2', r'^571.3', r'^571.4', r'^571.5', r'^571.6', r'^571.8', r'^571.9',
        r'^571.40', r'^571.41', r'^571.42', r'^571.49', r'^K70', r'^K71', r'^K72', r'^K73', r'^K74', r'^K75', r'^K76', r'^K77'
        ]

    circulatory_system_diseases_codes_2=[
        r'^413.9', r'^410.11', r'^410.91', r'^414.01', r'^414.00', r'^412', r'^414.4', r'^414.9',
        r'^I20.9', r'^I21.09', r'^I21.3', r'^I25.10', r'^I25.2', r'^I25.84', r'^I25.9'
    ]

    ischemic_heart_diseases_codes_2=[
        r'^410.00', r'^410.10', r'^412', r'^413.9', r'^414.00', r'^414.01',
        r'^I21.09', r'^I25.2', r'^I20.9', r'^I25.10'
    ]

    congestive_heart_failure_codes_2=[
        r'^428',  
        r'^I50.22'  
    ]

    stroke_codes=[
        r'^434.91',  
        r'^I63.9'    
    ]

    cardiovascular_diseases_codes=[
        # Cardiovascular and Ischaemic Disease
        r'^I25.10', r'^I50.9', r'^I63.9', r'^I65.23', r'^I65.29', r'^I67.2', r'^I67.9', r'^I73.9',
        r'^I20.9', r'^I21.09', r'^I21.3', r'^I25.2', r'^I25.84', r'^I25.9',

        # Hypertensive Disease
        r'^I10', r'^I11.0', r'^I11.9',

        # Metabolic and Nutritional Diseases
        r'^E11.65', r'^E11.9', r'^E55.9', r'^E78.00', r'^E78.1', r'^E78.2', r'^E78.5', r'^E88.81',

        # Abnormal Glucose
        r'^R73.01', r'^R73.09', r'^R73.9',

        r'^414.00', r'^428.0', r'^434.91', r'^433.10', r'^433.90', r'^437.0', r'^437.9', r'^443.9',
        r'^413.9', r'^410.10', r'^410.90', r'^412', r'^414.4', r'^414.9', r'^401.9', r'^402.01', 
        r'^402.10', r'^250.60', r'^250.00', r'^268.9', r'^272.0', r'^272.1', r'^272.2', r'^272.4', 
        r'^277.7', r'^790.21', r'^790.29', r'^790.6']
    # Cancer Codes
    cancer_codes_2 = [
        r'^D00', r'^D01', r'^D02', r'^D03', r'^D04', r'^D05', r'^D06', r'^D07', r'^D08', r'^D09',
        r'^D10', r'^D11', r'^D12', r'^D13', r'^D14', r'^D15', r'^D16', r'^D17', r'^D18', r'^D19',
        r'^D20', r'^D21', r'^D22', r'^D23', r'^D24', r'^D25', r'^D26', r'^D27', r'^D28', r'^D29',
        r'^D30', r'^D31', r'^D32', r'^D33', r'^D34', r'^D35', r'^D36', r'^D37', r'^D38', r'^D39',
        r'^D40', r'^D41', r'^D42', r'^D43', r'^D44', r'^D45', r'^D46', r'^D47', r'^D48', r'^D49',

        r'^C00', r'^C01', r'^C02', r'^C03', r'^C04', r'^C05', r'^C06', r'^C07', r'^C08', r'^C09',
        r'^C10', r'^C11', r'^C12', r'^C13', r'^C14', r'^C15', r'^C16', r'^C17', r'^C18', r'^C19',
        r'^C20', r'^C21', r'^C22', r'^C23', r'^C24', r'^C25', r'^C26', r'^C27', r'^C28', r'^C29',
        r'^C30', r'^C31', r'^C32', r'^C33', r'^C34', r'^C35', r'^C36', r'^C37', r'^C38', r'^C39',
        r'^C40', r'^C41', r'^C42', r'^C43', r'^C44', r'^C45', r'^C46', r'^C47', r'^C48', r'^C49',
        r'^C50', r'^C51', r'^C52', r'^C53', r'^C54', r'^C55', r'^C56', r'^C57', r'^C58', r'^C59',
        r'^C60', r'^C61', r'^C62', r'^C63', r'^C64', r'^C65', r'^C66', r'^C67', r'^C68', r'^C69',
        r'^C70', r'^C71', r'^C72', r'^C73', r'^C74', r'^C75', r'^C76', r'^C77', r'^C78', r'^C79',
        r'^C80', r'^C81', r'^C82', r'^C83', r'^C84', r'^C85', r'^C86', r'^C87', r'^C88', r'^C89',
        r'^C90', r'^C91', r'^C92', r'^C93', r'^C94', r'^C95', r'^C96',

        r'^140', r'^141', r'^142', r'^143', r'^144', r'^145', r'^146', r'^147', r'^148', r'^149',
        r'^150', r'^151', r'^152', r'^153', r'^154', r'^155', r'^156', r'^157', r'^158', r'^159',
        r'^160', r'^161', r'^162', r'^163', r'^164', r'^165', r'^166', r'^167', r'^168', r'^169',
        r'^170', r'^171', r'^172', r'^173', r'^174', r'^175', r'^176', r'^177', r'^178', r'^179',
        r'^180', r'^181', r'^182', r'^183', r'^184', r'^185', r'^186', r'^187', r'^188', r'^189',
        r'^190', r'^191', r'^192', r'^193', r'^194', r'^195', r'^196', r'^197', r'^198', r'^199',
        r'^200', r'^201', r'^202', r'^203', r'^204', r'^205', r'^206', r'^207', r'^208', r'^209',
        r'^210', r'^211', r'^212', r'^213', r'^214', r'^215', r'^216', r'^217', r'^218', r'^219',
        r'^220', r'^221', r'^222', r'^223', r'^224', r'^225', r'^226', r'^227', r'^228', r'^229',
        r'^230', r'^231', r'^232', r'^233', r'^234', r'^235', r'^236', r'^237', r'^238', r'^239',
        r'^240'
    ]

    # Chronic Obstructive Pulmonary Disease (COPD) and bronchiectasis Codes
    copd_bronchiectasis_codes = [    
        r'^J41.8',   
        r'^J42',     
        r'^J43.1', r'^J43.8', r'^J43.9',  
        r'^J44.1',   
        r'^J47',     
        r'^J45.00', r'^J45.01', r'^J45.10', r'^J45.11', r'^J45.80', r'^J45.90', r'^J45.91'  
        r'^491',  
        r'^492',  
        r'^494',  
        r'^496'   
    ]  



    # Malnourished and Metabolic Syndrome Patients Codes 
    nutritional_deficiencies_codes = [r'^E41', r'^E42', r'^E43', r'^E46',  
                          r'^E55.0', r'^E55.9', r'^M83.8',r'^M83.9',
                          r'^260', r'^261', r'^262', r'^263', r'^268']  

    other_nutritional_endocrine_immunity_metabolic_codes = [r'^E41', r'^E42', r'^E43', r'^E46', 
        r'^270', r'^271', r'^273',  r'^275', r'^277', r'^278', r'^278' 
        r'^C88.00',
        r'^D47.2',
        r'^D76.0',
        r'^D80.1', r'^D80.2', r'^D80.3', r'^D80.4',
        r'^D81.1', r'^D81.2', r'^D81.3', r'^D81.8', r'^D81.9',
        r'^D82.1', r'^D82.3', r'^D82.9',
        r'^D83.1', r'^D83.9',
        r'^D84.1', r'^D84.8', r'^D84.9',
        r'^D89.1', r'^D89.9',
        r'^E65',
        r'^E66.0', r'^E66.2', r'^E66.8', r'^E66.9',
        r'^E67.3',
        r'^E70.1', r'^E70.2', r'^E70.3',
        r'^E71.0', r'^E71.3',
        r'^E72.0', r'^E72.1', r'^E72.2', r'^E72.3', r'^E72.4', r'^E72.5', r'^E72.8',
        r'^E73.0', r'^E73.9',
        r'^E74.0', r'^E74.2', r'^E74.3', r'^E74.4', r'^E74.8',
        r'^E76.0',
        r'^E77.1', r'^E77.8',
        r'^E80.0', r'^E80.1', r'^E80.2', r'^E80.4', r'^E80.6', r'^E80.7',
        r'^E83.0', r'^E83.10', r'^E83.18', r'^E83.3', r'^E83.4', r'^E83.5',
        r'^E84.0', r'^E84.1', r'^E84.8', r'^E84.9',
        r'^E85.0', r'^E85.1', r'^E85.3', r'^E85.4', r'^E85.8', r'^E85.9',
        r'^E88.0', r'^E88.8', r'^E88.9',
        r'^E89.8'
    ]
    metabolic_disease_codes_2 = [
        r'^268.9', r'^272.0', r'^272.1', r'^272.2', r'^272.4', r'^277.7',
            r'^E55.9', r'^E78.0', r'^E78.1', r'^E78.2', r'^E78.5', r'^E88.81'
    ]

    Alcohol_disorders_codes = [ r'^291', r'^292', r'^303', r'^304', r'^305',
        r'^F10.0', r'^F10.1', r'^F10.2', r'^F10.3', r'^F10.4', r'^F10.5', r'^F10.6', r'^F10.7', r'^F10.8', r'^F10.9',
        r'^F11.0', r'^F11.1', r'^F11.2', r'^F11.3', r'^F11.4', r'^F11.5', r'^F11.6', r'^F11.7', r'^F11.8', r'^F11.9',
        r'^F12.0', r'^F12.1', r'^F12.2', r'^F12.3', r'^F12.4', r'^F12.5', r'^F12.6', r'^F12.7', r'^F12.8', r'^F12.9',
        r'^F13.0', r'^F13.1', r'^F13.2', r'^F13.3', r'^F13.4', r'^F13.5', r'^F13.6', r'^F13.7', r'^F13.8', r'^F13.9',
        r'^F14.0', r'^F14.1', r'^F14.2', r'^F14.3', r'^F14.4', r'^F14.5', r'^F14.6', r'^F14.7', r'^F14.8', r'^F14.9',
        r'^F15.0', r'^F15.1', r'^F15.2', r'^F15.3', r'^F15.4', r'^F15.5', r'^F15.6', r'^F15.7', r'^F15.8', r'^F15.9',
        r'^F16.0', r'^F16.1', r'^F16.2', r'^F16.3', r'^F16.4', r'^F16.5', r'^F16.7', r'^F16.8',
        r'^F17.2', r'^F17.3',
        r'^F18.0', r'^F18.1', r'^F18.2', r'^F18.5', r'^F18.7',
        r'^F19.0', r'^F19.1', r'^F19.2', r'^F19.3', r'^F19.4', r'^F19.5', r'^F19.6', r'^F19.7', r'^F19.8', r'^F19.9',
        r'^F55',
        r'^G31.2']  

    return (
        ckd_codes,
        ckd_codes_2,
        Peripheral_vascular_code_2,
        diabetes_codes,
        diabetes_codes_2,
        hypertension_essential_codes,
        hypertension_all_codes,
        hypertensive_disease_codes,
        hepatitis_other_liver_diseases_codes,
        chronic_liver_disease_codes_2,
        circulatory_system_diseases_codes_2,
        ischemic_heart_diseases_codes_2,
        congestive_heart_failure_codes_2,
        stroke_codes,
        cardiovascular_diseases_codes,
        cancer_codes_2,
        copd_bronchiectasis_codes,
        nutritional_deficiencies_codes,
        other_nutritional_endocrine_immunity_metabolic_codes,
        metabolic_disease_codes_2,
        Alcohol_disorders_codes
    )


def filter_dxdata_by_year(dxdata_by_year, target_codes):
    filtered_dxdata_by_year = {}

    regex_pattern = '|'.join(target_codes)

    for year, data in dxdata_by_year.items():
        try:
            filtered_data = data[data['DX'].astype(str).str.contains(regex_pattern, regex=True)]
            filtered_dxdata_by_year[year] = filtered_data
        except KeyError as e:
            print(f"Year {year} data has no 'DX' column: {e}")
        except Exception as e:
            print(f"Error occurred while processing year {year}: {e}")

    return filtered_dxdata_by_year


def filter_all_diseases_by_code(dxdata_by_year,
                                 ckd_codes,
                                 ckd_codes_2,
                                 diabetes_codes,
                                 diabetes_codes_2,
                                 hypertension_essential_codes,
                                 hypertension_all_codes,
                                 hypertensive_disease_codes,
                                 hepatitis_other_liver_diseases_codes,
                                 chronic_liver_disease_codes_2,
                                 circulatory_system_diseases_codes_2,
                                 ischemic_heart_diseases_codes_2,
                                 congestive_heart_failure_codes_2,
                                 stroke_codes,
                                 cardiovascular_diseases_codes,
                                 cancer_codes_2,
                                 copd_bronchiectasis_codes,
                                 nutritional_deficiencies_codes,
                                 other_nutritional_endocrine_immunity_metabolic_codes,
                                 metabolic_disease_codes_2,
                                 Alcohol_disorders_codes,
                                 Peripheral_vascular_code_2):
    
    filtered_dxdata_by_year_ckd_codes = filter_dxdata_by_year(dxdata_by_year, ckd_codes)
    filtered_dxdata_by_year_ckd_codes_2 = filter_dxdata_by_year(dxdata_by_year, ckd_codes_2)
    filtered_dxdata_by_year_diabetes_codes = filter_dxdata_by_year(dxdata_by_year, diabetes_codes)
    filtered_dxdata_by_year_diabetes_codes_2 = filter_dxdata_by_year(dxdata_by_year, diabetes_codes_2)
    filtered_dxdata_by_year_hypertension_essential_codes = filter_dxdata_by_year(dxdata_by_year, hypertension_essential_codes)
    filtered_dxdata_by_year_hypertension_all_codes = filter_dxdata_by_year(dxdata_by_year, hypertension_all_codes)
    filtered_dxdata_by_year_hypertensive_disease_codes = filter_dxdata_by_year(dxdata_by_year, hypertensive_disease_codes)
    filtered_dxdata_by_year_hepatitis_other_liver_diseases_codes = filter_dxdata_by_year(dxdata_by_year, hepatitis_other_liver_diseases_codes)
    filtered_dxdata_by_year_chronic_liver_disease_codes_2 = filter_dxdata_by_year(dxdata_by_year, chronic_liver_disease_codes_2)
    filtered_dxdata_by_year_circulatory_system_diseases_codes_2 = filter_dxdata_by_year(dxdata_by_year, circulatory_system_diseases_codes_2)
    filtered_dxdata_by_year_ischemic_heart_diseases_codes_2 = filter_dxdata_by_year(dxdata_by_year, ischemic_heart_diseases_codes_2)
    filtered_dxdata_by_year_congestive_heart_failure_codes_2 = filter_dxdata_by_year(dxdata_by_year, congestive_heart_failure_codes_2)
    filtered_dxdata_by_year_stroke_codes = filter_dxdata_by_year(dxdata_by_year, stroke_codes)
    filtered_dxdata_by_year_cardiovascular_diseases_codes = filter_dxdata_by_year(dxdata_by_year, cardiovascular_diseases_codes)
    filtered_dxdata_by_year_cancer_codes_2 = filter_dxdata_by_year(dxdata_by_year, cancer_codes_2)
    filtered_dxdata_by_year_copd_bronchiectasis_codes = filter_dxdata_by_year(dxdata_by_year, copd_bronchiectasis_codes)
    filtered_dxdata_by_year_nutritional_deficiencies_codes = filter_dxdata_by_year(dxdata_by_year, nutritional_deficiencies_codes)
    filtered_dxdata_by_year_other_nutritional_endocrine_immunity_metabolic_codes = filter_dxdata_by_year(dxdata_by_year, other_nutritional_endocrine_immunity_metabolic_codes)
    filtered_dxdata_by_year_metabolic_disease_codes_2 = filter_dxdata_by_year(dxdata_by_year, metabolic_disease_codes_2)
    filtered_dxdata_by_year_Alcohol_disorders_codes = filter_dxdata_by_year(dxdata_by_year, Alcohol_disorders_codes)
    filtered_dxdata_by_year_Peripheral_vascular_code_2 = filter_dxdata_by_year(dxdata_by_year, Peripheral_vascular_code_2)


    return (
        filtered_dxdata_by_year_ckd_codes,
        filtered_dxdata_by_year_ckd_codes_2,
        filtered_dxdata_by_year_diabetes_codes,
        filtered_dxdata_by_year_diabetes_codes_2,
        filtered_dxdata_by_year_hypertension_essential_codes,
        filtered_dxdata_by_year_hypertension_all_codes,
        filtered_dxdata_by_year_hypertensive_disease_codes,
        filtered_dxdata_by_year_hepatitis_other_liver_diseases_codes,
        filtered_dxdata_by_year_chronic_liver_disease_codes_2,
        filtered_dxdata_by_year_circulatory_system_diseases_codes_2,
        filtered_dxdata_by_year_ischemic_heart_diseases_codes_2,
        filtered_dxdata_by_year_congestive_heart_failure_codes_2,
        filtered_dxdata_by_year_stroke_codes,
        filtered_dxdata_by_year_cardiovascular_diseases_codes,
        filtered_dxdata_by_year_cancer_codes_2,
        filtered_dxdata_by_year_copd_bronchiectasis_codes,
        filtered_dxdata_by_year_nutritional_deficiencies_codes,
        filtered_dxdata_by_year_other_nutritional_endocrine_immunity_metabolic_codes,
        filtered_dxdata_by_year_metabolic_disease_codes_2,
        filtered_dxdata_by_year_Alcohol_disorders_codes,
        filtered_dxdata_by_year_Peripheral_vascular_code_2
    )

def process_filtered_data_by_year(filtered_data_by_year_dict, disease_name):
    for year, data in filtered_data_by_year_dict.items():
        try:
            data = data.copy()

            if 'DX_DATE' in data.columns:
                data.loc[:, 'DX_DATE'] = pd.to_datetime(data['DX_DATE'], errors='coerce')
                print(f"{disease_name} year {year}: converted DX_DATE to datetime")
            else:
                print(f"Warning: {disease_name} year {year} missing 'DX_DATE' column")

            for col in ['PATID', 'ENCOUNTERID']:
                if col in data.columns:
                    data.loc[:, col] = pd.to_numeric(data[col], errors='coerce').astype(float)
                    print(f"{disease_name} year {year}: converted {col} to float")
                else:
                    print(f"Warning: {disease_name} year {year} missing '{col}' column")

            filtered_data_by_year_dict[year] = data

        except Exception as e:
            print(f"Error: processing {disease_name} year {year} failed: {e}")

    return filtered_data_by_year_dict

def process_all_filtered_dxdata(
    filtered_dxdata_by_year_ckd_codes,
    filtered_dxdata_by_year_ckd_codes_2,
    filtered_dxdata_by_year_diabetes_codes,
    filtered_dxdata_by_year_diabetes_codes_2,
    filtered_dxdata_by_year_hypertension_essential_codes,
    filtered_dxdata_by_year_hypertension_all_codes,
    filtered_dxdata_by_year_hypertensive_disease_codes,
    filtered_dxdata_by_year_hepatitis_other_liver_diseases_codes,
    filtered_dxdata_by_year_chronic_liver_disease_codes_2,
    filtered_dxdata_by_year_circulatory_system_diseases_codes_2,
    filtered_dxdata_by_year_ischemic_heart_diseases_codes_2,
    filtered_dxdata_by_year_congestive_heart_failure_codes_2,
    filtered_dxdata_by_year_stroke_codes,
    filtered_dxdata_by_year_cardiovascular_diseases_codes,
    filtered_dxdata_by_year_cancer_codes_2,
    filtered_dxdata_by_year_copd_bronchiectasis_codes,
    filtered_dxdata_by_year_nutritional_deficiencies_codes,
    filtered_dxdata_by_year_other_nutritional_endocrine_immunity_metabolic_codes,
    filtered_dxdata_by_year_metabolic_disease_codes_2,
    filtered_dxdata_by_year_Alcohol_disorders_codes,
    filtered_dxdata_by_year_Peripheral_vascular_code_2
):
    return (
        process_filtered_data_by_year(filtered_dxdata_by_year_ckd_codes, "CKD"),
        process_filtered_data_by_year(filtered_dxdata_by_year_ckd_codes_2, "CKD_2"),
        process_filtered_data_by_year(filtered_dxdata_by_year_diabetes_codes, "Diabetes"),
        process_filtered_data_by_year(filtered_dxdata_by_year_diabetes_codes_2, "Diabetes_2"),
        process_filtered_data_by_year(filtered_dxdata_by_year_hypertension_essential_codes, "Hypertension_Essential"),
        process_filtered_data_by_year(filtered_dxdata_by_year_hypertension_all_codes, "Hypertension_All"),
        process_filtered_data_by_year(filtered_dxdata_by_year_hypertensive_disease_codes, "Hypertensive_Disease"),
        process_filtered_data_by_year(filtered_dxdata_by_year_hepatitis_other_liver_diseases_codes, "Hepatitis_Liver_Diseases"),
        process_filtered_data_by_year(filtered_dxdata_by_year_chronic_liver_disease_codes_2, "Chronic_Liver_Disease"),
        process_filtered_data_by_year(filtered_dxdata_by_year_circulatory_system_diseases_codes_2, "Circulatory_System_Diseases"),
        process_filtered_data_by_year(filtered_dxdata_by_year_ischemic_heart_diseases_codes_2, "Ischemic_Heart_Diseases"),
        process_filtered_data_by_year(filtered_dxdata_by_year_congestive_heart_failure_codes_2, "Congestive_Heart_Failure"),
        process_filtered_data_by_year(filtered_dxdata_by_year_stroke_codes, "Stroke"),
        process_filtered_data_by_year(filtered_dxdata_by_year_cardiovascular_diseases_codes, "Cardiovascular_Diseases"),
        process_filtered_data_by_year(filtered_dxdata_by_year_cancer_codes_2, "Cancer"),
        process_filtered_data_by_year(filtered_dxdata_by_year_copd_bronchiectasis_codes, "COPD_Bronchiectasis"),
        process_filtered_data_by_year(filtered_dxdata_by_year_nutritional_deficiencies_codes, "Nutritional_Deficiencies"),
        process_filtered_data_by_year(filtered_dxdata_by_year_other_nutritional_endocrine_immunity_metabolic_codes, "Other_Nutritional_Endocrine_Immunity_Metabolic"),
        process_filtered_data_by_year(filtered_dxdata_by_year_metabolic_disease_codes_2, "Metabolic_Disease"),
        process_filtered_data_by_year(filtered_dxdata_by_year_Alcohol_disorders_codes, "Alcohol_Disorders"),
        process_filtered_data_by_year(filtered_dxdata_by_year_Peripheral_vascular_code_2, "Peripheral_Vascular")
    )


def load_and_process_train_test_data(save_path="./"):
    data_by_year_training = {}
    data_by_year_test = {}

    for file_name in os.listdir(save_path):
        if file_name.startswith("training_data_"):
            year = int(file_name.split("_")[2].split(".")[0])
            file_path = os.path.join(save_path, file_name)
            data_by_year_training[year] = pd.read_csv(file_path)
            print(f"Training data for year {year} loaded from {file_path}.")

        elif file_name.startswith("test_data_"):
            year = int(file_name.split("_")[2].split(".")[0])
            file_path = os.path.join(save_path, file_name)
            data_by_year_test[year] = pd.read_csv(file_path)
            print(f"Test data for year {year} loaded from {file_path}.")

    print("All training and test datasets have been loaded.")

    for dataset_dict, name in [(data_by_year_training, "training"), (data_by_year_test, "test")]:
        for year, data in dataset_dict.items():
            try:
                dataset_dict[year]['ADMIT_DATE'] = pd.to_datetime(data['ADMIT_DATE'], errors='coerce')
                print(f"{name.title()} {year}: ADMIT_DATE converted to datetime")
            except KeyError as e:
                print(f"{name.title()} {year}: missing 'ADMIT_DATE' column: {e}")

            try:
                dataset_dict[year]['PATID'] = data['PATID'].astype(float)
                dataset_dict[year]['ENCOUNTERID'] = data['ENCOUNTERID'].astype(float)
                print(f"{name.title()} {year}: PATID & ENCOUNTERID converted to float")
            except KeyError as e:
                print(f"{name.title()} {year}: missing ID columns: {e}")
            except ValueError as e:
                print(f"{name.title()} {year}: error converting IDs to float: {e}")

            try:
                is_unique = not data.duplicated(subset=['PATID', 'ENCOUNTERID']).any()
                if is_unique:
                    print(f"{name.title()} {year}: PATID + ENCOUNTERID is unique")
                else:
                    print(f"{name.title()} {year}: duplicate PATID + ENCOUNTERID found")
                    duplicated = data[data.duplicated(subset=['PATID', 'ENCOUNTERID'], keep=False)]
                    print(f"Duplicated records:\n{duplicated}")
            except KeyError as e:
                print(f"{name.title()} {year}: missing columns for uniqueness check: {e}")

            try:
                print(f"{name.title()} {year}: total records = {len(data)}")
            except Exception as e:
                print(f"{name.title()} {year}: error counting records: {e}")

    return data_by_year_training, data_by_year_test

def update_data_by_year_with_condition(data_by_year, filtered_dxdata_by_year, column_name):
    for year, data in data_by_year.items():
        try:
            if year not in filtered_dxdata_by_year:
                print(f"filtered_dxdata_by_year[{year}] not found. Skipping.")
                continue

            filtered_data = filtered_dxdata_by_year[year]

            merged_data = data.merge(
                filtered_data[['PATID', 'ENCOUNTERID', 'DX', 'DX_DATE']],
                on=['PATID', 'ENCOUNTERID'],
                how='left'
            )

            merged_data['DX_DATE'] = pd.to_datetime(merged_data['DX_DATE'], errors='coerce')
            merged_data['ADMIT_DATE'] = pd.to_datetime(merged_data['ADMIT_DATE'], errors='coerce')

            merged_data[column_name] = merged_data.apply(
                lambda row: 1 if (
                    pd.notna(row['DX_DATE']) and
                    pd.notna(row['ADMIT_DATE']) and
                    row['DX_DATE'] < row['ADMIT_DATE'] + pd.Timedelta(days=row['SINCE_ADMIT'])
                ) else 0,
                axis=1
            )

            final_data = merged_data.groupby(['PATID', 'ENCOUNTERID'], as_index=False).agg({
                column_name: 'max'
            })

            data_by_year[year] = data.merge(
                final_data[['PATID', 'ENCOUNTERID', column_name]],
                on=['PATID', 'ENCOUNTERID'],
                how='left'
            )

            data_by_year[year][column_name] = data_by_year[year][column_name].fillna(0).astype(int)

            print(f"{year}: column '{column_name}' updated.")

        except Exception as e:
            print(f"Error updating year {year} for column '{column_name}': {e}")
