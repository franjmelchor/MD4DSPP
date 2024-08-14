import os

import pandas as pd
import numpy as np
import functions.contract_invariants as contract_invariants
import functions.contract_pre_post as contract_pre_post
from helpers.enumerations import Belong, Operator, Operation, SpecialType, DataType, DerivedType, Closure, FilterType
from helpers.logger import set_logger

def generateWorkflow():
#-----------------New DataProcessing-----------------
#--------------------------------------Input data dictionaries--------------------------------------
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary='./python_dataDictionaries/missing_input_dataDictionary.csv'
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_sep=','
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary='./python_dataDictionaries/missing_input_dataDictionary.csv'
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_sep=','
	imputeMissingByMean_avg_income_distance__input_dataDictionary='./python_dataDictionaries/missing_input_dataDictionary.csv'
	imputeMissingByMean_avg_income_distance__input_dataDictionary_sep=','
	imputeMissingByLinearInterpolation_satscore__input_dataDictionary='./python_dataDictionaries/missing_input_dataDictionary.csv'
	imputeMissingByLinearInterpolation_satscore__input_dataDictionary_sep=','
	rowFilterRange_init_span__input_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	rowFilterRange_init_span__input_dataDictionary_sep=','
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary='./python_dataDictionaries/rowFilter_output_dataDictionary.csv'
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary_sep=','
	mapping_TERRITORY__input_dataDictionary='./python_dataDictionaries/columnFilter_output_dataDictionary.csv'
	mapping_TERRITORY__input_dataDictionary_sep=','
	mapping_Instate__input_dataDictionary='./python_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv'
	mapping_Instate__input_dataDictionary_sep=','
	stringToNumber_TERRITORY_Instate__input_dataDictionary='./python_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv'
	stringToNumber_TERRITORY_Instate__input_dataDictionary_sep=','
	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary='./python_dataDictionaries/stringToNumber_output_dataDictionary.csv'
	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_sep=','
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_sep=','
	binner_avg_income__input_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	binner_avg_income__input_dataDictionary_sep=','
	binner_satscore__input_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	binner_satscore__input_dataDictionary_sep=','
	binner_avg_income__input_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	binner_avg_income__input_dataDictionary_sep=','
#--------------------------------------Output data dictionaries--------------------------------------
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary_sep=','
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary_sep=','
	imputeMissingByMean_avg_income_distance__output_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	imputeMissingByMean_avg_income_distance__output_dataDictionary_sep=','
	imputeMissingByLinearInterpolation_satscore__output_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	imputeMissingByLinearInterpolation_satscore__output_dataDictionary_sep=','
	rowFilterRange_init_span__output_dataDictionary='./python_dataDictionaries/rowFilter_output_dataDictionary.csv'
	rowFilterRange_init_span__output_dataDictionary_sep=','
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary='./python_dataDictionaries/columnFilter_output_dataDictionary.csv'
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary_sep=','
	mapping_TERRITORY__output_dataDictionary='./python_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv'
	mapping_TERRITORY__output_dataDictionary_sep=','
	invalid='./python_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv'
	invalid_sep=','
	invalid='./python_dataDictionaries/stringToNumber_output_dataDictionary.csv'
	invalid_sep=','
	imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary_sep=','
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary='./python_dataDictionaries/numericBinner_output_dataDictionary.csv'
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_sep=','
	binner_TERRITORY__output_dataDictionary='./python_dataDictionaries/numericBinner_output_dataDictionary.csv'
	binner_TERRITORY__output_dataDictionary_sep=','
	binner_satscore__output_dataDictionary='./python_dataDictionaries/numericBinner_output_dataDictionary.csv'
	binner_satscore__output_dataDictionary_sep=','
	binner_avg_income__output_dataDictionary='./python_dataDictionaries/numericBinner_output_dataDictionary.csv'
	binner_avg_income__output_dataDictionary_sep=','
set_logger("contracts")
generateWorkflow()
