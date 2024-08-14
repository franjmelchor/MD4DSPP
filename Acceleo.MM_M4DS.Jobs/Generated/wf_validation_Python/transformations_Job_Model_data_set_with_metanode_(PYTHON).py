import pandas as pd
import numpy as np
import functions.data_transformations as data_transformations
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
	#-----------------New DataProcessing-----------------
		imputeByDerivedValue_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_input_dataDictionary.csv', sep = ',')
		imputeByDerivedValue_input_dataDictionary_transformed=imputeByDerivedValue_input_dataDictionary.copy()
		missing_values_list=[]
		
		imputeByDerivedValue_input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeByDerivedValue_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'sex', field_out = 'sex')
		
		missing_values_list=[]
		
		imputeByDerivedValue_input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeByDerivedValue_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'IRSCHOOL', field_out = 'IRSCHOOL')
		
		missing_values_list=[]
		
		imputeByDerivedValue_input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeByDerivedValue_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'ETHNICITY', field_out = 'ETHNICITY')
		
		imputeByDerivedValue_output_dataDictionary=imputeByDerivedValue_input_dataDictionary_transformed
		imputeByDerivedValue_output_dataDictionary.to_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv')
		imputeByDerivedValue_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		imputeByFixValue_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_input_dataDictionary.csv', sep=',')
	
		imputeByFixValue_input_dataDictionary_transformed=imputeByFixValue_input_dataDictionary.copy()
		missing_values_list=[]
		
		imputeByFixValue_input_dataDictionary_transformed=data_transformations.transform_special_value_fix_value(data_dictionary=imputeByFixValue_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(0), fix_value_output='Unknown',
																	  missing_values=missing_values_list,		
								                                      data_type_output = DataType(0),
																	  axis_param=0, field_in = 'ACADEMIC_INTEREST_2', field_out = 'ACADEMIC_INTEREST_2')
		
		missing_values_list=[]
		
		imputeByFixValue_input_dataDictionary_transformed=data_transformations.transform_special_value_fix_value(data_dictionary=imputeByFixValue_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(0), fix_value_output='Unknown',
																	  missing_values=missing_values_list,		
								                                      data_type_output = DataType(0),
																	  axis_param=0, field_in = 'ACADEMIC_INTEREST_1', field_out = 'ACADEMIC_INTEREST_1')
		
		imputeByFixValue_output_dataDictionary=imputeByFixValue_input_dataDictionary_transformed
		imputeByFixValue_output_dataDictionary.to_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv')
		imputeByFixValue_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		imputeByNumericOp_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_input_dataDictionary.csv', sep=',')
	
		imputeByNumericOp_input_dataDictionary_transformed=imputeByNumericOp_input_dataDictionary.copy()
		missing_values_list=[]
		
		imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(0), num_op_output=Operation(1),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'avg_income', field_out = 'avg_income')
		
		missing_values_list=[]
		
		imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(0), num_op_output=Operation(1),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'distance', field_out = 'distance')
		
		imputeByNumericOp_output_dataDictionary=imputeByNumericOp_input_dataDictionary_transformed
		imputeByNumericOp_output_dataDictionary.to_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv')
		imputeByNumericOp_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		imputeByNumericOp_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_input_dataDictionary.csv', sep=',')
	
		imputeByNumericOp_input_dataDictionary_transformed=imputeByNumericOp_input_dataDictionary.copy()
		missing_values_list=[]
		
		imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(0), num_op_output=Operation(0),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'satscore', field_out = 'satscore')
		
		imputeByNumericOp_output_dataDictionary=imputeByNumericOp_input_dataDictionary_transformed
		imputeByNumericOp_output_dataDictionary.to_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv')
		imputeByNumericOp_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		rowFilter_input_DataDictionary=pd.read_csv('./knime_dataDictionaries/missing_output_dataDictionary.csv', sep=',')
	
		rowFilter_input_DataDictionary_transformed=rowFilter_input_DataDictionary.copy()
		columns_rowFilterRange_param_filter=['init_span']
		
		filter_range_left_values_list_rowFilterRange_param_filter=[-np.inf]
		filter_range_right_values_list_rowFilterRange_param_filter=[0.0]
		closure_type_list_rowFilterRange_param_filter=[Closure(3)]
		
		rowFilter_input_DataDictionary_transformed=data_transformations.transform_filter_rows_range(data_dictionary=rowFilter_input_DataDictionary_transformed,
																												columns=columns_rowFilterRange_param_filter,
																												left_margin_list=filter_range_left_values_list_rowFilterRange_param_filter,
																												right_margin_list=filter_range_right_values_list_rowFilterRange_param_filter,
																												filter_type=FilterType(0),
																												closure_type_list=closure_type_list_rowFilterRange_param_filter)
		rowFilterRange_output_DataDictionary=rowFilter_input_DataDictionary_transformed
		rowFilterRange_output_DataDictionary.to_csv('./knime_dataDictionaries/rowFilter_output_dataDictionary.csv')
		rowFilterRange_output_DataDictionary=pd.read_csv('./knime_dataDictionaries/rowFilter_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		columnFilter_input_DataDictionary=pd.read_csv('./knime_dataDictionaries/rowFilter_output_dataDictionary.csv', sep=',')
	
		columnFilter_input_DataDictionary_transformed=columnFilter_input_DataDictionary.copy()
		field_list_columnFilter_param_field=['TRAVEL_INIT_CNTCTS', 'REFERRAL_CNTCTS']
		
		columnFilter_input_DataDictionary_transformed=data_transformations.transform_filter_columns(data_dictionary=columnFilter_input_DataDictionary_transformed,
																		columns=field_list_columnFilter_param_field, belong_op=Belong.BELONG)
		
		columnFilter_output_DataDictionary=columnFilter_input_DataDictionary_transformed
		columnFilter_output_DataDictionary.to_csv('./knime_dataDictionaries/columnFilter_output_dataDictionary.csv')
		columnFilter_output_DataDictionary=pd.read_csv('./knime_dataDictionaries/columnFilter_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		mapping_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/columnFilter_output_dataDictionary.csv', sep=',')
	
		input_values_list=['A', 'N']
		output_values_list=['0', '0']
		data_type_input_list=[DataType(0), DataType(0)]
		data_type_output_list=[DataType(0), DataType(0)]
		
		
		mapping_output_dataDictionary=data_transformations.transform_fix_value_fix_value(data_dictionary=mapping_input_dataDictionary, input_values_list=input_values_list,
																	  output_values_list=output_values_list,
								                                      data_type_input_list = data_type_input_list,
								                                      data_type_output_list = data_type_output_list, field_in = 'TERRITORY', field_out = 'TERRITORY')
		
		mapping_output_dataDictionary.to_csv('./knime_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv')	
		mapping_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		mapping_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv', sep=',')
	
		input_values_list=['Y', 'N']
		output_values_list=['1', '0']
		data_type_input_list=[DataType(0), DataType(0)]
		data_type_output_list=[DataType(0), DataType(0)]
		
		
		mapping_output_dataDictionary=data_transformations.transform_fix_value_fix_value(data_dictionary=mapping_input_dataDictionary, input_values_list=input_values_list,
																	  output_values_list=output_values_list,
								                                      data_type_input_list = data_type_input_list,
								                                      data_type_output_list = data_type_output_list, field_in = 'Instate', field_out = 'Instate')
		
		mapping_output_dataDictionary.to_csv('./knime_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv')	
		mapping_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		categoricalToContinuous_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv', sep=',')
	
		categoricalToContinuous_input_dataDictionary_transformed=categoricalToContinuous_input_dataDictionary.copy()
		categoricalToContinuous_input_dataDictionary_transformed=data_transformations.transform_cast_type(data_dictionary=categoricalToContinuous_input_dataDictionary_transformed,
																		data_type_output= DataType(6),
																		field='TERRITORY')
		
		categoricalToContinuous_input_dataDictionary_transformed=data_transformations.transform_cast_type(data_dictionary=categoricalToContinuous_input_dataDictionary_transformed,
																		data_type_output= DataType(6),
																		field='Instate')
		
		categoricalToContinuous_output_dataDictionary=categoricalToContinuous_input_dataDictionary_transformed
		categoricalToContinuous_output_dataDictionary.to_csv('./knime_dataDictionaries/stringToNumber_output_dataDictionary.csv')
		categoricalToContinuous_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/stringToNumber_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		imputeByNumericOp_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/stringToNumber_output_dataDictionary.csv', sep=',')
	
		imputeByNumericOp_input_dataDictionary_transformed=imputeByNumericOp_input_dataDictionary.copy()
		missing_values_list=[]
		
		imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'avg_income', field_out = 'avg_income')
		
		missing_values_list=[]
		
		imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'distance', field_out = 'distance')
		
		missing_values_list=[]
		
		imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
																	  missing_values=missing_values_list,		
																	  axis_param=0, field_in = 'Instate', field_out = 'Instate')
		
		imputeByNumericOp_output_dataDictionary=imputeByNumericOp_input_dataDictionary_transformed
		imputeByNumericOp_output_dataDictionary.to_csv('./knime_dataDictionaries/numericOutliers_output_dataDictionary.csv')
		imputeByNumericOp_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericOutliers_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		binner_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericOutliers_output_dataDictionary.csv', sep=',')
	
		binner_input_dataDictionary_transformed=binner_input_dataDictionary.copy()
		binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																	  data_type_output = DataType(0),
																	  field_in = 'TOTAL_CONTACTS', field_out = 'TOTAL_CONTACTS_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																	  data_type_output = DataType(0),
																	  field_in = 'SELF_INIT_CNTCTS', field_out = 'SELF_INIT_CNTCTS_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																	  data_type_output = DataType(0),
																	  field_in = 'SOLICITED_CNTCTS', field_out = 'SOLICITED_CNTCTS_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=-1000.0, right_margin=1.0,
																	  closure_type=Closure(0),
																	  fix_value_output='Low',
								                                      data_type_output = DataType(0),
																	  field_in = 'TOTAL_CONTACTS',
																	  field_out = 'TOTAL_CONTACTS_binned')
		
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=-1000.0, right_margin=1.0,
																	  closure_type=Closure(0),
																	  fix_value_output='Low',
								                                      data_type_output = DataType(0),
																	  field_in = 'SELF_INIT_CNTCTS',
																	  field_out = 'SELF_INIT_CNTCTS_binned')
		
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=-1000.0, right_margin=1.0,
																	  closure_type=Closure(0),
																	  fix_value_output='Low',
								                                      data_type_output = DataType(0),
																	  field_in = 'SOLICITED_CNTCTS',
																	  field_out = 'SOLICITED_CNTCTS_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=1.0, right_margin=4.0,
																	  closure_type=Closure(2),
																	  fix_value_output='Moderate',
								                                      data_type_output = DataType(0),
																	  field_in = 'TOTAL_CONTACTS',
																	  field_out = 'TOTAL_CONTACTS_binned')
		
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=1.0, right_margin=4.0,
																	  closure_type=Closure(2),
																	  fix_value_output='Moderate',
								                                      data_type_output = DataType(0),
																	  field_in = 'SELF_INIT_CNTCTS',
																	  field_out = 'SELF_INIT_CNTCTS_binned')
		
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=1.0, right_margin=4.0,
																	  closure_type=Closure(2),
																	  fix_value_output='Moderate',
								                                      data_type_output = DataType(0),
																	  field_in = 'SOLICITED_CNTCTS',
																	  field_out = 'SOLICITED_CNTCTS_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=4.0, right_margin=1000.0,
																	  closure_type=Closure(2),
																	  fix_value_output='High',
								                                      data_type_output = DataType(0),
																	  field_in = 'TOTAL_CONTACTS',
																	  field_out = 'TOTAL_CONTACTS_binned')
		
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=4.0, right_margin=1000.0,
																	  closure_type=Closure(2),
																	  fix_value_output='High',
								                                      data_type_output = DataType(0),
																	  field_in = 'SELF_INIT_CNTCTS',
																	  field_out = 'SELF_INIT_CNTCTS_binned')
		
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=4.0, right_margin=1000.0,
																	  closure_type=Closure(2),
																	  fix_value_output='High',
								                                      data_type_output = DataType(0),
																	  field_in = 'SOLICITED_CNTCTS',
																	  field_out = 'SOLICITED_CNTCTS_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		binner_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericOutliers_output_dataDictionary.csv', sep=',')
	
		binner_input_dataDictionary_transformed=binner_input_dataDictionary.copy()
		binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																	  data_type_output = DataType(0),
																	  field_in = 'TERRITORY', field_out = 'TERRITORY_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=-1000.0, right_margin=1.0,
																	  closure_type=Closure(0),
																	  fix_value_output='Unknown',
								                                      data_type_output = DataType(0),
																	  field_in = 'TERRITORY',
																	  field_out = 'TERRITORY_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=1.0, right_margin=3.0,
																	  closure_type=Closure(2),
																	  fix_value_output='Zone 1',
								                                      data_type_output = DataType(0),
																	  field_in = 'TERRITORY',
																	  field_out = 'TERRITORY_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=3.0, right_margin=5.0,
																	  closure_type=Closure(2),
																	  fix_value_output='Zone 2',
								                                      data_type_output = DataType(0),
																	  field_in = 'TERRITORY',
																	  field_out = 'TERRITORY_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=5.0, right_margin=7.0,
																	  closure_type=Closure(2),
																	  fix_value_output='Zone 3',
								                                      data_type_output = DataType(0),
																	  field_in = 'TERRITORY',
																	  field_out = 'TERRITORY_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=7.0, right_margin=1000.0,
																	  closure_type=Closure(3),
																	  fix_value_output='Zone 4',
								                                      data_type_output = DataType(0),
																	  field_in = 'TERRITORY',
																	  field_out = 'TERRITORY_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		binner_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericOutliers_output_dataDictionary.csv', sep=',')
	
		binner_input_dataDictionary_transformed=binner_input_dataDictionary.copy()
		binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																	  data_type_output = DataType(0),
																	  field_in = 'satscore', field_out = 'satscore_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=-1000.0, right_margin=1040.0,
																	  closure_type=Closure(1),
																	  fix_value_output='54 Percentile and Under',
								                                      data_type_output = DataType(0),
																	  field_in = 'satscore',
																	  field_out = 'satscore_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=1040.0, right_margin=1160.0,
																	  closure_type=Closure(0),
																	  fix_value_output='55-75 Percentile',
								                                      data_type_output = DataType(0),
																	  field_in = 'satscore',
																	  field_out = 'satscore_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=1160.0, right_margin=1340.0,
																	  closure_type=Closure(2),
																	  fix_value_output='76-93 Percentile',
								                                      data_type_output = DataType(0),
																	  field_in = 'satscore',
																	  field_out = 'satscore_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=1340.0, right_margin=2000.0,
																	  closure_type=Closure(1),
																	  fix_value_output='94+ percentile',
								                                      data_type_output = DataType(0),
																	  field_in = 'satscore',
																	  field_out = 'satscore_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		
	#-----------------New DataProcessing-----------------
		binner_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericOutliers_output_dataDictionary.csv', sep=',')
	
		binner_input_dataDictionary_transformed=binner_input_dataDictionary.copy()
		binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																	  data_type_output = DataType(0),
																	  field_in = 'avg_income', field_out = 'avg_income_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=9.0, right_margin=42830.0,
																	  closure_type=Closure(1),
																	  fix_value_output='Low',
								                                      data_type_output = DataType(0),
																	  field_in = 'avg_income',
																	  field_out = 'avg_income_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=42830.0, right_margin=55590.0,
																	  closure_type=Closure(1),
																	  fix_value_output='Moderate',
								                                      data_type_output = DataType(0),
																	  field_in = 'avg_income',
																	  field_out = 'avg_income_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																	  left_margin=55590.0, right_margin=100000.0,
																	  closure_type=Closure(2),
																	  fix_value_output='High',
								                                      data_type_output = DataType(0),
																	  field_in = 'avg_income',
																	  field_out = 'avg_income_binned')
		
		binner_output_dataDictionary=binner_input_dataDictionary_transformed
		binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
		binner_output_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
set_logger("transformations")
generateWorkflow()
