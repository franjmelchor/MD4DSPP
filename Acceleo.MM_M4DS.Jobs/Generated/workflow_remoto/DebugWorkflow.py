print(impute_mostFrequent(sex, IRISCHOOL, ETHNICITY))	#DataProcessing name
print('NO SOURCE')         					#SOURCE
print('impute_fix_value(ACADEMIC_INTEREST_2, ACADEMIC_INTEREST_1)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_impute_in')         #INPUTPORT name
print('5160')         #INPUTPORT rows
print('../data_model.csv')         #INPUTPORT path
print('def_impute_categorical_columns_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('sex_in_cols_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('sex')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_columns_input_dataField')

	print('IRSCHOOL_in_cols_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('IRSCHOOL')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_columns_input_dataField')

	print('ETHNICITY_in_cols_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('ETHNICITY')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_columns_input_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_impute_out')         #OUTPUTPORT name
print('5160')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_impute_out.csv')         #OUTPUTPORT path
print('def_impute_categorical_columns_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('sex_out_cols_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('sex')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_columns_output_dataField')

	print('IRSCHOOL_out_cols_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('IRSCHOOL')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_columns_output_dataField')

	print('ETHNICITY_out_cols_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('ETHNICITY')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_columns_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_derivedValue_impute_mostFrequent')			#DerivedValue
	print('Missing')
	print('MostFrequent')
	print('def_parameter_impute_categorical_columns_der_value')
#-----------------------------------------------------------------------------
print('PRECONDITION_impute_sex_columns')
print('PRECONDITION_impute_IRSCHOOL_columns')
print('PRECONDITION_impute_ETHNICITY_columns')
print('POSTCONDITION_impute_sex_columns')
print('POSTCONDITION_impute_IRSCHOOL_columns')
print('POSTCONDITION_impute_ETHNICITY_columns')
print('INVARIANT_impute_sex_columns')
print('INVARIANT_impute_IRSCHOOL_columns')
print('INVARIANT_impute_ETHNICITY_columns')
############################################################
print(deprecated_impute(sex))	#DataProcessing name
print('NO SOURCE')         					#SOURCE
print('NO TARGET')         					#TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_impute_sex_in')         #INPUTPORT name
print('5160')         #INPUTPORT rows
print('../data_model.csv')         #INPUTPORT path
print('def_impute_categorical_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('sex_in')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('sex')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_input_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_impute_sex_out')         #OUTPUTPORT name
print('5160')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_impute_out.csv')         #OUTPUTPORT path
print('def_impute_categorical_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('sex_out')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('sex')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_impute_sex')			#DerivedValue
	print('Missing')
	print('MostFrequent')
	print('def_parameter_impute_categorical_col_der_value')
#-----------------------------------------------------------------------------
print('PRECONDITION_impute_sex')
print('POSTCONDITION_impute_sex')
print('INVARIANT_impute_sex')
############################################################
print(deprecated_impute(IRSCHOOL))	#DataProcessing name
print('NO SOURCE')         					#SOURCE
print('NO TARGET')         					#TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_impute_IRSCHOOL_in')         #INPUTPORT name
print('5160')         #INPUTPORT rows
print('../data_model.csv')         #INPUTPORT path
print('def_impute_categorical_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('IRSCHOOL_in')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('IRSCHOOL')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_input_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_impute_IRSCHOOL_out')         #OUTPUTPORT name
print('5160')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_impute_out.csv')         #OUTPUTPORT path
print('def_impute_categorical_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('IRSCHOOL_out')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('IRSCHOOL')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_impute_IRSCHOOL')			#DerivedValue
	print('Missing')
	print('MostFrequent')
	print('def_parameter_impute_categorical_col_der_value')
#-----------------------------------------------------------------------------
print('PRECONDITION_impute_IRSCHOOL')
print('POSTCONDITION_impute_IRSCHOOL')
print('INVARIANT_impute_IRSCHOOL')
############################################################
print(deprecated_impute(ETHNICITY))	#DataProcessing name
print('NO SOURCE')         					#SOURCE
print('NO TARGET')         					#TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_impute_ETHNICITY_in')         #INPUTPORT name
print('5160')         #INPUTPORT rows
print('../data_model.csv')         #INPUTPORT path
print('def_impute_categorical_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('ETHNICITY_in')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('ETHNICITY')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_input_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_impute_ETHNICITY_out')         #OUTPUTPORT name
print('5160')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_impute_out.csv')         #OUTPUTPORT path
print('def_impute_categorical_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('ETHNICITY_out')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('ETHNICITY')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_impute_ETHNICITY')			#DerivedValue
	print('Missing')
	print('MostFrequent')
	print('def_parameter_impute_categorical_col_der_value')
#-----------------------------------------------------------------------------
print('PRECONDITION_impute_ETHNICITY')
print('POSTCONDITION_impute_ETHNICITY')
print('INVARIANT_impute_ETHNICITY')
############################################################
print(impute_fix_value(ACADEMIC_INTEREST_2, ACADEMIC_INTEREST_1))	#DataProcessing name
print('impute_mostFrequent(sex, IRISCHOOL, ETHNICITY)')         #SOURCE
print('impute_mean(avg_income, distance)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_impute_ACADEMIC_INTEREST_2_in')         #INPUTPORT name
print('5160')         #INPUTPORT rows
print('../data_model.csv')         #INPUTPORT path
print('def_impute_categorical_fix_value_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('ACADEMIC_INTEREST_2_in_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('ACADEMIC_INTEREST_2')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_fix_value_input_dataField')

	print('ACADEMIC_INTEREST_1_in_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('ACADEMIC_INTEREST_1')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_fix_value_input_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_impute_ACADEMIC_INTEREST_2_out')         #OUTPUTPORT name
print('5160')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_impute_out.csv')         #OUTPUTPORT path
print('def_impute_categorical_fix_value_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('ACADEMIC_INTEREST_2_out_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('ACADEMIC_INTEREST_2')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_fix_value_output_dataField')

	print('ACADEMIC_INTEREST_1_out_impute')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('ACADEMIC_INTEREST_1')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_fix_value_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_fixValue_impute_fixValue')			#FixValue
	print('Missing')
	print('Unknown')			#Es unknown porque el valor concreto no se ha definido, se define en la llamada
	print('def_parameter_impute_categorical_col_fix_value')
#-----------------------------------------------------------------------------
print('PRECONDITION_impute_ACADEMIC_INTEREST_2')
print('POSTCONDITION_impute_ACADEMIC_INTEREST_2')
print('INVARIANT_impute_ACADEMIC_INTEREST_2')
print('PRECONDITION_impute_ACADEMIC_INTEREST_1')
print('POSTCONDITION_impute_ACADEMIC_INTEREST_1')
print('INVARIANT_impute_ACADEMIC_INTEREST_1')
############################################################
print(impute_mean(avg_income, distance))	#DataProcessing name
print('impute_fix_value(ACADEMIC_INTEREST_2, ACADEMIC_INTEREST_1)')         #SOURCE
print('impute_linear_interpolation(satscore)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_impute_mean_in')         #INPUTPORT name
print('5160')         #INPUTPORT rows
print('../data_model.csv')         #INPUTPORT path
print('def_impute_cont_columns_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('avg_income_in_impute')         #DataField name
	print('false')         #DataField id
	print('Double')         #DataField dataType
	print('false')         #DataField target
	print('avg_income')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('9.0')		#Interval
		print('202.0')
		print('opeOpen')

	print('distance_in_impute')         #DataField name
	print('false')         #DataField id
	print('Double')         #DataField dataType
	print('false')         #DataField target
	print('distance')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('0.791')		#Interval
		print('3882.192')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_impute_mean_out')         #OUTPUTPORT name
print('5160')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_impute_out.csv')         #OUTPUTPORT path
print('def_impute_cont_columns_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('avg_income_out_impute')         #DataField name
	print('false')         #DataField id
	print('Double')         #DataField dataType
	print('false')         #DataField target
	print('avg_income')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('9.0')		#Interval
		print('202.0')
		print('opeOpen')

	print('distance_out_impute')         #DataField name
	print('false')         #DataField id
	print('Double')         #DataField dataType
	print('false')         #DataField target
	print('distance')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('0.791')		#Interval
		print('3882.192')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_num_op_impute_mean')			#NumOp
	print('Missing')
	print('Mean')
	print('def_impute_cont_columns_param_numOp')
#-----------------------------------------------------------------------------
print('PRECONDITION_impute_mean_avg_income')
print('PRECONDITION_impute_mean_distance')
print('POSTCONDITION_impute_mean_avg_income')
print('POSTCONDITION_impute_mean_distance')
print('INVARIANT_impute_mean_avg_income')
print('INVARIANT_impute_mean_distance')
############################################################
print(impute_linear_interpolation(satscore))	#DataProcessing name
print('impute_mean(avg_income, distance)')         #SOURCE
print('row_filter(init_span)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_impute_linear_interpolation_in')         #INPUTPORT name
print('5160')         #INPUTPORT rows
print('../data_model.csv')         #INPUTPORT path
print('def_impute_cont_columns_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('satscore_in_impute')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('satscore')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('440.0')		#Interval
		print('1600.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_impute_linear_interpolation_out')         #OUTPUTPORT name
print('5160')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_impute_out.csv')         #OUTPUTPORT path
print('def_impute_cont_columns_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('satscore_out_impute')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('satscore')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('440.0')		#Interval
		print('1600.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_num_op_impute_mean')			#NumOp
	print('Missing')
	print('Mean')
	print('def_impute_cont_columns_param_numOp')
#-----------------------------------------------------------------------------
print('PRECONDITION_impute_linear_interpolation_satscore')
print('POSTCONDITION_impute_linear_interpolation_satscore')
print('INVARIANT_impute_linear_interpolation_satscore')
############################################################
print(row_filter(init_span))	#DataProcessing name
print('impute_linear_interpolation(satscore)')         #SOURCE
print('column_cont_filter(TRAVEL_INIT_CNTCTS, REFERRAL_CNCTS, telecq, interest, stuemail)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_row_filter_in')         #INPUTPORT name
print('5160')         #INPUTPORT rows
print('../workflow_datasets/data_model_impute_out.csv')         #INPUTPORT path
print('def_row_filter_input_DataDictionary')         #INPUTPORT dataDictionaryDef name
	print('init_span_in')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('init_span')         #DataField displayName((((()))))
	print('def_row_filter_input_dataField')		#Continuous
		print('0.0')		#Interval
		print('72.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_row_filter_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_row_filter_out.csv')         #OUTPUTPORT path
print('def_row_filter_output_DataDictionary')         #INPUTPORT dataDictionaryDef name
	print('init_span_out')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('init_span')         #DataField displayName((((()))))
	print('def_row_filter_output_dataField')		#Continuous
		print('3.0')		#Interval
		print('72.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_filter_row_filter')			#FilterValue
	print('def_parameter_filter_row_filter')
		print('parameter_primitive_row_filter')			#Primitive
		print('0')
		print('def_parameter_primitive_row_filter')
	print('parameter_primitive_row_filter')			#Primitive
	print('0')
	print('def_parameter_primitive_row_filter')
#-----------------------------------------------------------------------------
print('PRECONDITION_row_filter')
print('POSTCONDITION_row_filter')
############################################################
print(column_cont_filter(TRAVEL_INIT_CNTCTS, REFERRAL_CNCTS, telecq, interest, stuemail))	#DataProcessing name
print('row_filter(init_span)')         #SOURCE
print('column_cat_filter(CONTACT_CODE1)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_column_cont_filter_in')         #INPUTPORT name
print('5159')         #INPUTPORT rows
print('../workflow_datasets/data_model_row_filter_out.csv')         #INPUTPORT path
print('def_column_cont_filter_input_DataDictionary')         #INPUTPORT dataDictionaryDef name
	print('TRAVEL_INIT_CNTCTS_column_filter')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('TRAVEL_INIT_CNTCTS')         #DataField displayName((((()))))
	print('def_input_column_cont_filter')		#Continuous
		print('0.0')		#Interval
		print('5.0')
		print('closedClosed')

	print('REFERRAL_CNCTS_column_filter')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('REFERRAL_CNCTS')         #DataField displayName((((()))))
	print('def_input_column_cont_filter')		#Continuous
		print('0.0')		#Interval
		print('5.0')
		print('closedClosed')

	print('telecq_column_filter')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('telecq')         #DataField displayName((((()))))
	print('def_input_column_cont_filter')		#Continuous
		print('1.0')		#Interval
		print('4.0')
		print('closedClosed')

	print('interest_column_filter')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('interest')         #DataField displayName((((()))))
	print('def_input_column_cont_filter')		#Continuous
		print('0.0')		#Interval
		print('3.0')
		print('closedClosed')

	print('stuemail_column_filter')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('stuemail')         #DataField displayName((((()))))
	print('def_input_column_cont_filter')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_column_cont_filter_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_col_filter_out.csv')         #OUTPUTPORT path
print('def_column_cont_filter_output_DataDictionary')         #INPUTPORT dataDictionaryDef name
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	print('parameter_column_cont_filter')			#Field
	print('BELONG')
	print('def_field_parameter_column_cont_filter')
		print('TRAVEL_INIT_CNTCTS_column_filter')		#DataField
		print('false')
		print('Integer')
		print('false')
		print('TRAVEL_INIT_CNTCTS')
			print('def_input_column_cont_filter')			#Continuous
				print('0.0')			#Interval
				print('5.0')
				print('closedClosed')
			#-----------------------------------
		print('REFERRAL_CNCTS_column_filter')		#DataField
		print('false')
		print('String')
		print('false')
		print('REFERRAL_CNCTS')
			print('def_input_column_cont_filter')			#Continuous
				print('0.0')			#Interval
				print('5.0')
				print('closedClosed')
			#-----------------------------------
#-----------------------------------------------------------------------------
print('PRECONDITION_column_cont_filter')
print('POSTCONDITION_column_cont_filter')
############################################################
print(column_cat_filter(CONTACT_CODE1))	#DataProcessing name
print('column_cont_filter(TRAVEL_INIT_CNTCTS, REFERRAL_CNCTS, telecq, interest, stuemail)')         #SOURCE
print('map(TERRITORY)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_column_cat_filter_in')         #INPUTPORT name
print('5159')         #INPUTPORT rows
print('../workflow_datasets/data_model_row_filter_out.csv')         #INPUTPORT path
print('def_column_cat_filter_input_DataDictionary')         #INPUTPORT dataDictionaryDef name
	print('CONTACT_CODE1')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('CONTACT_CODE1')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_column_cat_filter_input_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_column_cat_filter_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_col_filter_out.csv')         #OUTPUTPORT path
print('def_column_cat_filter_output_DataDictionary')         #INPUTPORT dataDictionaryDef name
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	print('parameter_column_cat_filter')			#Field
	print('BELONG')
	print('def_field_parameter_column_cat_filter')
		print('CONTACT_CODE1')		#DataField
		print('false')
		print('String')
		print('false')
		print('CONTACT_CODE1')
			print('false')			#Categorical
			print('def_column_cat_filter_input_dataField')
			#-----------------------------------
#-----------------------------------------------------------------------------
print('PRECONDITION_column_cat_filter')
print('POSTCONDITION_column_cat_filter')
############################################################
print(map(TERRITORY))	#DataProcessing name
print('column_cat_filter(CONTACT_CODE1)')         #SOURCE
print('map(Instate)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_map_territory_in')         #INPUTPORT name
print('5159')         #INPUTPORT rows
print('../workflow_datasets/data_model_col_filter_out.csv')         #INPUTPORT path
print('def_map_categorical_col_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('TERRITORY_IN_map')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('TERRITORY')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_map_categorical_col_input_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_map_territory_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_map_territory_out.csv')         #OUTPUTPORT path
print('def_map_categorical_col_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('TERRITORY_OUT_map')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('TERRITORY')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_map_territory_A')			#Map
	print('0')
	print('def_parameter_map_categorical_col')
	print('parameter_map_territory_N')			#Map
	print('0')
	print('def_parameter_map_categorical_col')
#-----------------------------------------------------------------------------
print('PRECONDITION_map_territory')
print('POSTCONDITION_map_territory')
print('INVARIANT_map_categorical_col')
############################################################
print(map(Instate))	#DataProcessing name
print('map(TERRITORY)')         #SOURCE
print('stringToNumber (TERRITORY, init1rat, init2rat, hscrat, Instate)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_map_Instate_in')         #INPUTPORT name
print('5159')         #INPUTPORT rows
print('../workflow_datasets/data_model_map_territory_out.csv')         #INPUTPORT path
print('def_map_categorical_col_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('Instate_in_map')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('Instate')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_map_categorical_col_input_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_map_Instate_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_map_instate_out.csv')         #OUTPUTPORT path
print('def_map_categorical_col_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('Instate_out_map')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('Instate')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_impute_categorical_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_map_instate_Y')			#Map
	print('1')
	print('def_parameter_map_categorical_col')
	print('parameter_map_Instate_N')			#Map
	print('0')
	print('def_parameter_map_categorical_col')
#-----------------------------------------------------------------------------
print('PRECONDITION_map_Instate')
print('POSTCONDITION_map_Instate')
print('INVARIANT_map_Instate')
############################################################
print(stringToNumber (TERRITORY, init1rat, init2rat, hscrat, Instate))	#DataProcessing name
print('map(Instate)')         #SOURCE
print('impute_outliers_closest(avg_income, distance, premiere, sex, Enroll, Instate)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_stringToNumber_in')         #INPUTPORT name
print('5159')         #INPUTPORT rows
print('../workflow_datasets/data_model_map_instate_out.csv')         #INPUTPORT path
print('def_input_stringToNumber')         #INPUTPORT dataDictionaryDef name
	print('TERRITORY_in')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('TERRITORY')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_stringToNumber_input_datafield')

	print('init1rat_in')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('init1rat')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_stringToNumber_input_datafield')

	print('init2rat_in')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('init2rat')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_stringToNumber_input_datafield')

	print('hscrat_in')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('hscrat')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_stringToNumber_input_datafield')

	print('Instate_in')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('Instate')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_stringToNumber_input_datafield')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_stringToNumber_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_stringToNumber_out.csv')         #OUTPUTPORT path
print('def_output_stringToNumber')         #INPUTPORT dataDictionaryDef name
	print('TERRITORY_out')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('TERRITORY')         #DataField displayName((((()))))
	print('def_stringToNumber_output_datafield')		#Continuous
		print('0.0')		#Interval
		print('8.0')
		print('closedClosed')

	print('init1rat_out')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('init1rat')         #DataField displayName((((()))))
	print('def_stringToNumber_output_datafield')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('init2rat_out')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('init2rat')         #DataField displayName((((()))))
	print('def_stringToNumber_output_datafield')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('hscrat_out')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('hscrat')         #DataField displayName((((()))))
	print('def_stringToNumber_output_datafield')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('Instate_out')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('Instate')         #DataField displayName((((()))))
	print('def_stringToNumber_output_datafield')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#-----------------------------------------------------------------------------
print('INVARIANT_stringToNumber_TERRITORY')
print('INVARIANT_stringToNumber_init1rat')
print('INVARIANT_stringToNumber_init2rat')
print('INVARIANT_stringToNumber_hscrat')
print('INVARIANT_stringToNumber_Instate')
############################################################
print(impute_outliers_closest(avg_income, distance, premiere, sex, Enroll, Instate))	#DataProcessing name
print('stringToNumber (TERRITORY, init1rat, init2rat, hscrat, Instate)')         #SOURCE
print('binner (TOTAL_CONTACTS, SELF_INIT_CNTCTS, SOLICITED_CNTCTS)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_impute_outlier_closest_in')         #INPUTPORT name
print('5159')         #INPUTPORT rows
print('../workflow_datasets/data_model_stringToNumber_in.csv')         #INPUTPORT path
print('def_impute_cont_columns_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('avg_income_in_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('Double')         #DataField dataType
	print('false')         #DataField target
	print('avg_income')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('9.0')		#Interval
		print('202.0')
		print('opeOpen')

	print('distance_in_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('Double')         #DataField dataType
	print('false')         #DataField target
	print('distance')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('0.791')		#Interval
		print('3882.192')
		print('closedClosed')

	print('premiere_in_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('premiere')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('sex_in_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('sex')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('Enroll_in_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('Enroll')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('Instate_in_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('Instate')         #DataField displayName((((()))))
	print('def_impute_cont_columns_input_dataField')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_impute_outlier_closest_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_imputeOutliers_out.csv')         #OUTPUTPORT path
print('def_impute_cont_columns_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('avg_income_out_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('Double')         #DataField dataType
	print('false')         #DataField target
	print('avg_income')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('9.0')		#Interval
		print('202.0')
		print('opeOpen')

	print('distance_out_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('Double')         #DataField dataType
	print('false')         #DataField target
	print('distance')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('0.791')		#Interval
		print('3882.192')
		print('closedClosed')

	print('premiere_out_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('premiere')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('sex_out_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('sex')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('Enroll_out_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('Enroll')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

	print('Instate_out_impute_outliers')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('Instate')         #DataField displayName((((()))))
	print('def_impute_cont_columns_output_dataField')		#Continuous
		print('0.0')		#Interval
		print('1.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('parameter_num_op_impute_mean')			#NumOp
	print('Outlier')
	print('Closest')
	print('def_impute_cont_columns_param_numOp')
#-----------------------------------------------------------------------------
print('PRECONDITION_impute_outliers_closest_avg_income')
print('PRECONDITION_impute_outliers_closest_distance')
print('PRECONDITION_impute_outliers_closest_premiere')
print('PRECONDITION_impute_outliers_closest_sex')
print('PRECONDITION_impute_outliers_closest_Enroll')
print('PRECONDITION_impute_outliers_closest_Instate')
print('POSTCONDITION_impute_outliers_closest_avg_income')
print('POSTCONDITION_impute_outliers_closest_distance')
print('POSTCONDITION_impute_outliers_closest_premiere')
print('POSTCONDITION_impute_outliers_closest_sex')
print('POSTCONDITION_impute_outliers_closest_Enroll')
print('POSTCONDITION_impute_outliers_closest_Instate')
print('INVARIANT_impute_outliers_closest_avg_income')
print('INVARIANT_impute_outliers_closest_distance')
print('INVARIANT_impute_outliers_closest_premiere')
print('INVARIANT_impute_outliers_closest_sex')
print('INVARIANT_impute_outliers_closest_Enroll')
print('INVARIANT_impute_outliers_closest_Instate')
############################################################
print(binner (TOTAL_CONTACTS, SELF_INIT_CNTCTS, SOLICITED_CNTCTS))	#DataProcessing name
print('impute_outliers_closest(avg_income, distance, premiere, sex, Enroll, Instate)')         #SOURCE
print('binner (TERRITORY)')         #TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_binner_in')         #INPUTPORT name
print('5159')         #INPUTPORT rows
print('../workflow_datasets/data_model_stringToNumber_in.csv')         #INPUTPORT path
print('def_binner_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('binner_TOTAL_CONTACTS_in')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('TOTAL_CONTACTS')         #DataField displayName((((()))))
	print('def_binner_input_dataField')		#Continuous
		print('1.0')		#Interval
		print('28.0')
		print('opeOpen')

	print('binner_SELF_INIT_CNTCTS_in')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('SELF_INIT_CNTCTS')         #DataField displayName((((()))))
	print('def_binner_input_dataField')		#Continuous
		print('0.0')		#Interval
		print('21.0')
		print('opeOpen')

	print('binner_SOLICITED_CNTCTS_in')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('SOLICITED_CNTCTS')         #DataField displayName((((()))))
	print('def_binner_input_dataField')		#Continuous
		print('0.0')		#Interval
		print('9.0')
		print('closedClosed')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_binner_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_binner_out.csv')         #OUTPUTPORT path
print('def_binner_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('binner_TOTAL_CONTACTS_out')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('TOTAL_CONTACTS_binned')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_binner_output_dataField')

	print('binner_SELF_INIT_CNTCTS_out')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('SELF_INIT_CNTCTS_binned')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_binner_output_dataField')

	print('binner_SOLICITED_CNTCTS_out')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('SOLICITED_CNTCTS_binned')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_binner_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('binner_parameter_derivedField_TOTAL_CONTACTS')			#DerivedField
	print('TOTAL_CONTACTS_binned')
	print('String')
	print('def_binner_parameter_derived_field')
	print('binner_TOTAL_CONTACTS_out')
	print('false')
	print('String')
	print('false')
	print('TOTAL_CONTACTS_binned')						#DISPLAY_NAME
	print('binner_parameter_derivedField_SELF_INIT_CNTCTS')			#DerivedField
	print('SELF_INIT_CNTCTS_binned')
	print('String')
	print('def_binner_parameter_derived_field')
	print('binner_SELF_INIT_CNTCTS_out')
	print('false')
	print('String')
	print('false')
	print('SELF_INIT_CNTCTS_binned')						#DISPLAY_NAME
	print('binner_parameter_derivedField_SOLICITED_CNTCTS')			#DerivedField
	print('SOLICITED_CNTCTS_binned')
	print('String')
	print('def_binner_parameter_derived_field')
	print('binner_SOLICITED_CNTCTS_out')
	print('false')
	print('String')
	print('false')
	print('SOLICITED_CNTCTS_binned')						#DISPLAY_NAME
	print('binner_parameter_discretize_bin_Low')		#DiscretizeBin
	print('low')
	print('def_binner_parameter_bin')
		print('-1000.0')		#Interval
		print('1.0')
		print('opeOpen')
	print('binner_parameter_discretize_bin_Moderate')		#DiscretizeBin
	print('Moderate')
	print('def_binner_parameter_bin')
		print('1.0')		#Interval
		print('4.0')
		print('closedOpen')
	print('binner_parameter_discretize_bin_High')		#DiscretizeBin
	print('High')
	print('def_binner_parameter_bin')
		print('4.0')		#Interval
		print('1000.0')
		print('closedOpen')
#-----------------------------------------------------------------------------
print('PRECONDITION_binner_TOTAL_CONTACTS')
print('PRECONDITION_binner_SELF_INIT_CNTCTS')
print('PRECONDITION_binner_SOLICITED_CNTCTS')
print('POSTCONDITION_binner_TOTAL_CONTACTS')
print('POSTCONDITION_binner_SELF_INIT_CNTCTS')
print('POSTCONDITION_binner_SOLICITED_CNTCTS')
print('INVARIANT_binner_TOTAL_CONTACTS')
print('INVARIANT_binner_SELF_INIT_CNTCTS')
print('INVARIANT_binner_SOLICITED_CNTCTS')
############################################################
print(binner (TERRITORY))	#DataProcessing name
print('binner (TOTAL_CONTACTS, SELF_INIT_CNTCTS, SOLICITED_CNTCTS)')         #SOURCE
print('NO TARGET')         					#TARGET
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('data_model_binner_in')         #INPUTPORT name
print('5159')         #INPUTPORT rows
print('../workflow_datasets/data_model_stringToNumber_in.csv')         #INPUTPORT path
print('def_binner_input_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('binner_TERRITORY_in')         #DataField name
	print('false')         #DataField id
	print('Integer')         #DataField dataType
	print('false')         #DataField target
	print('TERRITORY')         #DataField displayName((((()))))
	print('def_binner_input_dataField')		#Continuous
		print('0.0')		#Interval
		print('8.0')
		print('opeOpen')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('data_model_binner_out')         #OUTPUTPORT name
print('5159')         #OUTPUTPORT rows
print('../workflow_datasets/data_model_binner_out.csv')         #OUTPUTPORT path
print('def_binner_output_dataDictionary')         #INPUTPORT dataDictionaryDef name
	print('binner_TERRITORY_out')         #DataField name
	print('false')         #DataField id
	print('String')         #DataField dataType
	print('false')         #DataField target
	print('TERRITORY_binned')         #DataField displayName((((()))))
	print('false')		#Categorical
	print('def_binner_output_dataField')

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	print('binner_parameter_derivedField_TERRITORY_binned')			#DerivedField
	print('TERRITORY_binned')
	print('String')
	print('def_binner_parameter_derived_field')
	print('binner_TERRITORY_out')
	print('false')
	print('String')
	print('false')
	print('TERRITORY_binned')						#DISPLAY_NAME
	print('binner_parameter_discretize_bin_Unknown')		#DiscretizeBin
	print('Unknown')
	print('def_binner_parameter_bin')
		print('-1000.0')		#Interval
		print('1.0')
		print('opeOpen')
	print('binner_parameter_discretize_bin_Zone1')		#DiscretizeBin
	print('Zone 1')
	print('def_binner_parameter_bin')
		print('1.0')		#Interval
		print('3.0')
		print('closedOpen')
	print('binner_parameter_discretize_bin_Zone2')		#DiscretizeBin
	print('Zone 2')
	print('def_binner_parameter_bin')
		print('3.0')		#Interval
		print('5.0')
		print('closedOpen')
	print('binner_parameter_discretize_bin_Zone3')		#DiscretizeBin
	print('Zone 3')
	print('def_binner_parameter_bin')
		print('5.0')		#Interval
		print('7.0')
		print('closedOpen')
	print('binner_parameter_discretize_bin_Zone4')		#DiscretizeBin
	print('Zone 4')
	print('def_binner_parameter_bin')
		print('7.0')		#Interval
		print('1000.0')
		print('closedClosed')
#-----------------------------------------------------------------------------
print('PRECONDITION_binner_TERRITORY')
print('POSTCONDITION_binner_TERRITORY')
print('INVARIANT_binner_TERRITORY')
############################################################








