import pandas as pd
import functions.contract_invariants as contract_invariants
import functions.contract_pre_post as contract_pre_post
from helpers.enumerations import Belong, Operator

class DataProcessing:
	def generateDataProcessing(self):
		pre_post=contract_pre_post.ContractsPrePost()
		invariants=contract_invariants.ContractsInvariants()
#-----------------New DataProcessing-----------------
		data_model_impute_in=pd.read_csv('../data_model.csv')
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, 
										field='sex', quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
		
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, 
										field='IRSCHOOL', quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
		
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, 
										field='ETHNICITY', quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_sex_in=pd.read_csv('../data_model.csv')
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_sex_in, 
										field='sex', quant_op=Operator(2), quant_rel=70.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_IRSCHOOL_in=pd.read_csv('../data_model.csv')
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_IRSCHOOL_in, 
										field='IRSCHOOL', quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_ETHNICITY_in=pd.read_csv('../data_model.csv')
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_ETHNICITY_in, 
										field='ETHNICITY', quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_ACADEMIC_INTEREST_2_in=pd.read_csv('../data_model.csv')
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_ACADEMIC_INTEREST_2_in, 
										field='ACADEMIC_INTEREST_2', quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
		
		
		
		
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_ACADEMIC_INTEREST_2_in, 
										field='ACADEMIC_INTEREST_1', quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_mean_in=pd.read_csv('../data_model.csv')
		
		
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_linear_interpolation_in=pd.read_csv('../data_model.csv')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_row_filter_in=pd.read_csv('../workflow_datasets/data_model_impute_out.csv')
		
		
#-----------------New DataProcessing-----------------
		data_model_column_cont_filter_in=pd.read_csv('../workflow_datasets/data_model_row_filter_out.csv')
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_column_cat_filter_in=pd.read_csv('../workflow_datasets/data_model_row_filter_out.csv')
		
		
#-----------------New DataProcessing-----------------
		data_model_map_territory_in=pd.read_csv('../workflow_datasets/data_model_col_filter_out.csv')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_map_Instate_in=pd.read_csv('../workflow_datasets/data_model_map_territory_out.csv')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_stringToNumber_in=pd.read_csv('../workflow_datasets/data_model_map_instate_out.csv')
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_outlier_closest_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_binner_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_binner_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
		
		
		
dp=DataProcessing()
dp.generateDataProcessing()
