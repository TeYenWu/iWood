from tsfresh.feature_extraction import feature_calculators
import pandas as pd
import numpy as np 

### setting tsfresh
WINDOW_SIZE = 512
LAG = int(WINDOW_SIZE/4)
CHUNK_LEN = int(WINDOW_SIZE/8)
ENTROPY_BIN_SIZE = 256

ts_feature_extraction_funcs = [[feature_calculators.abs_energy],
                               [feature_calculators.absolute_maximum],
                               # [feature_calculators.absolute_sum_of_changes],
                               # [feature_calculators.agg_autocorrelation, [{"f_agg":"mean", "maxlag":LAG}]], 
                               # [feature_calculators.agg_linear_trend, [{"attr": "rvalue", "chunk_len": CHUNK_LEN, "f_agg": "mean"}]], 
                               # [feature_calculators.benford_correlation], 
                               # [feature_calculators.binned_entropy, ENTROPY_BIN_SIZE],
                               # [feature_calculators.c3, LAG],
                               # [feature_calculators.cid_ce, True],
                               # [feature_calculators.count_above, 0],
                               # [feature_calculators.count_above_mean],
                               # [feature_calculators.count_below, 0],
                               # [feature_calculators.count_below_mean],
                               # [feature_calculators.number_cwt_peaks, WINDOW_SIZE],
                               # [feature_calculators.first_location_of_maximum],
                               # [feature_calculators.has_duplicate_max],
                               # [feature_calculators.kurtosis],
                               [feature_calculators.last_location_of_maximum],
                               # [feature_calculators.last_location_of_minimum],
                               # [feature_calculators.length],
                               # [feature_calculators.longest_strike_above_mean],
                               # [feature_calculators.longest_strike_below_mean],
                               [feature_calculators.maximum],
                               [feature_calculators.mean],
                               [feature_calculators.mean_abs_change],
                               [feature_calculators.mean_change],
                               # [feature_calculators.mean_n_absolute_max],
                               [feature_calculators.mean_second_derivative_central],
                               [feature_calculators.median],
                               [feature_calculators.minimum],
                               # [feature_calculators.number_crossing_m, 0],
                               # [feature_calculators.number_peaks, 10],
                               # [feature_calculators.partial_autocorrelation, [{"lag": LAG}]],
                               # [feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints],
                               # [feature_calculators.percentage_of_reoccurring_values_to_all_values],
                               # [feature_calculators.ratio_beyond_r_sigma, 2],
                               # [feature_calculators.skewness],
                               # [feature_calculators.time_reversal_asymmetry_statistic, LAG]
                               ]

def interpret_result(result):
	if isinstance(result, list):
		return [result[0][1]]
	elif isinstance(result, zip):
		return [pd.Series(result).tolist()[0][1]]
	return [result]

def calculate_tsfresh_features(sig):
	features = []
	for f_list in ts_feature_extraction_funcs:
		if len(f_list) == 1:
			f = f_list[0]
			r = f(sig)
			features = np.concatenate([features, interpret_result(r)])
		elif len(f_list) == 2:
			f = f_list[0]
			x1 = f_list[1]
			r = f(sig, x1)
			features = np.concatenate([features, interpret_result(r)])
		elif len(f_list) == 5:
			f = f_list[0]
			x1 = f_list[1]
			x2 = f_list[2] if len(f_list) >= 3 else None
			x3 = f_list[3] if len(f_list) >= 4 else None
			x4 = f_list[4] if len(f_list) >= 5 else None
			r = f(sig, x1, x2, x3, x4)
			features = np.concatenate([features, interpret_result(r)])
	return features
			

