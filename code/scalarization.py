import processed_data_10_10_10
import scalarization_lib as syn


# imagery_training
# print("imagery_training")
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt",
#                     0)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/median.txt",
#                     1)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/std.txt",
#                     2)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/var.txt",
#                     3)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max_min_distance.txt",
#                     4)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantiles_distance.txt",
#                     5)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max.txt",
#                     6)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/min.txt",
#                     7)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_1.txt",
#                     8)
# syn.scalarization_3(processed_data_10_10_10.imagery_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_2.txt",
#                     9)
# imagery_test
# print("imagery_test")
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt",
#                     0)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/median.txt",
#                     1)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/std.txt",
#                     2)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/var.txt",
#                     3)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max_min_distance.txt",
#                     4)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantiles_distance.txt",
#                     5)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max.txt",
#                     6)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/min.txt",
#                     7)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_1.txt",
#                     8)
# syn.scalarization_3(processed_data_10_10_10.imagery_test,
#                     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_2.txt",
#                     9)
# perception_training
# print("perception_training")
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt",
#                     0)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/median.txt",
#                     1)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/std.txt",
#                     2)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/var.txt",
#                     3)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max_min_distance.txt",
#                     4)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantiles_distance.txt",
#                     5)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max.txt",
#                     6)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/min.txt",
#                     7)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_1.txt",
#                     8)
# syn.scalarization_3(processed_data_10_10_10.perception_training,
#                     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_2.txt",
#                     9)
# perception_test
print("perception_test")
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt",
                    0)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/median.txt",
                    1)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/std.txt",
                    2)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/var.txt",
                    3)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max_min_distance.txt",
                    4)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantiles_distance.txt",
                    5)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max.txt",
                    6)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/min.txt",
                    7)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_1.txt",
                    8)
syn.scalarization_3(processed_data_10_10_10.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_2.txt",
                    9)
