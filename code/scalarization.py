import processed_data
import synolitic as syn

syn.scalarization_3(processed_data.imagery_training,
                    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max.txt",
                    6)
syn.scalarization_3(processed_data.imagery_training,
                    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/min.txt",
                    7)
syn.scalarization_3(processed_data.imagery_training,
                    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_1.txt",
                    8)
syn.scalarization_3(processed_data.imagery_training,
                    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_2.txt",
                    9)

syn.scalarization_3(processed_data.imagery_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max.txt",
                    6)
syn.scalarization_3(processed_data.imagery_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/min.txt",
                    7)
syn.scalarization_3(processed_data.imagery_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_1.txt",
                    8)
syn.scalarization_3(processed_data.imagery_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_2.txt",
                    9)

syn.scalarization_3(processed_data.perception_training,
                    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max.txt",
                    6)
syn.scalarization_3(processed_data.perception_training,
                    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/min.txt",
                    7)
syn.scalarization_3(processed_data.perception_training,
                    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_1.txt",
                    8)
syn.scalarization_3(processed_data.perception_training,
                    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_2.txt",
                    9)

syn.scalarization_3(processed_data.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max.txt",
                    6)
syn.scalarization_3(processed_data.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/min.txt",
                    7)
syn.scalarization_3(processed_data.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_1.txt",
                    8)
syn.scalarization_3(processed_data.perception_test,
                    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_2.txt",
                    9)
