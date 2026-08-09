from Environment import configure_tensorflow, load_cuda_libraries, test_gpu

# Must run before any TensorFlow import — this process's or any module's —
# so CUDA libs and env vars are in place before TensorFlow initializes.
load_cuda_libraries()
configure_tensorflow()

from Augmentation.PipelineBuilder import PipelineBuilder
from DataAcquisition.DataBuilder import DataBuilder
from DataPreparation.DataSplitter import DataSplitter
from DataPreprocessing.DataPreprocessor import DataPreprocessor
from Evaluation.ModelEvaluator import ModelEvaluator
from FeatureExtraction.FeatureExtractor import FeatureExtractor
from Globals import set_seeds
from Training.ModelTrainer import ModelTrainer


def main():
    test_gpu()
    set_seeds()

    print("\n=== Stage 1: Data Acquisition ===")
    data_builder = DataBuilder()
    data_builder.verify_layout()
    raw_df = data_builder.build_index()
    print(f"Files matching target composers: {len(raw_df)}")

    print("\n=== Stage 2: Data Preprocessing ===")
    preprocessor = DataPreprocessor()
    clean_df, dedup_df, stats = preprocessor.process(raw_df)
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Unreadable/invalid: {stats['unreadable']}")
    print(f"Usable files: {stats['usable']}")
    print(preprocessor.summarize(raw_df, dedup_df, clean_df))
    preprocessor.visualize(clean_df)

    print("\n=== Stage 3: Data Splitting ===")
    splitter = DataSplitter()
    splits = splitter.split(clean_df)
    print(splitter.describe(splits))

    print("\n=== Stage 4: Feature Extraction ===")
    extractor = FeatureExtractor()
    f_train, y_train = splits["train"]
    f_val, y_val = splits["val"]
    f_test, y_test = splits["test"]

    train_arr = extractor.cached_split_arrays("train", f_train, y_train)
    val_arr = extractor.cached_split_arrays("val", f_val, y_val)
    test_arr = extractor.cached_split_arrays("test", f_test, y_test)

    yseq_train = train_arr["seq"][1]
    ycnn_train = train_arr["cnn"][1]
    print("LSTM window class distribution:", extractor.class_distribution(yseq_train))
    print("CNN  window class distribution:", extractor.class_distribution(ycnn_train))

    cw_seq = extractor.class_weights(yseq_train)
    cw_cnn = extractor.class_weights(ycnn_train)
    extractor.visualize(yseq_train, ycnn_train)

    print("\n=== Stage 5: Data Augmentation (tf.data pipelines) ===")
    pipeline_builder = PipelineBuilder()
    pipelines = pipeline_builder.build_all(train_arr, val_arr, test_arr)

    print("\n=== Stage 6: Model Training ===")
    trainer = ModelTrainer()

    lstm_model, lstm_history, lstm_table = trainer.train_lstm(pipelines, cw_seq)
    print("\nLSTM grid-search results:\n", lstm_table)
    trainer.visualize(lstm_history, "LSTM")

    cnn_model, cnn_history, cnn_table = trainer.train_cnn(pipelines, cw_cnn)
    print("\nCNN grid-search results:\n", cnn_table)
    trainer.visualize(cnn_history, "CNN")

    print("\n=== Stage 7: Model Evaluation ===")
    evaluator = ModelEvaluator()

    _, yseq_test, seq_test_fid, _ = test_arr["seq"]
    _, ycnn_test, cnn_test_fid, _ = test_arr["cnn"]

    lstm_metrics = evaluator.evaluate(lstm_model, pipelines["seq_test"],
                                        yseq_test, seq_test_fid, "LSTM")
    cnn_metrics = evaluator.evaluate(cnn_model, pipelines["cnn_test"],
                                        ycnn_test, cnn_test_fid, "CNN")

    
    comparison = evaluator.compare(lstm_metrics, cnn_metrics)

    print("\nPipeline complete")
    print(comparison)
    return 0



if __name__ == "__main__":
    exit(main())
