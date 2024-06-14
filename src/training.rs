use crate::args::Training;
use crate::data::{AbcMusicDataset, SongBatcher};
use crate::model::MusicGenerationModelConfig;
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::DefaultRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::store::{Aggregate, Direction, Split},
        metric::{AccuracyMetric, CudaMetric, LossMetric},
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};
use std::collections::HashMap;

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 5)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize, //  Experiment between 1 and 64

    #[config(default = 100)]
    pub sequence_length: usize, // Experiment between 50 and 500

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 1e-3)]
    pub learning_rate: f64, // # Experiment between 1e-5 and 1e-1

    pub optimizer: AdamConfig,

    pub vocab_size: Option<usize>,
    pub char2index: Option<HashMap<char, i32>>,
    pub index2char: Option<Vec<char>>,
}

fn create_artifact_dir(artifact_dir: &str) {
    let path = std::path::Path::new(artifact_dir);
    if path.exists() {
        let mut dir_entries = std::fs::read_dir(artifact_dir)
            .unwrap_or_else(|_| panic!("Error reading directory"));
        if !dir_entries.next().is_none() {
            panic!("Artifact directory not empty!");
        }
    }

    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(devices: Vec<B::Device>, getopt: Training) {
    create_artifact_dir(&getopt.artifact_dir);

    let config_optimizer = AdamConfig::new();
    let mut config = TrainingConfig::new(config_optimizer);

    if let Some(val) = getopt.num_epochs {
        config.num_epochs = val;
    }

    if let Some(val) = getopt.batch_size {
        config.batch_size = val;
    }

    if let Some(val) = getopt.sequence_length {
        config.sequence_length = val;
    }

    if let Some(val) = getopt.learning_rate {
        config.learning_rate = val;
    }

    let (training_dataset, validation_dataset) = match AbcMusicDataset::new(
        &getopt.dataset_file,
        85, // 85% percent training data
        config.sequence_length,
    ) {
        Ok(songs) => songs,
        Err(e) => {
            println!("{}", e);
            return;
        }
    };

    config.char2index = Some(training_dataset.get_char2index().to_owned());
    config.index2char = Some(training_dataset.get_index2char().to_owned());
    config.vocab_size = Some(training_dataset.get_vocab_size());

    // num iterations in an epoch = ceil(dataset size / batch size)
    let model = MusicGenerationModelConfig::new(
        training_dataset.get_vocab_size(),
        256,  // embedding dim
        1024, // rnn units, Experiment between 1 and 2048
        config.batch_size,
    );

    // Data
    let batcher_train = SongBatcher::<B>::new(devices[0].clone());
    let batcher_valid = SongBatcher::<B::InnerBackend>::new(devices[0].clone());

    let seed = 469;

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(seed)
        .build(training_dataset);
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(seed)
        .build(validation_dataset);

    // Model
    let learner = LearnerBuilder::new(&getopt.artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(DefaultRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(devices.clone())
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model.init(&devices[0]),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{}/config.json", getopt.artifact_dir).as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{}/model", getopt.artifact_dir),
            &DefaultRecorder::new(),
        )
        .expect("Failed to save trained model");
}
