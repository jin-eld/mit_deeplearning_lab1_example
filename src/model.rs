use crate::data::SongBatch;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{
    lstm::{Lstm, LstmConfig},
    Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig,
};


use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::train::ClassificationOutput;
use burn::train::TrainOutput;
use burn::train::TrainStep;
use burn::train::ValidStep;

#[derive(Config)]
pub struct MusicGenerationModelConfig {
    vocab_size: usize,
    embedding_dim: usize,
    rnn_units: usize,
    batch_size: usize,
}

#[derive(Module, Debug)]
pub struct MusicGenerationModel<B: Backend> {
    input: Embedding<B>,
    lstm_layer: Lstm<B>,
    output: Linear<B>,
    output_size: usize,
}

impl<B: Backend> MusicGenerationModel<B> {
    // Core forward pass
    pub fn forward(&self, input: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let input_shape = input.dims();
        let batch_size = input_shape[0];
        let seq_length = input_shape[1];

        // Remove the third dimension to match the expected input of the embedding layer
        let input_reshaped = input.reshape([batch_size, seq_length]);

        // Embed the input
        let embedded = self.input.forward(input_reshaped);

        // Pass through the LSTM layer
        let (hidden_state, _) = self.lstm_layer.forward(embedded, None);

        let hidden_state = sigmoid(hidden_state);

        // Reshape hidden_state to [batch_size * seq_length, rnn_units] for the Linear layer
        let hidden_shape = hidden_state.dims();
        let hidden_state_reshaped =
            hidden_state.reshape([batch_size * seq_length, hidden_shape[2]]);

        // Pass through the Linear layer
        let logits = self.output.forward(hidden_state_reshaped);

        // Reshape logits back to [batch_size, seq_length, vocab_size]
        return logits.reshape([batch_size, seq_length, self.output_size]);
    }

    // Task-specific forward pass
    pub fn forward_classification(
        &self,
        item: SongBatch<B>,
    ) -> ClassificationOutput<B> {
        let device = &self.input.devices()[0];

        let input = item.x.to_device(device);
        let targets = item.y.to_device(device);

        let logits = self.forward(input);

        // Flatten logits and targets for ClassificationOutput
        let input_shape = logits.dims();
        let batch_size = input_shape[0];
        let seq_length = input_shape[1];

        let flat_logits = logits
            .reshape([batch_size * seq_length, self.output_size])
            .to_device(device);
        let flat_targets =
            targets.reshape([batch_size * seq_length]).to_device(device);

        // Compute loss using CrossEntropyLoss
        let loss = CrossEntropyLoss::new(None, device)
            .forward(flat_logits.clone(), flat_targets.clone());

        ClassificationOutput::new(loss, flat_logits, flat_targets)
    }
}

impl<B: AutodiffBackend> TrainStep<SongBatch<B>, ClassificationOutput<B>>
    for MusicGenerationModel<B>
{
    fn step(&self, item: SongBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Perform forward pass
        let item = self.forward_classification(item);
        return TrainOutput::new(self, item.loss.backward(), item);
    }
}

impl<B: Backend> ValidStep<SongBatch<B>, ClassificationOutput<B>>
    for MusicGenerationModel<B>
{
    fn step(&self, item: SongBatch<B>) -> ClassificationOutput<B> {
        return self.forward_classification(item);
    }
}

impl MusicGenerationModelConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> MusicGenerationModel<B> {
        let input = EmbeddingConfig::new(self.vocab_size, self.embedding_dim)
            .init(device);

        let lstm_layer =
            LstmConfig::new(self.embedding_dim, self.rnn_units, true)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device);

        let output =
            LinearConfig::new(self.rnn_units, self.vocab_size).init(device);

        MusicGenerationModel {
            input,
            lstm_layer,
            output,
            output_size: self.vocab_size,
        }
    }
}
