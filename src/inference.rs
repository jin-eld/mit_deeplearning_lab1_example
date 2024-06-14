use crate::model::MusicGenerationModelConfig;
use crate::training::TrainingConfig;

use burn::prelude::*;
use burn::record::DefaultRecorder;
use burn::record::Recorder;

use burn::tensor::activation::softmax;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    start_string: &str,
    generation_length: usize,
) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");

    let vocab_size = config.vocab_size.unwrap();
    let char2index = config.char2index.unwrap();
    let index2char = config.index2char.unwrap();

    let record = DefaultRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = MusicGenerationModelConfig::new(
        vocab_size,
        256,
        1024,
        config.batch_size,
    )
    .init::<B>(&device)
    .load_record(record);

    let input_eval: Vec<i32> = start_string
        .chars()
        .map(|c| *char2index.get(&c).unwrap() as i32)
        .collect();

    let mut input_eval =
        Tensor::<B, 1, Int>::from_ints(&input_eval[..], &device).reshape([
            1,
            input_eval.len(),
            1,
        ]); // Adjust the dimensions to match the model input

    let mut text_generated = start_string.to_string();

    // Model inference loop
    for _ in 0..generation_length {
        // Get model predictions
        let output = model.forward(input_eval.clone());
        let predictions = output.squeeze(0);

        // Sample from the predictions to get the next character
        let predicted_id = sample_from_logits(predictions) as i32;

        // Add the predicted character to the generated text
        let predicted_char = index2char[predicted_id as usize];
        text_generated.push(predicted_char);

        // Prepare the next input
        input_eval = Tensor::<B, 1, Int>::from_ints(*&[predicted_id], &device)
            .reshape([1, 1, 1]);
    }

    println!("Generated text: \n{}", text_generated);
}

// Helper function to sample from logits
fn sample_from_logits<B: Backend>(logits: Tensor<B, 2>) -> usize {
    let probabilities = softmax(logits, 1);

    // Extract the underlying data as a vector of floating-point numbers
    let probabilities_data = probabilities.into_data().convert::<f32>();
    let probabilities_vec: Vec<f32> = probabilities_data.value;

    // Sample from the probabilities
    let mut rng = rand::thread_rng();
    let distribution = WeightedIndex::new(&probabilities_vec).unwrap();
    distribution.sample(&mut rng)
}
