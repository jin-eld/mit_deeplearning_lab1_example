use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::Tensor,
};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Read;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct AbcMusicItem {
    pub input: Vec<i32>,
    pub target: Vec<i32>,
    index2char: Vec<char>, // REMOVE
}

pub struct AbcMusicDataset {
    songdata: Vec<i32>,
    pub char2index: HashMap<char, i32>,
    pub index2char: Vec<char>,
    sequence_length: usize,
}

impl AbcMusicDataset {
    /// returns a tuple containing a (training dataset, validation dataset)
    pub fn new(
        dataset_file: &str,
        train_percentage: usize,
        sequence_length: usize,
    ) -> Result<(Self, Self), String> {
        let songs = Self::load_training_data(dataset_file)?;
        if songs.is_empty() {
            return Err("No data has been extracted".into());
        }

        println!("found {} songs", songs.len());

        /*
        for i in 0..20 {
            println!("\nSONG: {}", i);
            println!("{}", songs[i]);
            let song: Vec<char> = songs[i].chars().collect();
            println!(
                "{}",
                format!(
                    "LAST TWO: {} {}",
                    song[song.len() - 2] as u32,
                    song[song.len() - 1] as u32
                )
            );
        }
        */

        let songs_joined = songs.join("\n\n");
        let vocab: HashSet<char> = songs_joined.chars().collect();
        let mut sorted_vocab: Vec<char> = vocab.clone().into_iter().collect();
        sorted_vocab.sort();

        let char2index: HashMap<char, i32> = sorted_vocab
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as i32))
            .collect();
        let index2char: Vec<char> = sorted_vocab.clone();

        let songdata: Vec<i32> = songs_joined
            .chars()
            .map(|c| *char2index.get(&c).unwrap())
            .collect();

        let split_index = (songdata.len() * train_percentage) / 100;
        let (training_data, validation_data) = songdata.split_at(split_index);

        let training_data = training_data.to_vec();
        let validation_data = validation_data.to_vec();

        /*
        if sequence_length > (training_data.len() / 5)
            || (sequence_length > validation_data.len())
        {
            return Err(
                "Reduce sequence length or increase the data in the dataset"
                    .into(),
            );
        }
        */
        return Ok((
            Self {
                songdata: training_data,
                char2index: char2index.clone(),
                index2char: index2char.clone(),
                sequence_length,
            },
            Self {
                songdata: validation_data,
                char2index,
                index2char,
                sequence_length,
            },
        ));
    }

    pub fn get_vocab_size(&self) -> usize {
        return self.index2char.len();
    }

    pub fn get_char2index(&self) -> &HashMap<char, i32> {
        return &self.char2index;
    }

    pub fn get_index2char(&self) -> &Vec<char> {
        return &self.index2char;
    }

    fn load_training_data(dataset_file: &str) -> Result<Vec<String>, String> {
        let path = Path::new(dataset_file);
        let mut file = File::open(&path)
            .map_err(|e| format!("Unable to open file: {}", e))?;
        let mut text = String::new();
        file.read_to_string(&mut text)
            .map_err(|e| format!("Unable to read file: {}", e))?;
        return Ok(Self::extract_songs(text));
    }

    fn trim_excess_newlines(s: String) -> String {
        let re = Regex::new(r"\n\n+$").unwrap();
        re.replace_all(&s, "").into_owned()
    }

    pub fn extract_songs(text: String) -> Vec<String> {
        let song_parts: Vec<&str> = text.split("X:").collect();
        let mut songs = Vec::new();
        for part in song_parts.into_iter().skip(1) {
            let song = format!("X:{}", part);
            songs.push(Self::trim_excess_newlines(song));
        }
        println!("Found {} songs in text", songs.len());
        return songs;
    }
}

impl Dataset<AbcMusicItem> for AbcMusicDataset {
    fn get(&self, index: usize) -> Option<AbcMusicItem> {
        if index + self.sequence_length + 1 > self.len() {
            return None;
        }

        let input: Vec<i32> =
            self.songdata[index..index + self.sequence_length].to_vec();
        let target: Vec<i32> =
            self.songdata[index + 1..index + 1 + self.sequence_length].to_vec();

        return Some(AbcMusicItem {
            input: input,
            target: target,
            index2char: self.get_index2char().to_owned(),
        });
    }

    fn len(&self) -> usize {
        if self.songdata.is_empty() {
            return 0;
        }

        return self.songdata.len() - self.sequence_length - 1;
    }
}

#[derive(Clone)]
pub struct SongBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct SongBatch<B: Backend> {
    // the LSTM expect an input of [batch_size, sequence_length, input_size]
    pub x: Tensor<B, 3, Int>,
    pub y: Tensor<B, 3, Int>,
}

impl<B: Backend> SongBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<AbcMusicItem, SongBatch<B>> for SongBatcher<B> {
    fn batch(&self, items: Vec<AbcMusicItem>) -> SongBatch<B> {
        let batch_size = items.len();
        let sequence_length = items[0].input.len();
        let input_size = 1; // number of features in the context: 1 single note

        let mut flat_inputs =
            Vec::with_capacity(batch_size * sequence_length * input_size);
        let mut flat_targets =
            Vec::with_capacity(batch_size * sequence_length * input_size);

        let path = "my-batch.txt";
        let mut options = OpenOptions::new();
        options.create(true); // Enable writing
        options.append(true); // Enable appending

        let mut file = match options.open(path) {
            Ok(f) => f,
            Err(e) => {
                log::info!("FOOOOOOOOOOOORK! {}", e);
                panic!();
            }
        };

        let _ = writeln!(file, "items: {}", items.len());
        for item in items.iter() {
            let x_str: String = item
                .input
                .iter()
                .map(|&index| item.index2char[index as usize])
                .collect();

            let y_str: String = item
                .target
                .iter()
                .map(|&index| item.index2char[index as usize])
                .collect();

            let _ = writeln!(file, "x ({}):\\[{}\\]", item.input.len(), x_str);
            let _ = writeln!(file, "y ({}):\\[{}\\]", item.target.len(), y_str);

            flat_inputs.extend_from_slice(&item.input);
            flat_targets.extend_from_slice(&item.target);
        }

        let flat_x_tensor =
            Tensor::<B, 1, Int>::from_ints(&flat_inputs[..], &self.device);
        let flat_y_tensor =
            Tensor::<B, 1, Int>::from_ints(&flat_targets[..], &self.device);

        // reshape to [batch_size, sequence_length, input_size], which is what
        // the LSTM expects
        let x_tensor =
            flat_x_tensor.reshape([batch_size, sequence_length, input_size]);
        let y_tensor =
            flat_y_tensor.reshape([batch_size, sequence_length, input_size]);

        return SongBatch {
            x: x_tensor,
            y: y_tensor,
        };
    }
}
