use argh::FromArgs;

#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand)]
pub(crate) enum Command {
    Training(Training),
}

#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "training")]
/// Training options.
pub(crate) struct Training {
    #[argh(option)]
    /// number of epochs
    pub(crate) num_epochs: Option<usize>,
    #[argh(option)]
    /// batch size                        
    pub(crate) batch_size: Option<usize>,
    #[argh(option)]
    /// length of the sequence of one training dataset item in characters
    pub(crate) sequence_length: Option<usize>,
    #[argh(option)]
    /// learning rate
    pub(crate) learning_rate: Option<f64>,
    #[argh(option, default = "String::from(\"data/irish.abc\")")]
    /// training dataset in ABC format notation
    pub(crate) dataset_file: String,
    #[argh(option, default = "String::from(\"/tmp/abcmusic\")")]
    /// where to save the training artifacts
    pub(crate) artifact_dir: String,
}

#[derive(FromArgs, PartialEq, Debug)]
/// Irish song music generator options
pub(crate) struct Opts {
    #[argh(subcommand)]
    pub(crate) command: Option<Command>,
    #[argh(option, default = "String::from(\"X\")")]
    /// start string for generation, first character must be X (ABC notation)
    pub(crate) start_string: String,
    #[argh(option, default = "default_generation_length()")]
    /// how much characters (ABC notation) to generate
    pub(crate) generation_length: usize,
    #[argh(option, default = "String::from(\"abcmusic\")")]
    /// directory where to find the trained model and its configuration
    pub(crate) model_dir: String,
}

fn default_generation_length() -> usize {
    return 1000;
}
