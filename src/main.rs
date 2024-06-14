use burn::backend::{
    libtorch::{LibTorch, LibTorchDevice},
    Autodiff,
};
use tch::Cuda;

mod args;
mod data;
mod inference;
mod model;
mod training;

fn main() {
    let getopt: args::Opts = argh::from_env();

    let mut devices = Vec::new();
    let device_count = Cuda::device_count() as usize;

    for i in 0..device_count {
        devices.push(LibTorchDevice::Cuda(i));
    }

    if device_count == 0 {
        devices.push(LibTorchDevice::Cpu);
    }

    match getopt.command {
        Some(args::Command::Training(cmd)) => {
            training::run::<Autodiff<LibTorch>>(devices, cmd);
        }
        _ => {
            inference::infer::<LibTorch>(
                &getopt.model_dir,
                devices[0],
                &getopt.start_string,
                getopt.generation_length,
            );
        }
    }
}
