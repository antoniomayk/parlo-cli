use clap::Parser;

/// Command-line interface (CLI) application that can transcribe audio files into text.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct ParloArgs {
    /// File system path for the audio
    #[arg(short, long)]
    audio: String,

    /// File system path for the GMML model
    #[arg(short, long)]
    model: String,
}

fn main() {
    let parlo_args = ParloArgs::parse();

    println!("audio: {}", parlo_args.audio);
    println!("model: {}", parlo_args.model);
}
