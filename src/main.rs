use std::{fs::File, io::ErrorKind, result};

use clap::Parser;
use indoc::formatdoc;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::{
    audio::{AudioBufferRef, SampleBuffer},
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    errors::Error,
    formats::FormatOptions,
    io::*,
    meta::MetadataOptions,
    probe::Hint,
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

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

    let audio = read_audio_file(&parlo_args.audio);
    let model = parlo_args.model;

    transcribe_audio_buffer(&audio, &model);
}

fn read_audio_file(path: &str) -> Vec<f32> {
    let media_src = match File::open(path) {
        Ok(it) => it,
        Err(err) => panic!(
            "{}",
            formatdoc! {
                "an unexpected error happened opening the audio file '{path}'. internal error: {err}",
                err = err.to_string().to_lowercase(),
                path = path.to_string().to_lowercase()
            }
        ),
    };

    let media_src_stream =
        MediaSourceStream::new(Box::new(media_src), MediaSourceStreamOptions::default());

    let mut format = match symphonia::default::get_probe().format(
        &Hint::new(),
        media_src_stream,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) {
        Ok(it) => it.format,
        Err(err) => panic!(
            "{}",
            formatdoc! {
                "audio format not supported. internal error: {err}",
                err = err.to_string().to_lowercase(),
            }
        ),
    };

    let (track, track_id) = match format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
    {
        Some(t) => (t, t.id),
        None => panic!(
            "{}",
            formatdoc! {
                "track not found inside the audio file '{path}'.",
                path = path.to_string().to_lowercase()
            }
        ),
    };

    let mut decoder = match symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
    {
        Ok(it) => it,
        Err(err) => panic!(
            "{}",
            formatdoc! {
                "audio decoder not supported. internal error: {err}",
                err = err.to_string().to_lowercase(),
            }
        ),
    };

    let mut audio_buffer = Vec::<f32>::new();

    let result: Result<(), symphonia::core::errors::Error> = loop {
        let packet = match format.next_packet() {
            Ok(it) => it,
            Err(err) => break Err(err),
        };

        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(it) => resample_audio_buffer(it)
                .iter()
                .for_each(|channel| channel.iter().for_each(|data| audio_buffer.push(*data))),
            Err(err) => break Err(err),
        }
    };

    match handle_eof_err(result) {
        Ok(_) => (),
        Err(err) => panic!(
            "{}",
            formatdoc! {
                "internal error: {err}",
                err = err.to_string().to_lowercase(),
            }
        ),
    }

    audio_buffer
}

fn handle_eof_err(result: result::Result<(), Error>) -> result::Result<(), Error> {
    match result {
        Err(Error::IoError(err))
            if err.kind() == ErrorKind::UnexpectedEof && err.to_string() == "end of stream" =>
        {
            Ok(())
        }
        _ => result,
    }
}

fn stereo_to_mono(samples_left: &[f32], samples_right: &[f32]) -> Vec<f32> {
    samples_left
        .iter()
        .zip(samples_right.iter())
        .map(|(x, y)| (x + y) / 2.0)
        .collect::<Vec<f32>>()
}

fn resample_audio_buffer<'a>(audio_buffer: AudioBufferRef<'_>) -> Vec<Vec<f32>> {
    let resample_ratio = 16_000 as f64 / audio_buffer.spec().rate as f64;
    let audio_buffer_capacity = audio_buffer.capacity();
    let audio_buffer_frames = audio_buffer.frames();
    let audio_buffer_channels_count = audio_buffer.spec().channels.count();

    let mut sample_buffer =
        SampleBuffer::<f32>::new(audio_buffer_capacity as u64, *audio_buffer.spec());

    sample_buffer.copy_planar_ref(audio_buffer);

    let samples = sample_buffer.samples().to_vec();
    let samples_mono = if audio_buffer_channels_count == 2 {
        stereo_to_mono(
            &samples[..audio_buffer_frames], // LEFT
            &samples[audio_buffer_frames..], // RIGHT
        )
    } else {
        samples
    };

    let interpolator = SincInterpolationParameters {
        sinc_len: 8,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler =
        match SincFixedIn::<f32>::new(resample_ratio, 2., interpolator, audio_buffer_frames, 1) {
            Ok(it) => it,
            Err(err) => panic!(
                "{}",
                formatdoc! {
                    "internal error: {err}",
                    err = err.to_string().to_lowercase(),
                }
            ),
        };

    match resampler.process(&[samples_mono.as_slice()], None) {
        Ok(it) => it,
        Err(err) => panic!(
            "{}",
            formatdoc! {
                "an error happened resampling the audio buffer. internal error: {err}",
                err = err.to_string().to_lowercase(),
            }
        ),
    }
}

fn transcribe_audio_buffer(audio: &Vec<f32>, model: &str) {
    let whisper_context = WhisperContext::new(model).unwrap();
    whisper_context.print_timings();

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 2 });
    params.set_n_threads((std::thread::available_parallelism().unwrap().get() / 2) as i32);
    params.set_language(Some("auto"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let mut state = whisper_context.create_state().unwrap();
    state.full(params, audio).unwrap();

    for i in 0..state.full_n_segments().unwrap() {
        let start_time_millis = segment_timestamp_to_millis(state.full_get_segment_t0(i).unwrap());
        let end_time_millis = segment_timestamp_to_millis(state.full_get_segment_t1(i).unwrap());
        let segment = state.full_get_segment_text(i).unwrap();

        println!(
            "{i}\n{} --> {}\n{}\n",
            millis_to_time(start_time_millis),
            millis_to_time(end_time_millis),
            segment.trim()
        );
    }
}

fn segment_timestamp_to_millis(t: i64) -> usize {
    t as usize * 10 as usize
}

fn millis_to_time(milliseconds: usize) -> String {
    let millis = milliseconds % 1000;
    let seconds = (milliseconds / 1000) % 60;
    let minutes = ((milliseconds / 1000) / 60) % 60;
    let hours = (milliseconds / 1000) / 60 / 60;

    format!("{hours:0>2}:{minutes:0>2}:{seconds:0>2},{millis:0>3}")
}
