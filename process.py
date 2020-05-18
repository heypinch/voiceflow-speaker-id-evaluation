from argparse import ArgumentParser
from diarization.trim_audio.audio_trim_vad import audio_trim_vad
from diarization.embed_audio.embed_wavfile import embed_wav
from diarization.cluster_embeddings.cluster_embeddings import cluster_embeddings
import librosa
import numpy as np
import pandas as pd
import json
import os
import glob

def process(wav_file, output_path, traced_model, n_clusters=None, ignore_spans=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fname = os.path.basename( wav_file )
    fname = os.path.splitext(fname)[0]

    vad_mappings = audio_trim_vad(wav_file)
    audio, sr = librosa.load(wav_file, sr=16000, res_type="kaiser_fast")
    audio_end_ms = 1000.0 * audio.shape[0] / sr
    vad_mappings.to_csv(f"{output_path}/{fname}_vad_mappings.csv")
    embeddings = embed_wav(audio, traced_file=traced_model, frame_ms=1500, hop_ms=100, model_file=None, use_trace=True, cuda=False)
    np.save(f"{output_path}/{fname}_embeddings.npy", embeddings)

    speaker_turns = None
    if n_clusters is None:
        speaker_turns, speaker_idxs = cluster_embeddings(embeddings, vad_mappings=vad_mappings, hop_ms=100, ignore_spans=ignore_spans)
    else:
        speaker_turns, speaker_idxs = cluster_embeddings(embeddings, vad_mappings=vad_mappings, hop_ms=100, force_clusters=n_clusters, ignore_spans=ignore_spans)

    if speaker_turns is not None:
        speaker_turns["end_time_ms"][-1] = audio_end_ms

    df = pd.DataFrame.from_dict(speaker_turns)
    df.index.name = "idx"
    df.to_csv(f"{output_path}/{fname}_speaker_turns.csv", columns=["start_time_ms", "end_time_ms", "speaker_label", "speaker_label_internal"])

    with open(f"{output_path}/{fname}_speaker_turns.json", "w") as f:
        json.dump(speaker_turns, f)

    with open(f"{output_path}/{fname}_speaker_turns.rttm", "w") as f:
        for st, en, cl in zip(speaker_turns["start_time_ms"], speaker_turns["end_time_ms"], speaker_turns["speaker_label"]):
            if cl > 0:
                print(
                        "SPEAKER meeting 1 ",
                        st / 1000.0,
                        " ",
                        (en - st) / 1000.0,
                        " <NA> <NA> ",
                        cl - 1,
                        " <NA> <NA>", file=f
                    )

    return


if __name__ == "__main__":
    # Parse command line arguments
    ap = ArgumentParser()
    ap.add_argument("--test_path", default=None, type=str)
    ap.add_argument("--wav_file", default=None, type=str)
    ap.add_argument("--traced_model", default=None, type=str)
    args = ap.parse_args()

    if ( args.test_path is None ) or ( args.traced_model is None ) :
        print( "Error: You must specify the test_path and traced_model to process" )
        exit(1)

    model_name = os.path.basename( args.traced_model )
    output_path = f"{args.test_path}/output/{model_name}"

    if args.wav_file is not None:
        process( args.wav_file, args.test_path, args.traced_model )
    else:
        wav_files = glob.glob(f"{args.test_path}/input/*.wav")
        for filename in wav_files:
            bname = os.path.basename(filename)
            print( f"Processing {bname}" )
            process( filename, output_path, args.traced_model )

