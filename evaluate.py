import os
import glob
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationCompleteness, DiarizationCoverage, DiarizationPurity, DiarizationHomogeneity
from pyannote.metrics.detection import DetectionErrorRate, DetectionAccuracy, DetectionPrecision, DetectionRecall
from pyannote.metrics.segmentation import SegmentationPurityCoverageFMeasure, SegmentationCoverage, SegmentationPrecision, SegmentationPurity, SegmentationRecall
from pyannote.metrics.identification import IdentificationErrorRate, IdentificationPrecision, IdentificationRecall


def get_detection_metrics( reference, hypothesis, uem=None ):
    metric_dict = {}
    metric = DetectionErrorRate()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = DetectionAccuracy()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = DetectionPrecision()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = DetectionRecall()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met

    return metric_dict

def get_segmentation_metrics( reference, hypothesis, uem=None ):
    metric_dict = {}
    metric = SegmentationCoverage()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = SegmentationRecall()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = SegmentationPrecision()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = SegmentationPurity()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = SegmentationPurityCoverageFMeasure()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met

    return metric_dict


def get_diarization_metrics( reference, hypothesis, uem=None ):
    metric_dict = {}
    metric = DiarizationErrorRate()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = DiarizationCompleteness()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = DiarizationCoverage()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = DiarizationPurity()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = DiarizationHomogeneity()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met

    return metric_dict

def get_identification_metrics( reference, hypothesis, uem=None ):
    metric_dict = {}
    metric = IdentificationErrorRate()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = IdentificationPrecision()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met
    metric = IdentificationRecall()
    met = metric( reference, hypothesis, uem=uem )
    metric_dict[ metric.metric_name() ] = met

    return metric_dict

def evaluate_file( ref_csv_file, hyp_csv_file, uem_csv_file ):
    bname=os.path.basename( hyp_csv_file )
    df = pd.read_csv(ref_csv_file)
    startTimes = df["start_time_ms"]
    endTimes = df["end_time_ms"]
    speakerLabels = df["speaker_label"]
    reference = Annotation( uri=bname )
    for st, en, sp in zip(startTimes, endTimes, speakerLabels):
        reference[Segment(st/1000.0, en/1000.0)] = sp

    df = pd.read_csv(hyp_csv_file)
    startTimes = df["start_time_ms"]
    endTimes = df["end_time_ms"]
    speakerLabels = df["speaker_label"]
    hypothesis = Annotation( uri=bname )
    for st, en, sp in zip(startTimes, endTimes, speakerLabels):
        hypothesis[Segment(st/1000.0, en/1000.0)] = sp

    df = pd.read_csv(uem_csv_file)
    startTimes = df["start_time_ms"]
    endTimes = df["end_time_ms"]
    segmentList = []
    for st, en in zip(startTimes, endTimes):
        segmentList.append(Segment(st/1000.0, en/1000.0))
    uem = Timeline(segmentList)
    print( uem )

    result_dict = {}
    metrics = get_detection_metrics( reference, hypothesis, uem )
    result_dict.update( metrics )
    metrics = get_segmentation_metrics( reference, hypothesis, uem )
    result_dict.update( metrics )
    metrics = get_diarization_metrics( reference, hypothesis, uem )
    result_dict.update( metrics )
    metrics = get_identification_metrics( reference, hypothesis, uem )
    result_dict.update( metrics )

    return result_dict


ap = ArgumentParser()
ap.add_argument("--test_path", type=str, default=None)
ap.add_argument("--output_csv", type=str, default=None)

args = ap.parse_args()

ref_path = f"{args.test_path}/input/"
out_path = f"{args.test_path}/output/"

results = pd.DataFrame()

model_dirs = [ name for name in os.listdir(out_path) if os.path.isdir(os.path.join(out_path, name)) ]
for model in model_dirs:
    print( f"Evaluating Model {model}" )
    hyp_csv_files = glob.glob(f"{out_path}/{model}/*_speaker_turns.csv")
    for hyp_csv_file in hyp_csv_files:
        bname = os.path.basename( hyp_csv_file )
        bname = os.path.splitext( bname )[0]
        print( f"--> {bname}" )
        ref_csv_file=f"{ref_path}/{bname}.csv"
        uem_csv_file=f"{ref_path}/{bname}.map"
        result = evaluate_file( ref_csv_file, hyp_csv_file, uem_csv_file  )
        result.update( { 'Model' : model, 'File' : bname } )
        results = results.append(result, ignore_index=True)

results.to_csv( args.output_csv )

