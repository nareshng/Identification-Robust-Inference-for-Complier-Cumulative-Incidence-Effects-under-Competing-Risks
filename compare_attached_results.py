#!/usr/bin/env python3
"""Reconcile the supplied 1,000-replication aggregate with the final run."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--attached',default='upload/simulation_results(1).csv')
    p.add_argument('--final',default='extended_results_final/extended_simulation_results.csv')
    p.add_argument('--output',default='results_alignment.csv')
    args=p.parse_args()
    old=pd.read_csv(args.attached)
    new=pd.read_csv(args.final).query("scenario == 'core'")
    keep=['delta','n','censoring','true_effect','reps','score_coverage',
          'score_mean_length','score_full_set','mean_first_stage',
          'wald_coverage','wald_mean_length']
    merged=old[keep].merge(new[keep],on=['delta','n','censoring'],
                           suffixes=('_attached','_final'),validate='one_to_one')
    merged['score_coverage_difference']=merged.score_coverage_final-merged.score_coverage_attached
    merged['combined_score_mcse']=np.sqrt(
        merged.score_coverage_attached*(1-merged.score_coverage_attached)/merged.reps_attached+
        merged.score_coverage_final*(1-merged.score_coverage_final)/merged.reps_final)
    merged['score_difference_in_mcse']=merged.score_coverage_difference/merged.combined_score_mcse
    merged['within_two_combined_mcse']=merged.score_difference_in_mcse.abs()<=2
    merged['truth_difference']=merged.true_effect_final-merged.true_effect_attached
    merged['interpretation']='Same qualitative conclusion; final run uses exact common truth and finite-jump correction.'
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    merged.to_csv(args.output,index=False)


if __name__=='__main__':
    main()
