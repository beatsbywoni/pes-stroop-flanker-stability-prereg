# Preregistration: Stability across sessions of post-error slowing in Stroop and flanker, tested in Hedge et al. (2019, Experiment 1)

Version 1.0, written 2026-11-26 (UTC), before any response-time-based quantity of these data was computed.

## 1. Purpose

A reanalysis of Hedge et al. (2018), in which undergraduates completed Stroop and flanker tasks twice, three weeks apart, found that conventional post-error slowing (PES) ranked people consistently within a session (split-half coefficients of about .62 to .83) but correlated between sessions less than these coefficients allowed: the disattenuated correlation ρ = *r*~12~/√(*rel*~1~*rel*~2~) was .66 in Stroop and .50 in flanker (about .5 to .8 across estimators). That analysis was exploratory. This registration tests the result in an independent data set with the same tasks and two sessions about four weeks apart.

## 2. Data

- Hedge, C., Vivian-Griffiths, S., Powell, G., Bompas, A., & Sumner, P. (2019). Slow and steady? Strategic adjustments in response caution are moderately reliable and correlate across tasks. *Consciousness and Cognition, 75*, 102797. https://doi.org/10.1016/j.concog.2019.102797
- Public data: https://osf.io/zag7c/, folder `Data/Experiment1/`, files (SHA-256):
  - `Exp1S1Stroop.mat` 2bd7be8bb60eb80876b8383865a4b47fb3ac79246ad20e5296e3dbfbb2796616
  - `Exp1S2Stroop.mat` 53d8b65dd503a31634ea5cb21327cecfc39b764abe40655b5c22f5e35c12d66f
  - `Exp1S1Flanker.mat` 3e29e9033a9be437595f9659a3ab892e6ddbd165164f5c59288d123ca949dcea
  - `Exp1S2Flanker.mat` 32822e2d17e8208e174b005d954cac0c7a73c2bb782f7a0fa8d1e117ed91a13d
- Each file holds, per trial, participant, condition (speed, standard, or accuracy emphasis crossed with congruent, neutral, or incongruent), correct (1) or incorrect (0), and RT in seconds (README of the data).
- Design (Hedge et al., 2019): 57 students recruited, two did not return; participants with average accuracy below 60% were excluded (final *N* 47 for flanker and 43 for Stroop). Two sessions about four weeks apart. Per task and session, 12 blocks of 144 trials (four each with speed, standard, and accuracy instructions, block order randomized), each with 48 congruent, 48 neutral, and 48 incongruent trials in random order. Flanker stimuli stayed on screen until a response; Stroop stimuli for at most 1,950 ms. 750 ms between trials. Speed blocks gave "too slow" and "too fast" messages, accuracy blocks an "Incorrect" message after errors, and every block ended with feedback on mean RT and accuracy.

**What is known about these data before this registration.** Hedge et al. (2019) reported diffusion-model boundary separation and RT and accuracy costs and adjustments; to the author's knowledge, no analysis of these data reports a post-error measure. The author computed counts that use no RT (`code/hedge2019_counts.py`, `hedge2019_counts.json`; the RT column was discarded on loading): 55 participants in session 1 and 47 in both sessions of each task; 1,728 trials per participant, task, and session; errors per session median 311 and 346 (Stroop) and 211 and 206 (flanker), of which 79 and 84 (Stroop) and 62 and 63 (flanker) in standard-emphasis blocks; four Stroop participants with accuracy below 60% in a session and none in flanker. The same script checked that the rows of each participant form 12 consecutive runs of 144 trials with a single emphasis and 48 trials of each congruency, with about 96 changes of condition per run, consistent with rows in presentation order. The analysis code was tested on synthetic data in which every recorded RT was replaced by a random draw (`synthetic_test.json`); correctness codes were not replaced, so that file contains the real split-half coefficients and correlations between sessions of the error rate (disattenuated correlations .55 in Stroop and .64 in flanker) and the number of participants eligible when every RT is valid (43 and 46). No other quantity of these data has been computed or displayed. All results of the earlier studies of this project (the reanalysis of Hedge et al., 2018; the preregistered tests in the Self-Regulation Ontology data, https://doi.org/10.5281/zenodo.24534671 and https://doi.org/10.5281/zenodo.24545639) are known.

## 3. Coding of trials

- Rows in file order are trials in presentation order; trial *i* (0-based) of a participant and session is in block ⌊*i*/144⌋ + 1.
- Congruency from the condition code: 1, 4, 7 congruent; 2, 5, 8 neutral; 3, 6, 9 incongruent. Emphasis: 1 to 3 speed, 4 to 6 accuracy, 7 to 9 standard.
- A trial without a recorded RT is an omission. An error is an incorrect response with a recorded RT; a correct trial is a correct response with a recorded RT. Omissions are neither error nor control events and are not scored.
- Scored trial: correct, RT from 0.15 to 3 s, in the same block as the preceding trial. **Conventional PES**: mean RT of scored trials after an error minus after a correct trial (as in the earlier reanalysis, function `pairs` of `stability.py`).

## 4. Participants

Per task, participants with both sessions and accuracy (correct responses per trial) of at least .60 in both sessions, which reproduces the exclusion of Hedge et al. (2019). A participant enters with at least 10 scored post-error trials in each session and at least two post-error and two post-correct scored trials in each half of an odd/even split in each session (the rule of the earlier reanalysis without its block splits, which do not apply to this block design).

## 5. Scores and coefficients

- **Primary score**: conventional PES in the standard-emphasis blocks (four blocks of 144 trials per session), which had no feedback after single trials, as in the earlier reanalysis.
- **Split-half coefficient** of each session, *rel*: mean over 1,000 stratified random splits (post-error and post-correct trials assigned separately to halves of equal size within the congruency of the scored trial) of the Spearman–Brown-corrected correlation between half-scores, each half scored from its own trials. *r*~12~: correlation of the full scores between sessions. **ρ** = *r*~12~/√(*rel*~1~*rel*~2~).
- 95% percentile bootstrap intervals over participants (2,000 resamples, split means recomputed in each resample; seeds in the code, base 20261207). Resamples in which a split-half coefficient is not positive leave ρ undefined; if more than 2.5% of resamples (50 of 2,000) are undefined, the upper limit is not determined (n.d.).
- The mean ρ over the two tasks is bootstrapped jointly by resampling the union of participants (the same 47 participants took both tasks); a resample in which ρ is undefined for either task is undefined.

## 6. Hypotheses and decision rules

- **H11 (primary)**: the mean over Stroop and flanker of ρ of conventional PES is below 1. Supported if the upper limit of its 95% interval is below 1 (and determined); otherwise not supported.
- **H12 (Stroop)** and **H13 (flanker)**: ρ of conventional PES is below 1 in each task, same rule.

Interpretation fixed now: if H11 is supported, within-session consistency overstates the stability of PES over about four weeks in these tasks, as in the earlier reanalysis; if not, a shortfall is not established in this sample, and the result is reported as such.

## 7. Further analyses (reported, not tested)

*rel*, *r*~12~, ρ, means, and change between sessions for PES, mean correct RT, and error rate (all trials of the standard blocks; omissions count as errors in the error rate) per task; the mean ρ over tasks for mean correct RT and error rate, and the paired difference of the mean ρ between PES and mean correct RT; PES computed from all 12 blocks with splits stratified by emphasis and congruency, per task and as a mean over tasks.

## 8. Power

`code/power_sim.py` (`power_sim.json`) simulated about 43 participants per task with true ρ, *rel*~1~, and *rel*~2~ equal to the estimates of the earlier reanalysis (Stroop .66, .79, .76; flanker .50, .62, .83): H11 was supported in 80% of simulations (H12 71%, H13 69%); with ρ = .75 in both tasks, 77%; with ρ = .85, 40%. The standard blocks contain more errors per session than the earlier tasks (medians about 60 to 85 against about 45), so the split-half coefficients may be higher, which would increase power.

## 9. Reporting and code

All tests in Section 6 are reported, whatever their outcome, as Study 4 of the article that reports the earlier studies, together with any deviation from this registration. Code: `code/study4_hedge2019.py`, which calls the functions of `stability.py` (identical to the copy deposited with the earlier registrations) for scoring, splits, coefficients, and bootstrap.
