# MOT20 Policy and Local-Development Use

Date reviewed: 2026-08-31

## Project decision

This repository uses MOT20 as an internal research milestone for improving a tracking and ReID system intended for deployment. The project is **not** preparing these test-adapted results for submission to the official MOTChallenge leaderboard.

Accordingly, local development may include analysis of MOT20 test images and predicted tracks, manual review of proposed identities, creation of pseudo-labels, ReID fine-tuning, and repeated iteration on the same test sequences. These activities are permitted for this project's internal development objective, but they change how any resulting MOT20 metrics must be described.

Results produced after learning from, manually reviewing, or adapting to MOT20 test data are not clean held-out benchmark results. They must be labeled as local, transductive, test-adapted, or deployment-development experiments and must not be presented as directly comparable official MOT20 leaderboard results.

## Official MOTChallenge policy

The official rules remain important context even though this project is not submitting its adapted results.

### Training and parameter selection

The current [MOTChallenge submission instructions](https://motchallenge.net/instructions/) strongly encourage participants to use only training sequences to select parameters. The benchmark paper similarly states that the best setting should be found using training data.

This requirement protects the meaning of the official test set as held-out evaluation data. Our internal test-adaptation work intentionally follows a different objective and must therefore be reported separately from standard benchmark evaluation.

### Test ground truth

The [MOTChallenge benchmark paper, Appendix A.1](https://arxiv.org/pdf/2010.07548#page=24), states that use of ground-truth labels on test data is strictly forbidden for benchmark participation.

MOT20 test ground truth is not available in this repository. Manually created identity decisions or cleaned pseudo-identities are project-generated supervision, not official MOT20 ground truth. Nevertheless, using them to fit a model means the test set is no longer held out.

### Evaluation-server use

The [submission instructions](https://motchallenge.net/instructions/) state that the evaluation server must not be used for training or parameter tuning. They impose a 72-hour waiting period and limit repeated submissions to discourage test-score optimization.

This project will not use the MOTChallenge evaluation server to guide the local adaptation process and will not upload the resulting test-adapted tracks as standard benchmark submissions.

### External training data and detections

The official policy permits additional training data and private detections when they are declared during submission. The [MOT20 results page](https://motchallenge.net/results/MOT20/) distinguishes public-detection and private-detection methods.

The supplied YOLOX detections and BoostTrack++ results should retain their detector, checkpoint, ReID, tracker, and post-processing provenance in every local experiment, regardless of whether a submission is planned.

### Multiple method variants

The [MOTChallenge FAQ](https://motchallenge.net/faq/) directs participants to compare method variants on training data and submit only one test result for a method. That restriction applies to leaderboard participation. Local deployment development may compare multiple test-adapted variants, provided none are misrepresented as held-out benchmark comparisons.

## Allowed local workflow

For this repository's deployment-development objective, the intended workflow may include:

- running frozen models on MOT20 test images
- analyzing BoostTrack++ test predictions
- extracting person crops and ReID embeddings
- detecting embedding outliers within tracks
- comparing identities across tracks
- manually reviewing same-person, different-person, and uncertain candidates
- splitting, merging, filtering, or relabeling local pseudo-identities
- fine-tuning ReID models using cleaned test-derived pseudo-labels
- repeating the process to improve deployment behavior

These actions are local research and adaptation, not official benchmark evaluation.

## Reporting requirements

Every experiment that uses MOT20 test-derived information must record:

- test sequences used
- source detection and track variant
- whether manual review affected labels or decisions
- ReID model and checkpoint before and after adaptation
- pseudo-label generation and cleaning procedure
- thresholds and reviewer decisions
- number of adaptation iterations
- exact output artifact paths
- whether reported metrics are diagnostic, test-adapted, or held out

Reports must not describe a model as evaluated on an unseen MOT20 test set if the model, its thresholds, its labels, or its selection procedure used those test sequences.

## Separation from any future benchmark submission

If an official MOT20 submission is considered later, it must use a separate protocol:

1. Develop and tune only on authorized training or validation data.
2. Freeze model weights, thresholds, and processing rules before test inference.
3. Do not use manual test identity corrections or test-derived fine-tuning.
4. Keep benchmark-valid artifacts separate from local test-adapted artifacts.
5. Declare external training data and private detections accurately.
6. Obtain organizer clarification for any test-time adaptation whose eligibility is uncertain.

The current project decision is that no such submission is planned. MOT20 is being used as a local development milestone rather than as a leaderboard target.

## Authoritative references

- [MOTChallenge submission instructions](https://motchallenge.net/instructions/)
- [MOTChallenge FAQ](https://motchallenge.net/faq/)
- [MOT20 dataset page](https://motchallenge.net/data/MOT20/)
- [MOT20 results and public/private detection protocols](https://motchallenge.net/results/MOT20/)
- [MOTChallenge benchmark paper and submission policy](https://arxiv.org/pdf/2010.07548#page=24)

