# Network Intrusion Detection with Concept Drift Handling (UNSW-NB15)

A network intrusion detection system (NIDS) that classifies network flows as normal or attack, demonstrates how its accuracy degrades when traffic changes over time (concept drift), and compares ways of detecting and handling that drift. A local Streamlit dashboard lets researchers and observers explore the data, replay the stream, watch drift alarms and retraining events, classify their own records and check how well the system scales.

The project runs entirely locally. Model training and the drift experiments were done in Google Colab notebooks, and the dashboard runs on a personal computer from the saved results.

---

## 1. What the project does

1. Trains a supervised classifier on the UNSW-NB15 dataset to separate normal traffic from attacks.
2. Builds a simulated data stream of 60 windows in which two kinds of drift are injected.
3. Shows that a model that is never updated (the static model) loses accuracy after the drift.
4. Detects the drift with two complementary detectors, one that needs labels (ADWIN) and one that does not (a per-feature Kolmogorov-Smirnov test).
5. Compares three ways of responding: no adaptation, drift-triggered retraining, and an online learner (Adaptive Random Forest).
6. Presents everything in a five-page dashboard, including a live prediction form and a performance and scalability page.

## 2. Key features

- Supervised binary classification of network flows (normal vs attack) with XGBoost and Random Forest baselines.
- Reusable preprocessing pipeline, fitted once and used identically in the notebooks and the dashboard.
- Controlled drift simulation: abrupt appearance of new attack families, and progressive growth in traffic volume.
- Two drift detectors: ADWIN on the model's error stream, and a Bonferroni-corrected KS test on feature distributions.
- Three adaptation strategies compared with prequential (predict first, learn afterwards) evaluation.
- Per-window, per-phase and per-attack-family evaluation.
- Interactive dashboard: data explorer, stream replay with drift alarms and retrain markers, manual classification with an adjustable decision threshold, and a capacity calculator.
- Measured throughput, latency and retraining cost, with a discussion of deployment.

## 3. Methodology

### 3.1 Data

The dataset is UNSW-NB15 (Moustafa and Slay, 2015), a labelled collection of network flows containing normal traffic and nine attack categories. The official training-set and testing-set files are used (about 257,000 rows combined). Each row has roughly 42 input features, a binary label (0 normal, 1 attack) and an attack category.

- Categorical features: protocol (proto), service, connection state (state).
- Heavily skewed numeric features: volume-type columns such as byte counts, packet counts, load and rate.
- Other numeric features: durations, TTL values, window sizes, jitter, inter-packet times, round-trip timings, connection-count features and flags.

The id column is dropped. The attack category is never predicted: it is used only to build the stream and to break results down by attack family. The dataset itself is not included in this repository and must be downloaded separately (it is available from the UNSW Canberra Cyber site and from Kaggle).

### 3.2 Exploration (notebook 1)

Class balance (the official training file is mostly attacks, while the test file is more balanced), the categorical columns and the attack rate for each value, the skew of the numeric columns, and the correlation between features were examined. These findings drove the preprocessing choices.

### 3.3 Preprocessing

A scikit-learn ColumnTransformer with three branches, fitted on the training file only so nothing from the test set leaks in:

- Skewed numeric columns (non-negative, skew above 5, more than 10 distinct values): log1p transform followed by standard scaling.
- Other numeric columns: standard scaling.
- Categorical columns: one-hot encoding, with unseen categories ignored and categories seen fewer than 50 times grouped together.

The fitted pipeline is saved once (preprocess.joblib) and reused for the warm-up data, the stream and the dashboard's manual input.

### 3.4 Baseline models (notebook 1)

- XGBoost: 300 trees, maximum depth 8, learning rate 0.1, 80% row and column sampling.
- Random Forest: 200 trees with balanced class weights.

Both are trained on the official training file and scored on the official test file with accuracy, precision, recall, F1 and ROC-AUC, plus confusion matrices and ROC curves. XGBoost feature importance was used to choose the editable fields on the prediction page. These baseline scores come from a separate experiment from the stream and should not be compared directly with the stream results.

### 3.5 Stream construction (notebook 2)

The training and testing files are merged into one pool, which is divided into three groups: normal traffic, attacks from four "old" families (Generic, Exploits, Fuzzers, DoS), and attacks from five "new" families (Reconnaissance, Analysis, Backdoor, Shellcode, Worms). Rows are handed out without repetition.

| Parameter | Value |
|---|---|
| Warm-up set | 30,000 rows: 60% normal, 40% attacks, attacks from old families only |
| Stream | 60 windows of 1,000 rows (600 normal and 400 attacks per window, shuffled within the window) |
| Abrupt drift | From window 20, 85% of attacks come from the new families |
| Volume drift | Windows 40 to 49 ramp the volume factor from about 1.4 to 5, then it stays at 5 |
| Volume columns | sbytes, dbytes, sload, dload, spkts, dpkts, rate |
| Random seed | 42 |

The order is simulated: windows are blocks of resampled rows, not real chronological traffic.

### 3.6 Concept drift simulated

- Abrupt (sudden) drift at window 20. The initial model has only seen four attack families. From window 20 most attacks come from five families it has never seen, so the attack class changes suddenly and the attack-family mix changes with it.
- Progressive volume drift from window 40 to 50. Seven traffic-volume features are multiplied by a factor that grows continuously, applied to the raw values before preprocessing. This is mainly covariate shift: the input distribution moves while labels stay attached to the original rows. It is called "gradual" in the dashboard.

Recurring drift and label-noise drift are not simulated.

### 3.7 Drift detection

- ADWIN (River, delta 0.002) receives a 0 or 1 for every row (was the static model wrong?) and raises an alarm when the error rate in the older and newer parts of its adaptive window differs significantly. It is performance-based and needs true labels, so it only notices drift that hurts accuracy.
- Per-feature two-sample Kolmogorov-Smirnov test compares every numeric feature of each window against a fixed 5,000-row warm-up reference. A feature counts as shifted when its p-value is below 0.001 divided by the number of features (Bonferroni correction). A window is flagged when at least 10% of numeric features shifted. It needs no labels and can notice drift before accuracy drops, but it can also flag harmless changes.

### 3.8 Adaptation strategies

| Strategy | How it works |
|---|---|
| Static (no adaptation) | The XGBoost model trained on the warm-up set (200 trees, depth 6) is never updated. It is the control. |
| Drift-triggered retraining | For each window the current model predicts first. ADWIN and the KS test are then checked. If either fires and more than two windows have passed since the last retrain (cooldown), a new XGBoost is trained from scratch on the last six windows of labelled data. The KS reference is replaced by the current window, ADWIN is reset, and a consolidation retrain is scheduled three windows later. Every retrain is logged. |
| Online ARF | River's Adaptive Random Forest (10 Hoeffding trees, seed 42) first learns from 15,000 warm-up rows, then predicts and learns one row at a time. Each tree has its own drift detector, and trees that degrade are replaced by trees trained on recent data. There is no external trigger. |

### 3.9 Evaluation

- Prequential evaluation: each window (or row) is predicted before it is used for learning.
- Metrics (attack is the positive class, threshold 0.5): accuracy, precision, recall and F1 per window, with ROC-AUC, confusion matrices and ROC curves for the Day 1 baselines.
- Phases: before drift (windows 0 to 19), new attack families (20 to 39), progressive volume drift (40 to 49) and after (50 to 59), with mean F1 per phase.
- Extra analyses: recall per attack family after window 20, F1 and recall over all post-drift rows, detector false alarms and detection delay, the share of features flagged by the KS test per window, and the number of retrains.

### 3.10 Performance and scalability measurement

A timing cell measures preprocessing throughput, XGBoost batch throughput and single-record latency, ARF single-row prediction time, the time for one retrain, KS test time per window, ADWIN time per row and model size. The best of three runs is kept. The results are saved as performance_report.csv and shown on the dashboard.

## 4. The dashboard

The dashboard replays the saved results and runs no detection live. Only the prediction page computes anything on the spot.

1. Overview: experiment setup, the strategy comparison table and chart, and the Day 1 baseline scores.
2. Data explorer: three tabs. Stream composition shows the stacked mix of normal, old-attack and new-attack rows per window. Attack families shows the count per family and the share each strategy detects after window 20. Feature distributions overlays a chosen feature across chosen phases, with an optional log scale.
3. Stream and drift monitor: a window slider and a Play button with adjustable speed. For the chosen window it shows the phase description, F1 per strategy, drift alarms and retrain events, the F1 chart with retrain and ADWIN markers, the KS shift chart, and the share classified correctly for each traffic type. Moving a control during Play interrupts the replay.
4. Try a prediction: choose a starting point (default values, a random normal record, a random known-family attack or a random new-family attack), edit eight key fields, adjust the decision threshold, and compare the attack probability from the initial model and the adapted model.
5. Performance and scalability: headline measurements, a table that explains each one, a throughput chart, a capacity calculator for a chosen flows-per-second rate, and expanders on the deployment design and on the limits of the measurements.

## 5. Industry relevance

Detection models decay silently as attackers and traffic patterns change, and in practice keeping a model healthy means knowing when to retrain and paying for labels. This project quantifies the decay, compares ways of reacting, and measures the cost.

- Security analysts and network administrators: the threshold slider makes the trade-off between false alarms and missed attacks visible, and the drift monitor shows when to retrain instead of retraining blindly.
- Researchers: the stream, the alarms and the strategy comparison can be explored interactively.
- Scalability and cost: the performance page reports measured throughput, latency and retraining time, and lets a user check a traffic rate against the measured capacity.
- Deployment (design only, not built): a prediction service behind a REST API, a streaming source that groups flows into windows for the detectors, exported monitoring metrics, and analyst-approved retraining in the background. Extracting flow features from raw packets, authentication and late or missing labels are not covered.

## 6. Limitations

- The drift is simulated by resampling and scaling UNSW-NB15, not observed in real chronological traffic.
- Labels are assumed to arrive immediately after each window, and retraining uses them. In practice labels are slow and expensive.
- Detector and retraining parameters (the 10% KS cutoff, six-window buffer, cooldown, ADWIN delta) were set by hand, not tuned.
- The comparison between retraining and ARF is not perfectly clean: the base learners differ (boosted trees vs Hoeffding trees), and ARF warms up on 15,000 rows against 30,000 for XGBoost.
- Every result comes from one seeded run, so small differences between strategies should not be over-interpreted.
- The task is supervised, so attack types unlike anything in training can be missed. There is no unsupervised anomaly detector.
- The drift lines on the dashboard are shown only because the drift is simulated. A real system would not know them.
- Performance figures depend on the machine they were measured on, and exclude feature extraction from raw packets and input/output costs.

## 7. Tools Used

Google Colab and Google Drive, pandas, NumPy, scikit-learn, XGBoost, River, SciPy, matplotlib, seaborn, joblib, Streamlit, Plotly, VS Code.
