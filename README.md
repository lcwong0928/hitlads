# HITLADS

Human in the Loop Anomaly Detection System

Anomaly detection in time series data is increasingly common and useful in various domains that require monitoring
metrics, including fraud detection, fault detection, system health monitor-ing, etc. Due to a lack of labeled samples,
anomaly detection is often treated as an unsupervised machine learning problem. For example, unsupervised
reconstruction-based approaches like Generative Adversarial Networks (GAN) with LSTM-RNN encoder-decoder architectures
have seen success in tackling this task. However, this proposal offers alternative transformer-based architectures to
allow for parallel computation and capture long dependencies in anomaly detection models for multivariate time series
data. Then, this proposal explores incorporating a human-in-the-loop (HITL) framework to reformulate existing
unsupervised models into semi-supervised models and iteratively train the semi-supervised model starting with little to
no labeled data. This new HITL framework for anomaly detection in time-series data could also serve as a benchmark for
similar research in the future.
