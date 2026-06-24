TimeGAN: Anomaly Detection in IoT Temperature Series

TimeGAN · Autoencoder · Anomaly Detection · TensorFlow · Python

Implementation of TimeGAN for synthetic IoT temperature time series generation, followed by anomaly detection via autoencoder reconstruction error. Built on a simulated circadian temperature sensor (250 daily cycles, 24h resolution).


Pipeline

Synthetic IoT data → TimeGAN training (3 phases) → Synthetic generation
→ Real vs. synthetic comparison → Anomaly injection → Detection via MSE threshold


Dataset

Synthetic hourly temperature series simulating an IoT ambient sensor over 250 days:


Circadian curve: minimum ~12°C at 6h, maximum ~28°C at 15h
Light seasonal upward trend toward summer
Gaussian noise ±0.8°C


No external dataset required — data is generated in-notebook.


TimeGAN Architecture

Five GRU-based components following Yoon et al. (2019):

ComponentRoleEmbedderReal series → latent spaceRecoveryLatent space → reconstructed seriesGeneratorGaussian noise → synthetic latentSupervisorCaptures temporal dynamics in latent spaceDiscriminatorClassifies real vs. synthetic latent vectors

Training phases:


Phase 1 — Autoencoder (embedder + recovery): reconstruction loss 0.00080
Phase 2 — Supervisor: loss 0.00006
Phase 3 — Adversarial joint training (limited epochs → Gaussian fallback activated)



Anomaly Detection

Three anomalies injected on real trajectories:

AnomalyDescriptionMSEThresholdDetectedA1Point spike +15°C at 12h0.0055550.002099✅ (2.6×)A2Level shift +8°C hours 12–230.0028450.002099✅ (1.4×)A3Full circadian inversion0.0020260.002099❌ (marginally below)

Detection threshold: μ + 2σ of reconstruction MSE on real series.

A3 was not detected because the autoencoder's smoothed latent representation does not sufficiently distinguish a normal curve from its inverse — a known limitation of under-trained TimeGAN.


Key Results


Synthetic series preserve temperature scale (mean 22.7°C vs. real 26.2°C) and very short-term dependencies (lag-1 ACF)
Circadian seasonality was not learned due to computational constraints (limited adversarial epochs)
PCA 2D projection shows complete separation between real and synthetic trajectory clusters
Autoencoder is an effective detector for high-magnitude and sustained-level anomalies



Usage

bash# Install dependencies
pip install tensorflow numpy matplotlib scikit-learn statsmodels

# Run notebook
jupyter notebook timegan_anomaly_detection.ipynb

Output plots are saved to outputs_timegan/ automatically.


Repository Structure

TimeGAN-Anomaly-Detection-IoT-Temperature/
├── src/
│   └── timegan_anomaly_detection.ipynb
├── outputs/
│   └── .gitkeep     ← generated plots saved here at runtime
├── README.md
└── .gitignore


References

Yoon, J., Jarrett, D., & van der Schaar, M. (2019). Time-series Generative Adversarial Networks. NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html


Author

María Pais Fajín · Biomedical Data Scientist

Master's in Data Analysis & Interpretation · UNIR · 2026

GitHub: Finarosalina
