# FortifAI – Federated Cybersecurity Agent Hub  
### Next-Generation Distributed Threat Detection Using Federated Learning & On-Device ML

---

## 🚀 Overview

FortifAI is an **innovative cybersecurity platform** that combines:

✅ Federated Learning  
✅ On–device machine learning anomaly detection  
✅ Endpoint behavioral monitoring  
✅ Privacy-preserving threat intelligence sharing  
✅ Adaptive model evolution  
✅ Real-time malicious activity visualization (client GUI)  

Unlike traditional security products that:

- centralize raw logs,
- require full data visibility,
- depend on signatures or cloud-only models,

FortifAI enables **each workstation (edge client)** to learn from its own behavior autonomously, detect threats locally, and collaboratively improve a **shared global model** *without exposing private data*.

This creates a **self-improving defense network** across an enterprise environment.

---

# 🧠 Core Innovation

FortifAI introduces three unique innovations:

---

### 🔹 1. Federated Learning for Cybersecurity

Instead of sending raw telemetry, clients send:

- model parameter updates
- statistical summaries
- anomaly patterns

The server aggregates these and distributes an improved global model.

This provides:

✅ privacy  
✅ scalability  
✅ rapid adaptation to emerging threats  
✅ heterogeneous environment support  

---

### 🔹 2. On-Device ML Ensemble

Each client runs a **multi-stage anomaly detection pipeline**:

1. Z-Score statistical deviation  
2. Isolation Forest (tree ensemble)  
3. Autoencoder neural network  
4. Optional One-Class SVM

This hybrid approach detects:

- unknown malware behaviors
- lateral movement
- abnormal network spikes
- rogue processes
- suspicious file activity

---

### 🔹 3. Intelligent Secure Aggregation

The server performs:

- L2 norm clipping
- Trimmed-mean aggregation
- Differential privacy noise addition
- Client reputation scoring
- Validation & rollback

This prevents:

✅ model poisoning attacks  
✅ outlier influence  
✅ corrupted updates  
✅ adversarial manipulation

---

# 🏗 Architecture

```

+--------------------+         +--------------------+
|   Client Device    |  FL     |     Admin Server   |
|                    |-------> |  Federated Trainer |
|  Local ML Models   | Update  |  Robust Aggregator |
|  Anomaly Detector  |         |  DP + Validation   |
+--------------------+         +--------------------+
^                                |
|          Global Model          |
+--------------------------------+

```

---

# 🖥 Client Components

Each workstation runs an **agent executable** containing:

### ✅ Data Collectors
- network activity
- process monitoring
- file system events

### ✅ Feature Extraction
Every 60 seconds, features are computed:

```

conn_count
unique_dst_count
bytes_sent
bytes_recv
port_entropy
proc_spawn_count
file_create_count
file_exec_count
hash_novelty
CPU usage
memory usage

```

These are normalized using running mean/std.

---

# 🤖 Local Anomaly Detection Logic (Edge ML)

## 1. Z-Score Detection

For feature value \( x \):

\[
z = \frac{x - \mu}{\sigma}
\]

If \(|z| > threshold\) → anomaly candidate

---

## 2. Isolation Forest

Algorithm intuition:

- randomly selects feature and split value
- isolates anomalies faster than normal points

Isolation path length \( h(x) \):

\[
Score(x) = 2^{- \frac{h(x)}{c(n)} }
\]

where \(c(n)\) is normalization term.

Lower score → more anomalous

---

## 3. Autoencoder Neural Network

Architecture:

```

Input -> Dense(64) -> Dense(32) -> Latent(8)
-> Dense(32) -> Dense(64) -> Output

```

Training goal:

\[
\hat{x} = AE(x)
\]
\[
Loss = || x - \hat{x} ||^2
\]

If reconstruction error > threshold → anomaly

---

## 4. One-Class SVM (Optional)

Learns a boundary enclosing normal data.

---

# 🧩 Final Decision Logic

```

if zscore_flag:
anomaly
elif iso_score < iso_threshold:
anomaly
elif ae_error > ae_threshold:
anomaly

```

If anomaly:

- logged locally
- shown in GUI
- influences FL update metadata

---

# 🌍 Federated Learning Logic

## 🔁 Client Side

Each training cycle:

1. Train autoencoder 1–3 epochs on local data
2. Refit Isolation Forest
3. Compute model delta:

\[
\Delta W = W_{local} - W_{global}
\]

4. Clip update:

\[
\Delta W = \Delta W \cdot \frac{clip}{||\Delta W||}
\]

5. Send metadata:

```

samples_used
local_loss
anomaly_rate
delta_norm

```

---

## 🏢 Server Side Aggregation

### Trimmed Mean

For each weight tensor:

- sort updates
- drop top 10% and bottom 10%
- compute mean of remaining

### Differential Privacy

Noise added:

\[
Noise \sim \mathcal{N}(0, \sigma^2)
\]

### Validation & Rollback

If global model performance decreases → revert

---

# 🛡 Security Advantages Over Existing Systems

| Feature | Traditional SIEM | Cloud AV | FortifAI |
|---------|------------------|----------|----------|
| Raw data uploaded | ✅ | ✅ | ❌ |
| Learns locally | ❌ | ❌ | ✅ |
| Zero-day detection | ⚠️ | ⚠️ | ✅ |
| Privacy preserving | ❌ | ❌ | ✅ |
| Resistant to poisoning | ❌ | ❌ | ✅ |
| Adaptive to environment | ❌ | ✅ | ✅ |

---

# 🧪 Threat Model & Defenses

FortifAI protects against:

✅ Model poisoning  
✅ Data poisoning  
✅ Evasion attacks  
✅ Byzantine clients  
✅ Malicious insiders  

Mechanisms:

- norm clipping
- robust aggregation
- DP noise
- reputation scoring
- quarantine logic

---

# 🖥 Client GUI

The client includes a user GUI showing:

- real-time anomalies
- severity scores
- contributing features
- federated updates received
- model versioning
- local training status

This transforms the agent from a silent background process into a **transparent security tool** for users.

---

# 🏅 Why This Project is Unique

Most cybersecurity projects:

- upload raw logs
- rely on signatures
- centralize detection
- do not adapt per machine
- ignore privacy

FortifAI:

✅ learns per-device behavior  
✅ shares only model knowledge  
✅ improves enterprise-wide protection  
✅ runs autonomously  
✅ resists federated attacks  
✅ provides visibility through GUI  
✅ integrates ML training on the edge

---

# 🧰 Technologies Used

- Python
- Federated Learning
- Isolation Forest (scikit-learn)
- Autoencoder (TensorFlow/Keras)
- One Class SVM
- Differential Privacy
- SQLite / Local logging
- Tkinter (GUI)
- Networking (REST/WebSockets)

---

# 🚀 Deployment

Each workstation receives:

- executable client agent
- embedded GUI
- background ML trainer

Admin server:

- aggregates models
- pushes global updates
- displays network-wide alerts

---

# ✅ Conclusion

FortifAI represents a **new class of cybersecurity system**:

- decentralized
- adaptive
- privacy-preserving
- self-improving

By combining federated learning and on-device anomaly detection, it offers **zero-trust, zero-raw-data threat detection** that becomes stronger with every client.

