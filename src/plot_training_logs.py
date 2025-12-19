import re
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURATION
# ----------------------------
LOG_FILE = "new_prog/b.txt"      # Change if your log file has a different name
SAVE_LOSS_PLOT = "loss_curve_1.png"
SAVE_METRICS_PLOT = "test_metrics_curve_1.png"

# ----------------------------
# LOAD LOG FILE
# ----------------------------
print("Loading log file:", LOG_FILE)
with open(LOG_FILE, "r") as f:
    text = f.read()

# ----------------------------
# PARSE TRAINING METRICS
# ----------------------------

epochs = list(map(int, re.findall(r"epoch:\s*(\d+)", text)))
train_loss = list(map(float, re.findall(r"train loss:\s*([0-9.]+)", text)))

auc = list(map(float, re.findall(r"test auc:\s*([0-9.]+)", text)))
aupr = list(map(float, re.findall(r"test aupr:\s*([0-9.]+)", text)))
f1 = list(map(float, re.findall(r"test f1:\s*([0-9.]+)", text)))
acc = list(map(float, re.findall(r"test acc:\s*([0-9.]+)", text)))

print(f"Parsed {len(epochs)} epochs.")

# ----------------------------
# PLOT 1: TRAIN LOSS
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss Over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_LOSS_PLOT)
plt.close()
print(f"Saved loss plot to {SAVE_LOSS_PLOT}")

# ----------------------------
# PLOT 2: TEST METRICS
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs, auc, label="AUC")
plt.plot(epochs, aupr, label="AUPR")
plt.plot(epochs, f1, label="F1 Score")
plt.plot(epochs, acc, label="Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Test Metrics Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_METRICS_PLOT)
plt.close()
print(f"Saved metrics plot to {SAVE_METRICS_PLOT}")

print("\nDone! Check the generated PNG files.\n")