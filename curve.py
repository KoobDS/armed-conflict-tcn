import json
import matplotlib.pyplot as plt
from pathlib import Path

# Path to training log
log_path = Path("TCN_results/train_losses.json")
data = json.load(open(log_path))

train = data["train"]
val   = data["val"]
epochs = range(1, len(train)+1)

plt.figure(figsize=(8,5))
plt.plot(epochs, train, label="Train Loss", linewidth=2)
plt.plot(epochs, val,   label="Val Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("curve.png")
plt.show()
