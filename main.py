import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Importiere deine RBM Funktionen
# (falls alles in einer Datei ist, kannst du das weglassen)
from RBM_MNIST_encoder import (
    train,
    positive_phase,
    negative_phase,
    bias_hidden_c,
    bias_visible_b
)

def main():

    # ======================
    # MNIST laden
    # ======================
    (x_train, _), (x_test, _) = mnist.load_data()

    # Normalisieren
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Flatten
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Binarisieren (wichtig für RBM)
    x_train = (x_train > 0.5).astype(np.float32)
    x_test = (x_test > 0.5).astype(np.float32)

    # ======================
    # Training
    # ======================
    mse_errors, valid_errors = train(
        x_train[:2000],
        x_test[:1000],
        num_of_epochs=40
    )

    print("Training abgeschlossen.")
    print("Letzter Train MSE:", mse_errors[-1])
    print("Letzter Valid MSE:", valid_errors[-1])

    # ======================
    # Loss Plot
    # ======================
    plt.figure()
    plt.plot(mse_errors, label="Train Error")
    plt.plot(valid_errors, label="Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("RBM Reconstruction Error")
    plt.legend()
    plt.show()

    # ======================
    # Rekonstruktionen
    # ======================
    num_samples = 10
    plt.figure(figsize=(10, 4))

    # zufällige Samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)

    for i, idx in enumerate(indices):

        v0 = x_test[idx]

        # Forward pass
        h0_prob, h0_sample = positive_phase(v0, bias_hidden_c)

        # Reconstruction
        v1_prob, _, _ = negative_phase(
            h0_sample, bias_visible_b, bias_hidden_c
        )

        # Original
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(v0.reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Original")

        # Reconstruction
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(v1_prob.reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Reconstruction")

    plt.suptitle("RBM MNIST Reconstructions")
    plt.show()


if __name__ == "__main__":
    main()