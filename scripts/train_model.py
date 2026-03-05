import argparse
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

def create_1d_cnn(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
    ])
    loss = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs('/content/Project/models', exist_ok=True)

    with open(os.path.join(args.dataset_dir, 'breathing_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    X      = dataset['X']          # (N, 960, 3)
    y_raw  = dataset['y']          # string labels
    groups = dataset['groups']

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    logo = LeaveOneGroupOut()
    results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        participant = groups[test_idx[0]]
        print(f"\n--- Fold {fold+1}: Leaving out Participant {participant} ---")
        print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        print(f"Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test  label dist: {dict(zip(*np.unique(y_test,  return_counts=True)))}")

        # Compute class weights to handle imbalance
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight = {i: w for i, w in enumerate(weights)}
        print(f"Class weights: {class_weight}")

        model = create_1d_cnn((X.shape[1], X.shape[2]), n_classes)

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.1,
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=0
        )

        if n_classes == 2:
            y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        results.append({'fold': fold+1, 'participant': participant,
                        'accuracy': acc, 'precision': prec, 'recall': rec})

        model.save(f'/content/Project/models/cnn_fold_{fold+1}.keras')

    # Summary across all folds
    df = pd.DataFrame(results)
    print("\n===== Cross-Validation Summary =====")
    print(df.to_string(index=False))
    print(f"\nMean Accuracy:  {df['accuracy'].mean():.4f}")
    print(f"Mean Precision: {df['precision'].mean():.4f}")
    print(f"Mean Recall:    {df['recall'].mean():.4f}")
    df.to_csv('/content/Project/models/cv_results.csv', index=False)

if __name__ == "__main__":
    main()
