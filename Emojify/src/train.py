import tensorflow as tf
import os

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

# ====================== XỬ LÝ ĐƯỜNG DẪN ======================
base_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.normpath(os.path.join(base_dir, '..', 'data', 'train'))
test_dir  = os.path.normpath(os.path.join(base_dir, '..', 'data', 'test'))

print(f"--- Dang quet train data tai: {train_dir}")
print(f"--- Dang quet test data tai : {test_dir}")

# ====================== DATA GENERATOR ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load dữ liệu
train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Kiểm tra có dữ liệu không
if train_generator.samples == 0:
    print("Loi: Khong tim thay anh nao trong thu muc train!")
    print("Vui long kiem tra lai duong dan data/train")
    exit()

print(f"Tim thay {train_generator.samples} anh train va {test_generator.samples} anh test")

# ====================== XÂY DỰNG MÔ HÌNH ======================
model = Sequential([
    Input(shape=(48, 48, 1)),
    
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# ====================== CALLBACKS ======================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(base_dir, '..', 'model', 'emotion_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ====================== TRAINING ======================
print("Bat đau huan luyen...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=50,
    validation_data=test_generator,
    validation_steps=max(1, test_generator.samples // 64),
    callbacks=callbacks,
    verbose=1
)

# ====================== LƯU MODEL ======================
model_dir = os.path.join(base_dir, '..', 'model')
os.makedirs(model_dir, exist_ok=True)

final_model_path = os.path.join(model_dir, 'emotion_model_final.h5')
model.save(final_model_path)

print(f"Huan luyen hoan tat!")
print(f"Model cuoi cung đa luu tai: {final_model_path}")
print(f"Model tot nhat đa luu tai: {callbacks[1].filepath}")