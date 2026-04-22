import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("=== EMOJIFY MODEL TRAIN ULTIMATE ===")

# ====================== ĐƯỜNG DẪN CHÍNH XÁC ✅ ======================
PROJECT_ROOT = r"E:/AI/project-Emojify-Create-your-own-emoji-with-Python"  
train_dir = os.path.join(PROJECT_ROOT, "Emojify", "data", "train")
test_dir = os.path.join(PROJECT_ROOT, "Emojify", "data", "test")
model_dir = os.path.join(PROJECT_ROOT, "Emojify", "model")

print(f"📁 Train: {train_dir}")
print(f"📁 Test:  {test_dir}")
print(f"💾 Model: {model_dir}")

if not os.path.exists(train_dir):
    print(f"❌ ERROR: {train_dir} không tồn tại!")
    exit()

# ====================== DATA AUGMENTATION ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ====================== LOAD DATA ======================
print("🔄 Loading dataset...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Train: {train_generator.samples} ảnh, {len(train_generator.class_indices)} classes")
print(f"✅ Test:  {test_generator.samples} ảnh")

if train_generator.samples == 0:
    print("❌ NO DATA - Check folder structure!")
    exit()

print("Classes:", train_generator.class_indices)

# ====================== MODEL ARCHITECTURE ======================
model = Sequential([
    Input(shape=(48, 48, 1)),
    
    # Block 1
    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ====================== TRAINING ======================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        os.path.join(model_dir, 'emotion_model_best.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

print("🚀 START TRAINING...")
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // 64),
    epochs=100,
    validation_data=test_generator,
    validation_steps=max(1, test_generator.samples // 64),
    callbacks=callbacks,
    verbose=1
)

# ====================== SAVE FINAL ======================
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, 'emotion_model_final.h5'))
print("🎉 DONE! Model saved. Run `py main.py` to test!")

