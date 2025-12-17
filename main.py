import cv2
import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import datetime
import csv

# --- КОНФИГУРАЦИЯ ---
DATASET_PATH = "dataset"
MODEL_FILE = "face_model.pkl"
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FACE_SIZE = (128, 128)

class FaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Параметры HOG (как в билете)
        self.hog = cv2.HOGDescriptor(
            _winSize=FACE_SIZE,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
        # Пытаемся загрузить модель при запуске
        self.load_model()

    def detect_and_normalize(self, image):
        """Находит лицо, делает Grayscale, Resize, EqualizeHist."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
            
        # Берем самое большое лицо
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
        face_roi = gray[y:y+h, x:x+w]
        
        # Нормализация
        face_resized = cv2.resize(face_roi, FACE_SIZE)
        face_eq = cv2.equalizeHist(face_resized)
        face_norm = face_eq.astype("float32") / 255.0
        
        return face_norm, (x, y, w, h)

    def extract_features(self, face_norm):
        """Извлекает HOG признаки."""
        face_uint8 = (face_norm * 255).astype(np.uint8)
        features = self.hog.compute(face_uint8)
        return features.flatten()

    def train(self):
        """Считывает dataset, обучает SVM и сохраняет модель."""
        print("--- Начало обучения модели ---")
        X = []
        y = []
        
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
            print(f"Создана папка {DATASET_PATH}. Заполните её фото!")
            return

        persons = os.listdir(DATASET_PATH)
        if not persons:
            print("Ошибка: Датасет пуст.")
            return

        for person_name in persons:
            person_dir = os.path.join(DATASET_PATH, person_name)
            if not os.path.isdir(person_dir): continue
            
            print(f"Обработка: {person_name}")
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                
                face, _ = self.detect_and_normalize(img)
                if face is not None:
                    feats = self.extract_features(face)
                    X.append(feats)
                    y.append(person_name)

        if len(X) == 0:
            print("Лица не найдены в датасете.")
            return

        # Обучение
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # SVM с probability=True для получения % уверенности
        self.model = SVC(kernel='linear', probability=True, C=1.0)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Сохранение
        joblib.dump({'model': self.model, 'scaler': self.scaler}, MODEL_FILE)
        print(f"Модель успешно обучена на {len(X)} фото и сохранена в {MODEL_FILE}")

    def load_model(self):
        """Загружает обученную модель с диска."""
        if os.path.exists(MODEL_FILE):
            data = joblib.load(MODEL_FILE)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True
            print("Модель загружена.")
        else:
            print("Модель не найдена. Требуется обучение.")

    def predict(self, image):
        """Возвращает: Имя, Уверенность (%), Координаты."""
        if not self.is_trained:
            return "Need Training", 0.0, None

        face, coords = self.detect_and_normalize(image)
        if face is None:
            return None, 0.0, None

        feats = self.extract_features(face).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        
        prob = self.model.predict_proba(feats_scaled)[0]
        idx = np.argmax(prob)
        confidence = prob[idx] * 100
        name = self.model.classes_[idx]
        
        return name, confidence, coords

    def register_user(self, image, name):
        """Создает папку и сохраняет фото."""
        path = os.path.join(DATASET_PATH, name)
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Считаем сколько уже есть фото, чтобы дать имя
        count = len(os.listdir(path)) + 1
        filename = os.path.join(path, f"{count}.jpg")
        
        cv2.imwrite(filename, image)
        print(f"Пользователь {name} зарегистрирован. Фото сохранено: {filename}")
        
        print("Запускаю переобучение модели...")
        self.train()

    def log_access(self, name, status):
        """Записывает событие в CSV файл (Вариант Е)."""
        log_file = "access_log.csv"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Определение заголовков, если файл создается впервые
        file_exists = os.path.isfile(log_file)
        
        try:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Name", "Status"])
                writer.writerow([now, name, status])
        except Exception as e:
            print(f"Ошибка записи в журнал: {e}")



# --- ФУНКЦИИ ИНТЕРФЕЙСА ---

def draw_result(image, name, confidence, coords):
    """Рисует рамку и возвращает данные для лога."""
    # Если лицо НЕ найдено
    if coords is None:
        return "None", "No Face Detected"

    (x, y, w, h) = coords
    BANNED_CLASSES = ["stranger", "Stranger", "unknown"]
    
    # Логика определения доступа
    is_access_granted = name and (name not in BANNED_CLASSES) and confidence > 75
    
    if is_access_granted:
        color = (0, 255, 0)
        label = f"Welcome, {name}! ({confidence:.1f}%)"
        msg = "ACCESS GRANTED"
        status = "Access Granted"
    else:
        color = (0, 0, 255)
        status = "Access Denied"
        if name in BANNED_CLASSES:
            label = "Stranger Detected"
            name = "Stranger"
        elif not name:
            label = "Unknown"
            name = "Unknown"
        else:
            label = f"Denied: {name} ({confidence:.1f}%)"
            
        msg = "ACCESS DENIED"

    # Рисование
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Возвращаем результат, если лицо было найдено
    return name, status

def run_webcam_mode(system):
    """Режим работы с веб-камерой в реальном времени."""
    print("\n--- ЗАПУСК ВЕБ-КАМЕРЫ ---")
    print("Управление:")
    print(" [q] - Выход в меню")
    print(" [r] - РЕГИСТРАЦИЯ текущего лица (Вариант D)")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Камера не найдена!")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Предсказание
        name, conf, coords = system.predict(frame)
        
        # Визуализация
        draw_result(frame, name, conf, coords)
        
        # Инструкция на экране
        cv2.putText(frame, "[q] Quit  [r] Register", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Получаем текущий статус из функции отрисовки
        current_name, current_status = draw_result(frame, name, conf, coords)
        
        cv2.imshow("Face Access System", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # ЗАПИСЬ В ЛОГ при нажатии клавиши 'l' (Log) или 'Enter' (13)
        if key == ord('l') or key == 13:
            if coords is not None:
                system.log_access(current_name, current_status)
                print(f"[LOG] Записано: {current_name} — {current_status}")
        
        # Выход
        if key == ord('q'):
            break
            
        # Регистрация (Вариант D)
        if key == ord('r'):
            if coords is None:
                print("Лицо не найдено для регистрации!")
            else:
                # Ставим на паузу видеопоток для ввода имени
                print("\n--- РЕГИСТРАЦИЯ НОВОГО ПОЛЬЗОВАТЕЛЯ ---")
                new_name = input("Введите имя нового пользователя: ").strip()
                if new_name:
                    # Вырезаем лицо с текущего кадра и сохраняем
                    # (можно сохранить весь кадр, detect_and_normalize сам обрежет)
                    system.register_user(frame, new_name)
                    print("Вернитесь в окно камеры...")

    cap.release()
    cv2.destroyAllWindows()

def run_photo_mode(system):
    """Режим проверки отдельного файла."""
    print("\n--- РЕЖИМ ФОТО ---")
    path = input("Введите путь к файлу (например, test.jpg): ").strip()
    
    if not os.path.exists(path):
        print("Файл не найден.")
        return

    image = cv2.imread(path)
    if image is None:
        print("Ошибка чтения изображения.")
        return

    name, conf, coords = system.predict(image)
    
    # Рисуем результат
    draw_result(image, name, conf, coords)
    
    # Показываем окно
    print(f"Результат: {name if name and conf > 70 else 'Неизвестный'} ({conf:.1f}%)")
    cv2.imshow("Photo Result", image)
    print("Нажмите любую клавишу, чтобы закрыть окно...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Логика регистрации для фото
    if (name is None or conf < 70) and coords is not None:
        ans = input("Лицо не распознано. Хотите зарегистрировать его? (да/нет): ").lower()
        if ans == 'да':
            new_name = input("Введите имя: ").strip()
            system.register_user(image, new_name)

# --- ГЛАВНОЕ МЕНЮ ---

def main():
    system = FaceRecognitionSystem()
    
    while True:
        print("\n==============================")
        print("   СИСТЕМА КОНТРОЛЯ ДОСТУПА   ")
        print("==============================")
        print(f"Статус модели: {'ОБУЧЕНА' if system.is_trained else 'НЕ ОБУЧЕНА'}")
        print("1. Обучить модель (на основе папки dataset)")
        print("2. Режим Веб-камеры (Real-time)")
        print("3. Режим Фото (Загрузить файл)")
        print("4. Выход")
        
        choice = input("Выберите действие (1-4): ")
        
        if choice == '1':
            system.train()
        elif choice == '2':
            if not system.is_trained:
                print("Сначала обучите модель (пункт 1)!")
            else:
                run_webcam_mode(system)
        elif choice == '3':
            if not system.is_trained:
                print("Сначала обучите модель (пункт 1)!")
            else:
                run_photo_mode(system)
        elif choice == '4':
            print("Выход...")
            break
        else:
            print("Неверный ввод.")

if __name__ == "__main__":
    main()