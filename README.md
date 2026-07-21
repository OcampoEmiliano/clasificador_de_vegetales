# Clasificador de Vegetales

App móvil que clasifica imágenes de vegetales usando una CNN. El usuario puede tomar una foto o seleccionar una de la galería para obtener la predicción con su nivel de confianza.

## Estructura del proyecto

```
vision por computadora/
├── backend/                  # API REST (FastAPI + PyTorch)
│   ├── main.py               # Endpoints de la API
│   ├── model.py              # Definición de la CNN (VegetableClassifier)
│   ├── classes.py            # Nombres de las 15 clases en español
│   ├── requirements.txt      # Dependencias de Python
│   └── modelo_vegetales.pth  # Pesos del modelo entrenado
├── mobile/                   # App React Native (Expo)
│   ├── App.tsx               # Componente principal con UI y lógica
│   ├── package.json          # Dependencias de Node.js
│   └── tsconfig.json         # Configuración de TypeScript
├── cnn_vegetables.ipynb      # Notebook para entrenar la CNN
├── MLP_Evaluacion.ipynb      # Notebook de evaluación (MLP anterior)
└── test_images/              # Imágenes de prueba
```

## Clases soportadas (15 vegetales)

| # | Clase |
|---|-------|
| 0 | Frijol |
| 1 | Calabaza Amarga |
| 2 | Calabaza de Botella |
| 3 | Berenjena |
| 4 | Brócoli |
| 5 | Repollo |
| 6 | Pimiento |
| 7 | Zanahoria |
| 8 | Coliflor |
| 9 | Pepino |
| 10 | Papaya |
| 11 | Papa |
| 12 | Calabaza |
| 13 | Rábano |
| 14 | Tomate |

## Requisitos

- **Python** 3.10+
- **Node.js** 18+
- **pnpm** (para el mobile)
- **Expo Go** (en el teléfono) o emulador Android/iOS

## Instalación y ejecución

### Backend

```bash
cd backend

# Crear entorno virtual (opcional si no existe)
python -m venv env
.\env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

La API queda disponible en `http://<IP>:8000`.

### Mobile

```bash
cd mobile

# Instalar dependencias
pnpm install

# Iniciar Expo
pnpm expo start
```

Escanea el QR con **Expo Go** (iOS/Android) o abre en emulador.

### Conectar mobile con backend

En `mobile/App.tsx`, actualiza `API_URL` con la IP de tu computadora:

```typescript
const API_URL = "http://192.168.1.X:8000";
```

Asegúrate de que el teléfono y la computadora estén en la **misma red WiFi**.

## API REST

### `POST /predict`

Envía una imagen y recibe la clasificación.

**Request**: `multipart/form-data` con campo `file` (imagen JPEG/PNG).

**Response** (confianza ≥ 50%):
```json
{
  "class": "Papa",
  "confidence": 0.9532,
  "probabilities": {
    "Frijol": 0.0001,
    "Papa": 0.9532,
    ...
  }
}
```

**Response** (confianza < 50%):
```json
{
  "class": "No reconocido",
  "confidence": 0.2341,
  "message": "La imagen no pertenece a ninguna clase conocida"
}
```

## Entrenamiento del modelo

1. Abre `cnn_vegetables.ipynb` en [Google Colab](https://colab.research.google.com/)
2. Monta Google Drive y coloca el dataset en `/content/drive/MyDrive/Vegetables_dataset/`
3. El notebook copia el dataset a `/tmp/` (I/O más rápido) y entrena con:
   - **Arquitectura**: CNN con 5 capas convolucionales + 3 capas fully connected
   - **Optimizador**: Adam con weight decay (1e-4)
   - **Scheduler**: ReduceLROnPlateau (reduce LR si la loss no mejora en 3 épocas)
   - **Early stopping**: Detiene si test accuracy no mejora en 5 épocas
   - **Batch size**: 32 (train), 64 (test)
4. Descarga el `modelo_vegetales.pth` generado y colócalo en `backend/`
